"""
DeepDecoder
Decodes visual features into text using SmolLM2-1.7B-Instruct + LoRA.

Architecture:
  [local_features (B,256,768) | global_features (B,256,768)]
      → visual_projection MLP (768 → 2048)
      → [vision_tokens×512 | prompt_tokens | transcription_tokens]
      → SmolLM2-1.7B (base frozen, LoRA on q/k/v/o)
      → transcription
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from utils.colors import bcolors

_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

_DEFAULT_PROMPT = "Transcribe the text visible in this document."


class DeepDecoder(nn.Module):
    """
    Autoregressive Decoder for LightSeek-OCR.
    Wraps SmolLM2-1.7B-Instruct with LoRA and a visual projection MLP.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device=None,
        verbose: bool = True,
        vision_hidden_size: int = 768,
    ):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        if self.verbose:
            print(f"{bcolors.OKCYAN}Loading DeepDecoder ({model_name})...{bcolors.ENDC}")

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- LLM: load in fp16, then wrap with LoRA ---
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.model = get_peft_model(base_model, _LORA_CONFIG)
        self.model = self.model.to(self.device)

        self.hidden_size = self.model.config.hidden_size  # 2048 for SmolLM2-1.7B

        # --- Visual Projection MLP: float32 for training stability ---
        # Projects concatenated local+global features (B, 512, vision_hidden_size) → (B, 512, hidden_size)
        self.visual_projection = nn.Sequential(
            nn.Linear(vision_hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        ).to(self.device)

        if self.verbose:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"{bcolors.OKGREEN}DeepDecoder ready on {self.device}{bcolors.ENDC}")
            print(f"  - LLM hidden size : {self.hidden_size}")
            print(f"  - LoRA trainable  : {trainable:,} / {total:,} ({trainable / total:.2%})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token embeddings (fp16) from the base model's embedding table."""
        return self.model.get_input_embeddings()(input_ids)

    def _project_visual(self, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Concatenate and project visual features.
        Input : (B, 256, 768) + (B, 256, 768)
        Output: (B, 512, 2048) in fp16
        """
        visual_cat = torch.cat([local_features, global_features], dim=1)  # (B, 512, 768)
        projected = self.visual_projection(visual_cat.float())             # (B, 512, 2048) fp32
        return projected.to(torch.float16)                                 # cast to fp16 for LLM

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        text_input_ids: torch.Tensor = None,
        text_attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass for training.

        Sequence layout:
          inputs_embeds : [visual×512 | text (prompt + transcription)]
          labels        : [-100×512   | -100×N_prompt | transcription_ids]

        Args:
            local_features     : (B, 256, 768) — SAM features via compressor
            global_features    : (B, 256, 768) — CLIP features
            text_input_ids     : (B, N_text)   — prompt + transcription token ids
            text_attention_mask: (B, N_text)   — ones for all text tokens
            labels             : (B, 512+N_text) — -100 for visual+prompt, ids for transcription
        """
        visual_embeds = self._project_visual(local_features, global_features)  # (B, 512, 2048)
        B, N_vis, _ = visual_embeds.shape
        visual_mask = torch.ones((B, N_vis), dtype=torch.long, device=self.device)

        if text_input_ids is not None:
            text_embeds = self._embed_tokens(text_input_ids)  # (B, N_text, 2048) fp16
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            if text_attention_mask is not None:
                attention_mask = torch.cat([visual_mask, text_attention_mask], dim=1)
            else:
                attention_mask = torch.cat(
                    [visual_mask, torch.ones((B, text_input_ids.shape[1]), dtype=torch.long, device=self.device)],
                    dim=1,
                )
        else:
            inputs_embeds = visual_embeds
            attention_mask = visual_mask

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decode(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        prompt: str = _DEFAULT_PROMPT,
        max_new_tokens: int = 200,
    ) -> list[str]:
        """
        Generate transcription from visual features.

        Returns a list of decoded strings (one per batch element).
        """
        self.model.eval()
        with torch.no_grad():
            visual_embeds = self._project_visual(local_features, global_features)
            B, N_vis, _ = visual_embeds.shape

            # Tokenise and embed the prompt
            prompt_ids = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)
            prompt_embeds = self._embed_tokens(prompt_ids.expand(B, -1))  # (B, N_p, 2048)

            # Prefix: [visual | prompt]
            inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1)
            attention_mask = torch.ones(
                (B, inputs_embeds.shape[1]), dtype=torch.long, device=self.device
            )

            generated = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    decoder = DeepDecoder()
    B = 1
    local_f = torch.randn(B, 256, 768).to(decoder.device)
    global_f = torch.randn(B, 256, 768).to(decoder.device)
    print("Generating...")
    text = decoder.decode(local_f, global_f)
    print(f"Generated: {text}")
