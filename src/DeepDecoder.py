"""
DeepDecoder
Decodes visual features into text using a pretrained GPT-2 model.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, infer_device

from utils.colors import bcolors


class DeepDecoder(nn.Module):
    """
    Autoregressive Decoder for LightSeek-OCR.
    Wraps a pretrained GPT-2 model and adapts it for visual-conditioned generation.
    """

    def __init__(self, model_name="gpt2", device=None, verbose=True):
        """
        Initialize the DeepDecoder.

        Args:
            model_name: HuggingFace model name (default: "gpt2")
            device: torch device
            verbose: Print initialization details
        """
        super().__init__()
        self.device = device if device is not None else infer_device()
        self.verbose = verbose

        if self.verbose:
            print(
                f"{bcolors.OKCYAN}Loading DeepDecoder (Base: {model_name})...{bcolors.ENDC}"
            )

        # Load Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # GPT-2 doesn't have a pad token by default, use eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Model
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.hidden_size = self.model.config.n_embd  # 768 for GPT-2 small

        # Visual Projection Layer
        # Projects visual features to the exact embedding space of the LLM
        # Even if dims match (768->768), this layer helps align the feature distributions
        # Upgraded to MLP for better capacity
        self.visual_projection = nn.Sequential(
            nn.Linear(768, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        ).to(self.device)

        if self.verbose:
            print(
                f"{bcolors.OKGREEN}DeepDecoder initialized on {self.device}{bcolors.ENDC}"
            )
            print(f"  - Vocab Size: {self.model.config.vocab_size}")
            print(f"  - Hidden Size: {self.hidden_size}")

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

        Args:
            local_features: SAM features (B, 256, 768)
            global_features: CLIP features (B, 256, 768)
            text_input_ids: Tokenized text (B, Seq_Len)
            text_attention_mask: Attention mask for text (B, Seq_Len)
        """

        # 1. Project Visual Features
        # Concatenate local and global features: (B, 512, 768)
        visual_features = torch.cat([local_features, global_features], dim=1)
        visual_embeds = self.visual_projection(visual_features)  # (B, 512, Hidden)

        # 2. Prepare Text Embeddings
        if text_input_ids is not None:
            # Get text embeddings from GPT-2's embedding layer
            wte = self.model.transformer.wte
            text_embeds = wte(text_input_ids)  # (B, Seq_Len, Hidden)

            # 3. Concatenate: [Visual, Text]
            # New sequence length = 512 + Seq_Len
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

            # 4. Create Attention Mask
            # Visual tokens are always attended to (1)
            batch_size = visual_features.shape[0]
            visual_mask = torch.ones(
                (batch_size, visual_features.shape[1]),
                dtype=torch.long,
                device=self.device,
            )

            if text_attention_mask is not None:
                attention_mask = torch.cat([visual_mask, text_attention_mask], dim=1)
            else:
                attention_mask = None

            # 5. Forward through GPT-2
            # We use inputs_embeds instead of input_ids
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,  # Pass labels to compute loss
            )

            return outputs

    def decode(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor,
        max_new_tokens=50,
        temperature=0.7,
    ):
        """
        Decode visual features into text.
        """
        self.model.eval()

        # 1. Prepare Visual Context
        visual_features = torch.cat([local_features, global_features], dim=1)
        visual_embeds = self.visual_projection(visual_features)  # (B, 512, Hidden)

        batch_size = visual_features.shape[0]

        # 2. Start with [BOS] (or just start generation if model allows)
        # GPT-2 doesn't have a standard BOS, but we can start with a prompt or empty
        # Here we assume we start generating from scratch.
        # We need to feed visual_embeds as 'past_key_values' or prefix.
        # Simpler approach: Feed visual_embeds as the initial input sequence.

        generated_ids = []

        # Initial forward pass with visual features only
        # We treat visual features as the "prompt" embeddings
        outputs = self.model(inputs_embeds=visual_embeds)
        past_key_values = outputs.past_key_values

        # Get the last token's logits to predict the first text token
        next_token_logits = outputs.logits[:, -1, :]

        # Greedy or Sample
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated_ids.append(next_token)

        # Autoregressive loop
        current_input_ids = next_token

        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=current_input_ids, past_key_values=past_key_values
            )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :] / temperature

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids.append(next_token)
            current_input_ids = next_token

            # Stop if EOS (if we had one defined, GPT-2 uses EOS for PAD usually)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Concatenate all generated tokens
        generated_ids = torch.cat(generated_ids, dim=1)

        # Decode
        decoded_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        return decoded_text


if __name__ == "__main__":
    # Test
    decoder = DeepDecoder()

    # Dummy features
    B = 1
    local_f = torch.randn(B, 256, 768).to(decoder.device)
    global_f = torch.randn(B, 256, 768).to(decoder.device)

    print("Generating...")
    text = decoder.decode(local_f, global_f)
    print(f"Generated: {text}")
