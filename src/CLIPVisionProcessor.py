"""
CLIP Vision Processor
Processes compressed features through CLIP vision encoder, bypassing embedding layer
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, infer_device

from utils.colors import bcolors


class CLIPVisionProcessor:
    """Process compressed features through CLIP vision encoder"""

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device if device is not None else infer_device()
        print(
            f"{bcolors.OKCYAN}Loading CLIP model on device: {self.device}{bcolors.ENDC}"
        )

        # https://huggingface.co/openai/clip-vit-base-patch32
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.vision_model = self.model.vision_model
        self.vision_hidden_size = self.model.config.vision_config.hidden_size
        print(
            f"{bcolors.OKCYAN}CLIP vision hidden size: {self.vision_hidden_size}{bcolors.ENDC}"
        )

    def _interpolate_pos_embedding(
        self, pos_embed: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings to match the number of tokens

        Args:
            pos_embed: Original position embedding (num_pos, D)
            num_tokens: Target number of tokens (N)

        Returns:
            Interpolated position embedding (N, D)
        """
        num_orig_pos = pos_embed.shape[0]
        embed_dim = pos_embed.shape[1]

        if num_tokens == num_orig_pos:
            return pos_embed

        # Reshape to (1, D, num_orig_pos) for interpolation
        pos_embed_reshaped = pos_embed.unsqueeze(0).permute(0, 2, 1)

        # Interpolate to target length
        interpolated = F.interpolate(
            pos_embed_reshaped, size=num_tokens, mode="linear", align_corners=False
        )

        # Reshape back to (num_tokens, D)
        interpolated = interpolated.permute(0, 2, 1).squeeze(0)

        return interpolated

    def process_compressed_features(self, compressed_map: torch.Tensor) -> torch.Tensor:
        """
        Process compressed feature map through CLIP vision encoder,
        bypassing the patch embedding layer.

        Returns the FULL sequence of tokens (NO CLS pooling), which serves as
        the global visual features for the decoder.

        Args:
            compressed_map: Compressed feature map (B, D, H, W)

        Returns:
            Global visual features: sequence of tokens (B, N, D) where N = H*W
        """
        batch_size = compressed_map.shape[0]

        # Convert to tokens (B, N, D) where N = H*W
        tokens = compressed_map.flatten(2).permute(0, 2, 1)
        num_tokens = tokens.shape[1]

        print(f"  Input tokens shape: {tokens.shape}")

        # Get positional embeddings from CLIP
        position_embedding = self.vision_model.embeddings.position_embedding.weight
        expected_num_pos = position_embedding.shape[0]

        # Interpolate positional embeddings to match our token count
        if num_tokens != expected_num_pos:
            print(f"  Interpolating pos embeddings: {expected_num_pos} -> {num_tokens}")
            pos_embed_interpolated = self._interpolate_pos_embedding(
                position_embedding, num_tokens
            )
        else:
            pos_embed_interpolated = position_embedding

        # Add positional embeddings (no CLS token prepended)
        tokens_with_pos = tokens + pos_embed_interpolated.unsqueeze(0)

        # Apply pre-LayerNorm
        tokens_with_pos = self.vision_model.pre_layrnorm(tokens_with_pos)

        # Pass through encoder
        # REMOVED torch.no_grad() to allow gradients to flow back to compressor!
        encoder_outputs = self.vision_model.encoder(inputs_embeds=tokens_with_pos)
        final_hidden_states = encoder_outputs[0]

        # Return the full sequence (global features for OCR decoder)
        return final_hidden_states
