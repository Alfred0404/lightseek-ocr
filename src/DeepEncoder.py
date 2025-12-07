"""
DeepEncoder
Complete LightSeek-OCR encoding pipeline:
1. Text to image rendering
2. Image to SAM feature extraction (local features)
3. Feature map compression
4. CLIP vision encoder processing (global features)
"""

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import infer_device
import torch.nn as nn


from utils.colors import bcolors

from SAMFeatureExtractor import SAMFeatureExtractor
from CLIPVisionProcessor import CLIPVisionProcessor
from Conv2DCompressor import Conv2DCompressor


class DeepEncoder:
    """
    End-to-end encoder for LightSeek-OCR

    Transforms text into visual features ready for OCR decoding:
    - Local features (SAM): Fine-grained spatial details (B, 256, 64, 64)
    - Global features (CLIP): Semantic token sequence (B, N, 768)
    """

    def __init__(
        self,
        sam_model_name="facebook/sam-vit-base",
        clip_model_name="openai/clip-vit-base-patch32",
        device=None,
        verbose=True,
    ):
        """
        Initialize the DeepEncoder pipeline

        Args:
            sam_model_name: HuggingFace model name for SAM
            clip_model_name: HuggingFace model name for CLIP
            device: torch device (auto-detected if None)
            verbose: Print progress information
        """
        self.device = device if device is not None else infer_device()
        self.verbose = verbose

        if self.verbose:
            print("=" * 70)
            print(bcolors.BOLD + "Initializing DeepEncoder" + bcolors.ENDC)
            print("=" * 70)
            print(f"{bcolors.OKCYAN}Device: {self.device}{bcolors.ENDC}\n")

        # Initialize components
        self.sam_extractor = SAMFeatureExtractor(
            model_name=sam_model_name, device=self.device
        )
        self.clip_processor = CLIPVisionProcessor(
            model_name=clip_model_name, device=self.device
        )

        # Initialize compressor (256 → 1024 channels as per paper)
        self.compressor = Conv2DCompressor(
            in_channels=256,  # SAM output channels
            out_channels=1024,  # As specified in paper
        ).to(self.device)

        # Add projection layer to convert 1024 → 768 for CLIP compatibility
        self.channel_projection = nn.Conv2d(
            1024,  # Compressor output
            self.clip_processor.vision_hidden_size,  # CLIP expects 768
            kernel_size=1,  # 1×1 conv for channel reduction only
            bias=False,
        ).to(self.device)

        if self.verbose:
            print(
                f"\n{bcolors.OKGREEN}DeepEncoder initialized successfully!{bcolors.ENDC}\n"
            )

    def text_to_image(
        self,
        text: str,
        image_size=(1024, 1024),
        bg_color=(255, 255, 255),
        text_color=(0, 0, 0),
        font_size=24,
        image_path=None,
    ) -> Image.Image:
        """
        Render text onto an image

        Args:
            text: Text to render
            image_size: (width, height) tuple
            bg_color: Background RGB color
            text_color: Text RGB color
            font_size: Font size in points

        Returns:
            PIL Image with rendered text
        """
        image = Image.new("RGB", image_size, color=bg_color)
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except IOError:
            font = ImageFont.load_default()

        text_position = (10, 10)
        draw.text(text_position, text, fill=text_color, font=font)

        # for testing purpose
        image_path = "image.png"
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")

        return image

    def encode(self, text: str) -> dict:
        """
        Run the complete encoding pipeline: text → visual features

        Args:
            text: Input text to encode

        Returns:
            dict with keys:
                - image: Rendered PIL Image
                - local_features: SAM features (B, 256, 64, 64)
                - global_features: CLIP sequence (B, N, 768)
                - compressed_features: Intermediate compressed map (B, 768, H, W)
        """
        if self.verbose:
            print("=" * 70)
            print(f"{bcolors.HEADER}DeepEncoder Pipeline{bcolors.ENDC}")
            print("=" * 70)
            print(f"Input text: {text}\n")

        # Text to Image
        if self.verbose:
            print(f"{bcolors.OKBLUE}[1/4] Rendering text to image...{bcolors.ENDC}")
        pil_image = self.text_to_image(text)
        if self.verbose:
            print(f"  Image size: {pil_image.size}\n")

        features = self.extract_features(pil_image)

        return {
            "image": pil_image,
            "local_features": features["local_features"],
            "global_features": features["global_features"],
            "compressed_features": features["compressed_features"],
        }

    def extract_features(self, pil_image: Image.Image) -> dict:
        """
        Extract visual features from an image (without text rendering step).

        Args:
            pil_image: Input PIL Image

        Returns:
            dict with keys: local_features, global_features, compressed_features
        """
        # SAM Feature Extraction (Local Features)
        if self.verbose:
            print(
                f"{bcolors.OKBLUE}[2/4] Extracting SAM features (local)...{bcolors.ENDC}"
            )
        local_features = self.sam_extractor.extract(pil_image)
        if self.verbose:
            print(f"  Local features shape: {local_features.shape}\n")

        # Feature Compression
        if self.verbose:
            print(f"{bcolors.OKBLUE}[3/4] Compressing features...{bcolors.ENDC}")
        compressed_map = self.compressor(local_features)
        if self.verbose:
            print(f"  After compressor: {compressed_map.shape}")

        # Project channels from 1024 → 768 for CLIP
        compressed_map = self.channel_projection(compressed_map)
        if self.verbose:
            print(f"  After projection: {compressed_map.shape}\n")

        # CLIP Processing (Global Features)
        if self.verbose:
            print(
                f"{bcolors.OKBLUE}[4/4] Processing through CLIP (global)...{bcolors.ENDC}"
            )
        global_features = self.clip_processor.process_compressed_features(
            compressed_map
        )
        if self.verbose:
            print(f"  Global features shape: {global_features.shape}\n")

        return {
            "local_features": local_features,
            "compressed_features": compressed_map,
            "global_features": global_features,
        }

    def visualize_spatial_aggregation(
        self,
        local_features: torch.Tensor,
        compressed_features: torch.Tensor,
        original_image,
        save_path: str = "res/spatial_aggregation.png",
    ):
        """
        Visualize multi-scale spatial aggregation

        Args:
            local_features: SAM local features (B, 256, 64, 64) or (256, 64, 64)
            compressed_features: Compressed features (B, C, H, W) or (C, H, W)
            original_image: PIL Image
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Remove batch dimension if present
        if local_features.ndim == 4:
            local_features = local_features[0]
        if compressed_features.ndim == 4:
            compressed_features = compressed_features[0]

        local_np = local_features.detach().cpu().numpy()
        compressed_np = compressed_features.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Multi-Scale Feature Aggregation", fontsize=16, fontweight="bold")

        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Input Image\n1024×1024", fontweight="bold")
        axes[0].axis("off")

        # SAM local features (aggregate across channels)
        local_aggregated = np.mean(local_np, axis=0)  # (H, W)
        local_norm = (local_aggregated - local_aggregated.min()) / (
            local_aggregated.max() - local_aggregated.min() + 1e-8
        )

        im1 = axes[1].imshow(local_norm, cmap="plasma", interpolation="nearest")
        axes[1].set_title(
            f"SAM Local Features\n{local_np.shape[1]}×{local_np.shape[2]} spatial grid\n{local_np.shape[0]} channels",
            fontweight="bold",
        )
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Compressed tokens (aggregate across channels)
        compressed_aggregated = np.mean(compressed_np, axis=0)  # (H, W)
        compressed_norm = (compressed_aggregated - compressed_aggregated.min()) / (
            compressed_aggregated.max() - compressed_aggregated.min() + 1e-8
        )

        im2 = axes[2].imshow(compressed_norm, cmap="inferno", interpolation="nearest")
        h, w = compressed_np.shape[1], compressed_np.shape[2]
        axes[2].set_title(
            f"Compressed Tokens\n{h}×{w} spatial grid ({h*w} tokens)\n{compressed_np.shape[0]} channels",
            fontweight="bold",
        )
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Add grid to compressed visualization
        for i in range(h + 1):
            axes[2].axhline(i - 0.5, color="white", linewidth=0, alpha=0.5)
        for i in range(w + 1):
            axes[2].axvline(i - 0.5, color="white", linewidth=0, alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(
            f"{bcolors.OKGREEN}✓ Spatial aggregation saved to: {save_path}{bcolors.ENDC}"
        )
        plt.show()


if __name__ == "__main__":
    # Example usage
    sample_text = "A scenic view of mountains during sunrise"

    # Initialize encoder
    encoder = DeepEncoder(verbose=True)

    # Encode text to visual features
    results = encoder.encode(sample_text)

    # Access outputs
    print("\nOutput Summary:")
    print(f"  Local features (SAM):   {results['local_features'].shape}")
    print(f"  Global features (CLIP): {results['global_features'].shape}")
    print(f"\nSample global features (first 5 tokens, first 5 dims):")
    print(results["global_features"][0, :5, :5])

    # Save the rendered image
    results["image"].save("output_image.png")
    print(f"\n{bcolors.OKGREEN}Rendered image saved to: output_image.png{bcolors.ENDC}")

    # Visualize SAM local features using SAMFeatureExtractor method
    print("\n" + "=" * 70)
    print(f"{bcolors.HEADER}VISUALIZING SAM LOCAL FEATURES{bcolors.ENDC}")
    print("=" * 70)
    encoder.sam_extractor.visualize_all_channels(results["local_features"])

    # Visualize compressed tokens using Conv2DCompressor method
    print("\n" + "=" * 70)
    print(f"{bcolors.HEADER}VISUALIZING COMPRESSED TOKENS{bcolors.ENDC}")
    print("=" * 70)
    encoder.compressor.visualize_compressed_tokens(
        results["compressed_features"], original_image=results["image"]
    )

    # Visualize spatial aggregation using DeepEncoder method
    print("\n" + "=" * 70)
    print(f"{bcolors.HEADER}VISUALIZING SPATIAL AGGREGATION{bcolors.ENDC}")
    print("=" * 70)
    encoder.visualize_spatial_aggregation(
        results["local_features"],
        results["compressed_features"],
        results["image"],
    )

    print("\n" + "=" * 70)
    print(f"{bcolors.OKGREEN}VISUALIZATION COMPLETE{bcolors.ENDC}")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. output_image.png - Rendered text image")
    print("  2. res/sam_local_features.png - 256 SAM feature channels")
    print("  3. res/compressed_tokens.png - Sample compressed channels")
    print("  4. res/spatial_aggregation.png - Multi-scale overview")
    print("=" * 70 + "\n")
