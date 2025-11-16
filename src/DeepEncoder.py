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

from colors import bcolors

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

        # Initialize compressor to match CLIP's expected dimension
        self.compressor = Conv2DCompressor(
            in_channels=256,  # SAM output channels
            out_channels=self.clip_processor.vision_hidden_size,  # CLIP vision hidden size (768)
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

        # Step 1: Text to Image
        if self.verbose:
            print(f"{bcolors.OKBLUE}[1/4] Rendering text to image...{bcolors.ENDC}")
        pil_image = self.text_to_image(text)
        if self.verbose:
            print(f"  Image size: {pil_image.size}\n")

        # Step 2: SAM Feature Extraction (Local Features)
        if self.verbose:
            print(
                f"{bcolors.OKBLUE}[2/4] Extracting SAM features (local)...{bcolors.ENDC}"
            )
        local_features = self.sam_extractor.extract(pil_image)
        if self.verbose:
            print(f"  Local features shape: {local_features.shape}\n")

        # Step 3: Feature Compression
        if self.verbose:
            print(f"{bcolors.OKBLUE}[3/4] Compressing features...{bcolors.ENDC}")
        compressed_map = self.compressor(local_features)
        if self.verbose:
            print(f"  Compressed map shape: {compressed_map.shape}\n")

        # Step 4: CLIP Processing (Global Features)
        if self.verbose:
            print(
                f"{bcolors.OKBLUE}[4/4] Processing through CLIP (global)...{bcolors.ENDC}"
            )
        global_features = self.clip_processor.process_compressed_features(
            compressed_map
        )
        if self.verbose:
            print(f"  Global features shape: {global_features.shape}\n")

        if self.verbose:
            print("=" * 70)
            print(f"{bcolors.OKGREEN}Encoding completed successfully!{bcolors.ENDC}")
            print("=" * 70)
            print("\nVisual Features for Decoder:")
            print(
                f"  - Local features (SAM):   {local_features.shape}  # Fine-grained spatial"
            )
            print(
                f"  - Global features (CLIP): {global_features.shape}  # Semantic tokens"
            )
            print()

        return {
            "image": pil_image,
            "local_features": local_features,
            "global_features": global_features,
            "compressed_features": compressed_map,
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

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

    # Visualize SAM local features
    print("\n" + "=" * 70)
    print(f"{bcolors.HEADER}VISUALIZING SAM LOCAL FEATURES{bcolors.ENDC}")
    print("=" * 70)

    local_features = (
        results["local_features"][0].detach().cpu().numpy()
    )  # (256, 64, 64)

    # Display all SAM channels
    num_channels = local_features.shape[0]  # e.g., 256
    n_cols = 16
    n_rows = (num_channels + n_cols - 1) // n_cols
    figsize = (n_cols * 1.2, n_rows * 1.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle("SAM Local Features - All Channels", fontsize=16, fontweight="bold")

    axes = axes.flatten()
    for idx in range(n_rows * n_cols):
        ax = axes[idx]
        if idx < num_channels:
            channel_data = local_features[idx]

            # Normalize for visualization
            vmin, vmax = channel_data.min(), channel_data.max()
            normalized = (channel_data - vmin) / (vmax - vmin + 1e-8)

            ax.imshow(normalized, cmap="viridis", interpolation="nearest")
            ax.set_title(f"Ch {idx}", fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("res/sam_local_features.png", dpi=150, bbox_inches="tight")
    print(f"{bcolors.OKGREEN}✓ SAM local features saved to: res/sam_local_features.png{bcolors.ENDC}")
    plt.show()

    # Visualize compressed tokens
    print("\n" + "=" * 70)
    print(f"{bcolors.HEADER}VISUALIZING COMPRESSED TOKENS{bcolors.ENDC}")
    print("=" * 70)

    compressed = results["compressed_features"][0].detach().cpu().numpy()  # (768, 4, 4)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Compressed Feature Maps - Sample Channels", fontsize=16, fontweight="bold"
    )

    # Original image
    axes[0, 0].imshow(results["image"])
    axes[0, 0].set_title("Original Image", fontweight="bold")
    axes[0, 0].axis("off")

    # Show 5 compressed channels
    channels_to_show = [0, 150, 300, 450, 600]

    for idx, channel_idx in enumerate(channels_to_show):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        ax = axes[row, col]

        channel_data = compressed[channel_idx]

        # Normalize for visualization
        vmin, vmax = channel_data.min(), channel_data.max()
        normalized = (channel_data - vmin) / (vmax - vmin + 1e-8)

        im = ax.imshow(normalized, cmap="hot", interpolation="nearest")
        ax.set_title(f"Compressed Ch {channel_idx}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add grid to show 4x4 structure
        for i in range(5):
            ax.axhline(i - 0.5, color="white", linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color="white", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig("res/compressed_tokens.png", dpi=150, bbox_inches="tight")
    print(f"{bcolors.OKGREEN}✓ Compressed tokens saved to: res/compressed_tokens.png{bcolors.ENDC}")
    plt.show()

    # Visualize spatial aggregation
    print("\n" + "=" * 70)
    print(f"{bcolors.HEADER}VISUALIZING SPATIAL AGGREGATION{bcolors.ENDC}")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Multi-Scale Feature Aggregation", fontsize=16, fontweight="bold")

    # Original image
    axes[0].imshow(results["image"])
    axes[0].set_title("Input Image\n1024×1024", fontweight="bold")
    axes[0].axis("off")

    # SAM local features (aggregate across channels)
    local_aggregated = np.mean(local_features, axis=0)  # (64, 64)
    local_norm = (local_aggregated - local_aggregated.min()) / (
        local_aggregated.max() - local_aggregated.min() + 1e-8
    )

    im1 = axes[1].imshow(local_norm, cmap="plasma", interpolation="nearest")
    axes[1].set_title(
        "SAM Local Features\n64×64 spatial grid\n256 channels", fontweight="bold"
    )
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Compressed tokens (aggregate across channels)
    compressed_aggregated = np.mean(compressed, axis=0)  # (4, 4)
    compressed_norm = (compressed_aggregated - compressed_aggregated.min()) / (
        compressed_aggregated.max() - compressed_aggregated.min() + 1e-8
    )

    im2 = axes[2].imshow(compressed_norm, cmap="inferno", interpolation="nearest")
    axes[2].set_title(
        "Compressed Tokens\n4×4 spatial grid (16 tokens)\n768 channels",
        fontweight="bold",
    )
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Add grid
    for i in range(5):
        axes[2].axhline(i - 0.5, color="white", linewidth=2, alpha=0.5)
        axes[2].axvline(i - 0.5, color="white", linewidth=2, alpha=0.5)

    plt.tight_layout()
    plt.savefig("res/spatial_aggregation.png", dpi=150, bbox_inches="tight")
    print(
        f"{bcolors.OKGREEN}✓ Spatial aggregation saved to: res/spatial_aggregation.png{bcolors.ENDC}"
    )
    plt.show()

    print("\n" + "=" * 70)
    print(f"{bcolors.OKGREEN}VISUALIZATION COMPLETE{bcolors.ENDC}")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. output_image.png - Rendered text image")
    print("  2. sam_local_features.png - 12 SAM feature channels")
    print("  3. compressed_tokens.png - 5 compressed channels")
    print("  4. spatial_aggregation.png - Multi-scale overview")
    print("=" * 70 + "\n")
