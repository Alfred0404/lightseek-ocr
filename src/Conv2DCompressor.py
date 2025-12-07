import torch
from torch import nn


class Conv2DCompressor(nn.Module):
    """
    Compresse les feature maps en utilisant un module CNN à 2 couches.
    Implémente le sous-échantillonnage spatial 16x (H/4, W/4) et
    l'augmentation des canaux de 256 à 1024, comme décrit dans le papier.

    (In: B, 256, 64, 64) -> (Out: B, 1024, 16, 16)
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 1024):
        super(Conv2DCompressor, self).__init__()

        # Le papier n'est pas explicite sur le nombre de canaux intermédiaires.
        # Une progression logique est 256 -> 512 -> 1024.
        intermediate_channels = (
            in_channels + out_channels
        ) // 2  # (256+1024)//2 = 640 ou 512
        intermediate_channels = 512  # Forçons une valeur standard

        self.compressor = nn.Sequential(
            # In: (B, 256, 64, 64)
            # Out: (B, 512, 32, 32) (Downsample 4x)
            nn.Conv2d(
                in_channels, intermediate_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.GELU(),
            # In: (B, 512, 32, 32)
            # Out: (B, 1024, 16, 16) (Downsample 16x total)
            nn.Conv2d(
                intermediate_channels, out_channels, kernel_size=3, stride=2, padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresser la feature map

        Args:
            x: Tenseur d'entrée (B, C_in, H, W), ex: (B, 256, 64, 64)

        Returns:
            Tenseur compressé (B, C_out, H_out, W_out), ex: (B, 1024, 16, 16)
        """
        return self.compressor(x)

    def visualize_compressed_tokens(
        self,
        compressed_features: torch.Tensor,
        original_image=None,
        save_path: str = "res/compressed_tokens.png",
    ):
        """
        Visualize compressed feature tokens

        Args:
            compressed_features: Compressed features (B, C, H, W) or (C, H, W)
            original_image: Optional PIL Image to display alongside
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt

        # Remove batch dimension if present
        if compressed_features.ndim == 4:
            compressed_features = compressed_features[0]

        compressed = compressed_features.detach().cpu().numpy()  # (C, H, W)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            "Compressed Feature Maps - Sample Channels", fontsize=16, fontweight="bold"
        )

        # Original image (if provided)
        if original_image is not None:
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title("Original Image", fontweight="bold")
        else:
            axes[0, 0].text(0.5, 0.5, "No Image", ha="center", va="center", fontsize=14)
        axes[0, 0].axis("off")

        # Show 5 compressed channels
        num_channels = compressed.shape[0]
        step = max(1, num_channels // 5)
        channels_to_show = [i * step for i in range(5)]

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

            # Add grid to show spatial structure
            h, w = channel_data.shape
            for i in range(h + 1):
                ax.axhline(i - 0.5, color="white", linewidth=0.5, alpha=0.3)
            for i in range(w + 1):
                ax.axvline(i - 0.5, color="white", linewidth=0.5, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Compressed tokens saved to: {save_path}")
        plt.show()
