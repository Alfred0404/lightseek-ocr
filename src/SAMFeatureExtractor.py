"""
SAM Feature Extractor
Extracts visual features using SAM vision encoder
"""

import torch
from PIL import Image
from torchvision import transforms
from transformers import SamVisionModel, infer_device
import math
import numpy as np

from colors import bcolors


class SAMFeatureExtractor:
    """Extract visual features using SAM vision encoder"""

    def __init__(self, model_name="facebook/sam-vit-base", device=None):
        self.device = device if device is not None else infer_device()
        print(
            f"{bcolors.OKCYAN}Loading SAM model on device: {self.device}{bcolors.ENDC}"
        )

        # https://huggingface.co/facebook/sam-vit-base
        self.model = SamVisionModel.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Preprocessing transform
        self.transform = transforms.Compose(
            [transforms.Resize((1024, 1024)), transforms.ToTensor()]
        )

    def extract(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Extract features from PIL image

        Args:
            pil_image: PIL Image (RGB)

        Returns:
            Feature map tensor of shape (B, 256, 64, 64)
        """
        # Ensure RGB and correct size
        pil_image = pil_image.convert("RGB")

        # Transform to tensor and add batch dimension
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            outputs = self.model(input_tensor)
            feature_map = outputs.last_hidden_state  # (B, 256, 64, 64)

        return feature_map

    def show_feature_map(self, pil_image: Image.Image) -> Image.Image:
        """
        Visualize feature map as a colored heatmap image

        Args:
            pil_image: PIL Image (RGB)

        Returns:
            PIL Image visualizing the feature map as a color heatmap grid
        """
        import matplotlib.cm as cm

        feature_map = self.extract(pil_image)  # (B, 256, 64, 64)
        feature_map = feature_map.squeeze(0)  # (C, H, W)

        # Convert to numpy
        feature_map_np = feature_map.cpu().numpy()  # (C, H, W)
        n_channels, h, w = feature_map_np.shape

        # Create colormap
        cmap = cm.get_cmap("viridis")  # change to 'inferno' or 'magma' if you prefer

        # Grid layout (use ceil to handle non-perfect squares)
        grid_cols = int(math.ceil(math.sqrt(n_channels)))
        grid_rows = int(math.ceil(n_channels / grid_cols))

        # Create RGB grid image
        grid_image = Image.new("RGB", (grid_cols * w, grid_rows * h))

        for idx in range(n_channels):
            ch = feature_map_np[idx]
            # Normalize per-channel for better contrast
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                norm = (ch - ch_min) / (ch_max - ch_min)
            else:
                norm = np.zeros_like(ch)

            # Apply colormap -> RGBA float in [0,1], convert to uint8 RGB
            colored = (cmap(norm)[:, :, :3] * 255).astype("uint8")
            channel_img = Image.fromarray(colored, mode="RGB")

            row = idx // grid_cols
            col = idx % grid_cols
            grid_image.paste(channel_img, (col * w, row * h))

        return grid_image

    def visualize_all_channels(
        self, feature_map: torch.Tensor, save_path: str = "res/sam_local_features.png"
    ):
        """
        Visualize all SAM feature channels in a grid

        Args:
            feature_map: SAM features (B, 256, 64, 64) or (256, 64, 64)
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt

        # Remove batch dimension if present
        if feature_map.ndim == 4:
            feature_map = feature_map[0]

        local_features = feature_map.detach().cpu().numpy()  # (256, 64, 64)

        # Display all SAM channels
        num_channels = local_features.shape[0]
        n_cols = 16
        n_rows = (num_channels + n_cols - 1) // n_cols
        figsize = (n_cols * 1.2, n_rows * 1.2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(
            "SAM Local Features - All Channels", fontsize=16, fontweight="bold"
        )

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
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(
            f"{bcolors.OKGREEN}✓ SAM local features saved to: {save_path}{bcolors.ENDC}"
        )
        plt.show()


if __name__ == "__main__":
    # Test the SAMFeatureExtractor
    extractor = SAMFeatureExtractor()

    # Load a sample image
    sample_image = Image.open("image.png").convert("RGB")

    # Extract feature map
    features = extractor.extract(sample_image)
    print(
        f"{bcolors.OKGREEN}Extracted feature map shape: {features.shape}{bcolors.ENDC}"
    )

    # Visualize all channels
    extractor.visualize_all_channels(features)

    # Visualize feature map grid
    feature_map_image = extractor.show_feature_map(sample_image)
    feature_map_image.show()
    feature_map_image.save("sam_feature_map.png")
