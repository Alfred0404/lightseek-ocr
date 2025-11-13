"""
SAM Feature Extractor
Extracts visual features using SAM vision encoder
"""

import torch
from PIL import Image
from torchvision import transforms
from transformers import SamVisionModel, infer_device


class SAMFeatureExtractor:
    """Extract visual features using SAM vision encoder"""

    def __init__(self, model_name="facebook/sam-vit-base", device=None):
        self.device = device if device is not None else infer_device()
        print(f"Loading SAM model on device: {self.device}")

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
