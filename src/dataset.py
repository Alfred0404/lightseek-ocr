import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import random
import string


class SyntheticOCRDataset(Dataset):
    """
    Dataset that generates synthetic OCR images on the fly.
    """

    def __init__(self, length=1000, image_size=(1024, 1024)):
        self.length = length
        self.image_size = image_size
        self.fonts = [
            "arial.ttf",
            # Add more fonts if available
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate random text
        text_length = random.randint(3, 10)
        text = "".join(
            random.choices(string.ascii_letters + string.digits + " ", k=text_length)
        )

        # Render image
        image = Image.new("RGB", self.image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype(
                random.choice(self.fonts), size=random.randint(40, 80)
            )
        except:
            font = ImageFont.load_default()

        # Random position
        x = random.randint(50, 800)
        y = random.randint(50, 800)

        draw.text((x, y), text, fill=(0, 0, 0), font=font)

        return image, text
