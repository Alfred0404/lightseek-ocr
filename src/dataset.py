import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import random
import string
from wonderwords import RandomWord


class SyntheticOCRDataset(Dataset):
    """
    Dataset that generates synthetic OCR images on the fly.
    """

    def __init__(self, length=1000, image_size=(1024, 1024)):
        self.length = length
        self.image_size = image_size
        self.fonts = [
            "arial.ttf",
        ]
        self.r = RandomWord()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Generate random text (1 to 3 words)
        text_length = random.randint(1, 3)
        try:
            words = self.r.random_words(text_length)
            # wonderwords returns a list if amount > 1, or string if amount = 1
            if isinstance(words, str):
                words = [words]
        except:
            # Fallback if something fails
            words = ["Error"]

        text = " ".join(words)

        # Render image
        image = Image.new("RGB", self.image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype(random.choice(self.fonts), 150)
        except:
            font = ImageFont.load_default()

        # Fixed position
        x = 10
        y = 10

        draw.text((x, y), text, fill=(0, 0, 0), font=font)

        return image, text
