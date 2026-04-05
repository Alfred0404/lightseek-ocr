import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
from wonderwords import RandomWord


# Fonts available on Windows — each is tried at runtime, missing ones are skipped
CANDIDATE_FONTS = [
    "arial.ttf",
    "arialbd.ttf",
    "cour.ttf",
    "verdana.ttf",
    "calibri.ttf",
    "comic.ttf",
    "georgia.ttf",
    "trebuc.ttf",
]


def _available_fonts(size=60):
    """Return the subset of CANDIDATE_FONTS that can actually be loaded."""
    available = []
    for name in CANDIDATE_FONTS:
        try:
            ImageFont.truetype(name, size=size)
            available.append(name)
        except IOError:
            pass
    return available if available else None  # None → use default font


class SyntheticOCRDataset(Dataset):
    """
    Dataset that generates synthetic OCR images on the fly with visual variation.
    """

    def __init__(self, length=1000, image_size=(1024, 1024)):
        self.length = length
        self.image_size = image_size
        self.available_fonts = _available_fonts()
        self.r = RandomWord()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # --- Random text (1–3 words) ---
        text_length = random.randint(1, 3)
        try:
            words = self.r.random_words(text_length)
            if isinstance(words, str):
                words = [words]
        except Exception:
            words = ["error"]
        text = " ".join(words)

        # --- Random visual style ---
        font_size = random.randint(40, 180)

        # Slightly off-white background, slightly off-black text
        bg_val = random.randint(230, 255)
        bg_color = (bg_val, bg_val, bg_val)
        txt_val = random.randint(0, 25)
        txt_color = (txt_val, txt_val, txt_val)

        # --- Render ---
        W, H = self.image_size
        image = Image.new("RGB", self.image_size, color=bg_color)
        draw = ImageDraw.Draw(image)

        if self.available_fonts:
            font_name = random.choice(self.available_fonts)
            try:
                font = ImageFont.truetype(font_name, font_size)
            except IOError:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # Compute text bounding box to pick a valid random position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        max_x = max(0, W - text_w - 10)
        max_y = max(0, H - text_h - 10)
        x = random.randint(10, max(10, max_x))
        y = random.randint(10, max(10, max_y))

        draw.text((x, y), text, fill=txt_color, font=font)

        # Optional: very light Gaussian blur (simulates slight defocus)
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        # --- Crop to text bounding box + margin, resize back ---
        img_bbox = image.getbbox()  # bounding box of non-background pixels
        if img_bbox is not None:
            margin = 30
            x0 = max(0, img_bbox[0] - margin)
            y0 = max(0, img_bbox[1] - margin)
            x1 = min(W, img_bbox[2] + margin)
            y1 = min(H, img_bbox[3] + margin)
            image = image.crop((x0, y0, x1, y1)).resize(self.image_size, Image.LANCZOS)

        return image, text
