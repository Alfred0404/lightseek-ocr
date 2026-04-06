import glob
import json
import os
import random

import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch.utils.data import Dataset
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


def _pad_resize(image: Image.Image, target_size: tuple) -> Image.Image:
    """Resize with white padding, preserving aspect ratio."""
    target_w, target_h = target_size
    img_w, img_h = image.size
    ratio = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("RGB", target_size, (255, 255, 255))
    padded.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padded


class SyntheticOCRDataset(Dataset):
    """
    Dataset that generates synthetic OCR images on the fly with visual variation.
    Useful for overfit tests and quick iteration.
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

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        max_x = max(0, W - text_w - 10)
        max_y = max(0, H - text_h - 10)
        x = random.randint(10, max(10, max_x))
        y = random.randint(10, max(10, max_y))

        draw.text((x, y), text, fill=txt_color, font=font)

        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        # Crop to text bounding box + margin, resize back to input size
        img_bbox = image.getbbox()
        if img_bbox is not None:
            margin = 30
            x0 = max(0, img_bbox[0] - margin)
            y0 = max(0, img_bbox[1] - margin)
            x1 = min(W, img_bbox[2] + margin)
            y1 = min(H, img_bbox[3] + margin)
            image = image.crop((x0, y0, x1, y1)).resize(self.image_size, Image.LANCZOS)

        return image, text


class RealOCRDataset(Dataset):
    """
    Dataset wrapping a real OCR dataset from HuggingFace.

    Supported datasets:
      - "Teklia/IAM-line"  (default) : handwritten English lines, fields: image / text
      - "naver-clova-ix/cord-v2"     : printed receipts, fields: image / ground_truth (JSON)

    Usage:
        dataset = RealOCRDataset(split="train")
    """

    def __init__(
        self,
        hf_dataset: str = "Teklia/IAM-line",
        split: str = "train",
        image_size: tuple = (1024, 1024),
    ):
        from datasets import load_dataset
        self.hf_dataset = hf_dataset
        self.data = load_dataset(hf_dataset, split=split)
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # --- Image ---
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image = _pad_resize(image, self.image_size)

        # --- Transcription (dataset-specific) ---
        if "text" in item:
            text = item["text"]
        elif "words" in item:
            text = " ".join(item["words"])
        elif "ground_truth" in item:
            import json
            gt = json.loads(item["ground_truth"])
            # CORD: flatten all leaf string values from gt_parse
            def _collect(obj):
                if isinstance(obj, str):
                    return [obj]
                if isinstance(obj, dict):
                    return [v for sub in obj.values() for v in _collect(sub)]
                if isinstance(obj, list):
                    return [v for sub in obj for v in _collect(sub)]
                return [str(obj)]
            text = " ".join(_collect(gt.get("gt_parse", {})))
        else:
            text = ""

        return image, text


class CachedOCRDataset(Dataset):
    """
    Dataset backed by pre-computed encoder features (SAM + Compressor + CLIP).
    Generated by scripts/precompute_features.py.

    Each item: (local_f: Tensor (256, 768) fp16,
                global_f: Tensor (256, 768) fp16,
                text: str)
    """

    def __init__(self, cache_dir: str, split: str = "train"):
        self.files = sorted(glob.glob(os.path.join(cache_dir, f"{split}_*.pt")))
        if not self.files:
            raise FileNotFoundError(
                f"No cached features found in '{cache_dir}' for split='{split}'.\n"
                f"Run: python scripts/precompute_features.py --cache_dir {cache_dir}"
            )
        meta_path = os.path.join(cache_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"Cache: {meta.get('dataset')} / {split} — {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return data["local_f"], data["global_f"], data["text"]


def build_dataset(name: str = "synthetic", split: str = "train", **kwargs) -> Dataset:
    """
    Factory for dataset selection.

    Args:
        name  : "synthetic" | "iam" | "cord"
        split : HuggingFace split — ignored for synthetic
    """
    if name == "cached":
        cache_dir = kwargs.pop("cache_dir", "data/cache")
        return CachedOCRDataset(cache_dir=cache_dir, split=split)
    HF_NAMES = {
        "iam":  "Teklia/IAM-line",
        "cord": "naver-clova-ix/cord-v2",
    }
    if name in HF_NAMES:
        return RealOCRDataset(hf_dataset=HF_NAMES[name], split=split, **kwargs)
    return SyntheticOCRDataset(**kwargs)
