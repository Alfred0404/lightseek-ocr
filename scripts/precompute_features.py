"""
Pre-compute and cache encoder features (SAM + Compressor + CLIP).

Since SAM, Compressor, and CLIP are completely frozen during training, their
outputs are deterministic. Caching them eliminates the encoder forward pass
from the training loop, giving ~×6 speedup.

Disk usage: ~768KB per sample (2 × (256×768) float16) → ~5GB for IAM train.

Usage (from repo root):
    python scripts/precompute_features.py
    python scripts/precompute_features.py --dataset iam --split train --cache_dir data/cache --batch_size 8
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset import build_dataset
from DeepEncoder import DeepEncoder


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def precompute(dataset_name: str, split: str, cache_dir: str, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Dataset     : {dataset_name} / {split}")
    print(f"Cache dir   : {cache_dir}")
    print(f"Batch size  : {batch_size}")

    os.makedirs(cache_dir, exist_ok=True)

    # Check for already-computed samples (allow resuming)
    existing = set(os.listdir(cache_dir))

    # Load encoder only — no decoder needed
    encoder = DeepEncoder(verbose=False)
    for param in encoder.sam_extractor.model.parameters():
        param.requires_grad = False
    for param in encoder.clip_processor.model.parameters():
        param.requires_grad = False
    encoder.sam_extractor.model.eval()
    encoder.clip_processor.model.eval()

    dataset    = build_dataset(name=dataset_name, split=split)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    print(f"\nPre-computing {len(dataset)} samples...")

    idx = 0
    skipped = 0
    for images, texts in tqdm(dataloader):
        B = len(images)
        batch_files = [f"{split}_{idx + i:06d}.pt" for i in range(B)]
        need_compute = [f for f in batch_files if f not in existing]

        if not need_compute:
            idx += B
            skipped += B
            continue

        with torch.no_grad():
            features = encoder.extract_features_batch(images)

        compressed = features["compressed_features"]               # (B, 768, 16, 16)
        global_f   = features["global_features"]                   # (B, 256, 768)
        local_f    = compressed.flatten(2).permute(0, 2, 1)       # (B, 256, 768)

        for i in range(B):
            fname = os.path.join(cache_dir, f"{split}_{idx:06d}.pt")
            if os.path.basename(fname) not in existing:
                torch.save(
                    {
                        "local_f":  local_f[i].half().cpu(),
                        "global_f": global_f[i].half().cpu(),
                        "text":     texts[i],
                    },
                    fname,
                )
            idx += 1

    total_saved = idx - skipped
    print(f"\nDone: {total_saved} saved, {skipped} already existed → {idx} total in {cache_dir}")
    disk_gb = idx * 768 * 1024 / 1e9
    print(f"Estimated cache size: ~{disk_gb:.1f} GB")

    meta = {"dataset": dataset_name, "split": split, "n_samples": idx}
    with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute encoder features for fast training.")
    parser.add_argument("--dataset",    default="iam",        choices=["iam", "cord", "synthetic"])
    parser.add_argument("--split",      default="train")
    parser.add_argument("--cache_dir",  default="data/cache")
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()

    precompute(args.dataset, args.split, args.cache_dir, args.batch_size)
