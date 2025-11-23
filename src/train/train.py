"""
Training Script for LightSeek-OCR
Optimized for RTX 3070 (8GB VRAM)
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add src to path
# Current file is in src/train, so we need to add src (parent dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LightSeekOCR import LightSeekOCR
from dataset import SyntheticOCRDataset
from utils.colors import bcolors


def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return images, texts


def train():
    # --- Configuration ---
    BATCH_SIZE = 1  # Physical batch size (keep small for 8GB VRAM)
    ACCUMULATION_STEPS = 4  # Effective batch size = 4 (More updates)
    LEARNING_RATE = 1e-4
    EPOCHS = 5
    SAVE_DIR = "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{bcolors.HEADER}Starting Training on {device}{bcolors.ENDC}")

    # --- Model ---
    # Initialize with verbose=False to reduce noise during training
    model = LightSeekOCR(verbose=False).to(device)

    # --- Freezing Strategy ---
    print(f"\n{bcolors.OKBLUE}Configuring Freezing Strategy...{bcolors.ENDC}")

    # Freeze SAM (Encoder)
    for param in model.encoder.sam_extractor.model.parameters():
        param.requires_grad = False
    print("  - SAM Encoder: Frozen ❄️")

    # Freeze CLIP (Encoder)
    for param in model.encoder.clip_processor.model.parameters():
        param.requires_grad = False
    print("  - CLIP Encoder: Frozen ❄️")

    # Freeze GPT-2 (Decoder)
    for param in model.decoder.model.parameters():
        param.requires_grad = False
    print("  - GPT-2 Decoder: Frozen ❄️")

    # Trainable: Compressor, Projection, Visual Projection
    # Ensure Compressor and Projection are trainable
    for param in model.encoder.compressor.parameters():
        param.requires_grad = True
    for param in model.encoder.channel_projection.parameters():
        param.requires_grad = True
    # Visual Projection in Decoder MUST be trainable to adapt features to GPT-2
    for param in model.decoder.visual_projection.parameters():
        param.requires_grad = True

    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"  - Trainable Parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params:.1%})"
    )

    # --- Optimizer ---
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    # --- Dataset ---
    print(f"\n{bcolors.OKBLUE}Initializing Dataset...{bcolors.ENDC}")
    dataset = SyntheticOCRDataset(length=1000)  # 1000 samples per epoch
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    # --- Training Loop ---
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"\n{bcolors.HEADER}Training Start!{bcolors.ENDC}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, (images, texts) in enumerate(progress_bar):
            # images is list of PIL images (len=1)
            # texts is list of strings (len=1)

            image = images[0]  # BATCH_SIZE=1 assumption
            text = texts[0]

            # 1. Extract Features
            features = model.encoder.extract_features(image)

            # Flatten
            compressed = features["compressed_features"]  # (1, 768, 16, 16)
            global_f = features["global_features"]  # (1, 256, 768)
            local_f = compressed.flatten(2).permute(0, 2, 1)  # (1, 256, 768)

            # Tokenize Text
            tokenizer = model.decoder.tokenizer
            text_with_eos = text + tokenizer.eos_token
            text_inputs = tokenizer(
                text_with_eos, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            # Prepare Labels
            batch_size = text_inputs.input_ids.shape[0]
            visual_padding = torch.full(
                (batch_size, 512), -100, dtype=torch.long, device=device
            )
            labels = torch.cat([visual_padding, text_inputs.input_ids], dim=1)

            # Forward Decoder
            outputs = model.decoder(
                local_features=local_f,
                global_features=global_f,
                text_input_ids=text_inputs.input_ids,
                text_attention_mask=text_inputs.attention_mask,
                labels=labels,  # Auto-regressive training
            )

            loss = outputs.loss / ACCUMULATION_STEPS

            # Backward (No AMP)
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({"loss": loss.item() * ACCUMULATION_STEPS})

        # Save Checkpoint
        torch.save(
            model.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
        )
        print(f"Saved checkpoint for epoch {epoch+1}")


if __name__ == "__main__":
    train()
