"""
Training Script for LightSeek-OCR
Phase 1+2: SmolLM2-1.7B-Instruct + LoRA, SROIE dataset, transcription objective.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # src/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))                        # src/train/

from LightSeekOCR import LightSeekOCR
from dataset import build_dataset
from prompt_template import TRANSCRIPTION_PROMPT
from utils.colors import bcolors


def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return images, texts


def plot_loss(epoch_losses, loss_plot_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", color="royalblue", label="Train Loss")
    ax.set_title("Training Loss (live)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.annotate(
        f"{epoch_losses[-1]:.4f}",
        xy=(len(epoch_losses), epoch_losses[-1]),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=9,
        color="royalblue",
    )
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=120)
    plt.close(fig)


def train():
    # --- Configuration ---
    DATASET_NAME = "iam"     # "iam" | "cord" | "synthetic"
    BATCH_SIZE = 1           # Physical batch (SmolLM2 is large; 1 is safest on 8GB)
    ACCUMULATION_STEPS = 32  # Effective batch = 32
    EPOCHS = 30
    MAX_TEXT_TOKENS = 128    # Truncate transcription to this many tokens

    CHECKPOINT_DIR = "src/train/checkpoints"
    METRICS_DIR = "src/train/training_metrics"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{bcolors.HEADER}Starting Training on {device}{bcolors.ENDC}")

    # --- Model ---
    model = LightSeekOCR(verbose=True)
    # LightSeekOCR doesn't need .to(device) — each sub-module already placed its weights on device.

    # --- Freezing Strategy ---
    print(f"\n{bcolors.OKBLUE}Configuring Freezing Strategy...{bcolors.ENDC}")

    # Freeze SAM
    for param in model.encoder.sam_extractor.model.parameters():
        param.requires_grad = False
    print("  - SAM: Frozen")

    # Freeze CLIP
    for param in model.encoder.clip_processor.model.parameters():
        param.requires_grad = False
    print("  - CLIP: Frozen")

    # SmolLM2 base weights are already frozen by get_peft_model (only LoRA trainable).
    # Explicitly ensure compressor + channel_projection + visual_projection are trainable.
    for param in model.encoder.compressor.parameters():
        param.requires_grad = True
    for param in model.encoder.channel_projection.parameters():
        param.requires_grad = True
    for param in model.decoder.visual_projection.parameters():
        param.requires_grad = True
    print("  - SmolLM2: Base frozen, LoRA adapters trainable")
    print("  - Compressor + channel_projection + visual_projection: Trainable")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total trainable: {trainable_params:,} / {all_params:,} ({trainable_params / all_params:.2%})")

    # --- Optimizer ---
    # Differential LRs: visual projectors at 1e-4, LoRA at 5e-5
    lora_params = [p for p in model.decoder.model.parameters() if p.requires_grad]
    optimizer = optim.AdamW([
        {"params": model.encoder.compressor.parameters(),          "lr": 1e-4},
        {"params": model.encoder.channel_projection.parameters(),  "lr": 1e-4},
        {"params": model.decoder.visual_projection.parameters(),   "lr": 1e-4},
        {"params": lora_params,                                    "lr": 5e-5},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()  # fp16 gradient scaling

    # --- Dataset ---
    print(f"\n{bcolors.OKBLUE}Loading dataset '{DATASET_NAME}'...{bcolors.ENDC}")
    dataset = build_dataset(name=DATASET_NAME, split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 0 on Windows to avoid multiprocessing issues
    )
    print(f"  - Dataset size: {len(dataset)} samples")

    # --- Pre-tokenise prompt (same for every sample) ---
    tokenizer = model.decoder.tokenizer
    prompt_ids = tokenizer(
        TRANSCRIPTION_PROMPT,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    N_prompt = prompt_ids.shape[1]
    N_visual = 512  # local(256) + global(256)

    # --- Setup dirs ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    loss_plot_path = os.path.join(METRICS_DIR, "loss_curve.png")

    print(f"\n{bcolors.HEADER}Training Start!{bcolors.ENDC}")
    print(f"  Loss curve saved live to: {loss_plot_path}\n")

    epoch_losses = []
    step = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_samples = 0
        optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for images, texts in progress_bar:
            B = len(images)

            # --- Batch encode (single forward through SAM + Compressor + CLIP) ---
            with torch.no_grad():
                features = model.encoder.extract_features_batch(images)

            compressed = features["compressed_features"]                 # (B, 768, 16, 16)
            global_f   = features["global_features"]                     # (B, 256, 768)
            local_f    = compressed.flatten(2).permute(0, 2, 1)         # (B, 256, 768)

            # --- Batch tokenise transcriptions (pad to same length) ---
            text_inputs = tokenizer(
                [t + tokenizer.eos_token for t in texts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TEXT_TOKENS,
            ).to(device)
            # text_inputs.input_ids      : (B, N_max)
            # text_inputs.attention_mask : (B, N_max)  — 0 on padding positions

            # Labels: padding positions → -100 (ignored by loss)
            labels_text = text_inputs.input_ids.clone()
            labels_text[text_inputs.attention_mask == 0] = -100

            # Combined text input to decoder: [prompt | transcription]
            combined_text_ids  = torch.cat([prompt_ids.expand(B, -1), text_inputs.input_ids], dim=1)
            combined_text_mask = torch.cat([
                torch.ones((B, N_prompt), dtype=torch.long, device=device),
                text_inputs.attention_mask,
            ], dim=1)

            # Labels: [-100×N_visual | -100×N_prompt | text_labels (with -100 on padding)]
            labels = torch.cat([
                torch.full((B, N_visual), -100, dtype=torch.long, device=device),
                torch.full((B, N_prompt), -100, dtype=torch.long, device=device),
                labels_text,
            ], dim=1)

            # --- Forward with fp16 autocast ---
            with autocast(device_type="cuda"):
                outputs = model.decoder(
                    local_features=local_f,
                    global_features=global_f,
                    text_input_ids=combined_text_ids,
                    text_attention_mask=combined_text_mask,
                    labels=labels,
                )

            loss = outputs.loss / ACCUMULATION_STEPS
            scaler.scale(loss).backward()

            epoch_loss += outputs.loss.item() * B
            n_samples  += B
            step       += 1

            if step % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            progress_bar.set_postfix({
                "loss": f"{epoch_loss / max(n_samples, 1):.4f}",
                "lr_vis": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Flush leftover gradients
        if step % ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        avg_loss = epoch_loss / max(n_samples, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  Loss: {avg_loss:.4f}")

        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth"),
        )
        plot_loss(epoch_losses, loss_plot_path)

    print(f"\n{bcolors.OKGREEN}Training complete. Loss curve: {loss_plot_path}{bcolors.ENDC}")


if __name__ == "__main__":
    train()
