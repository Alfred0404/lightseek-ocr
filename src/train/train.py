"""
Training Script for LightSeek-OCR
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves to file, no display needed
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LightSeekOCR import LightSeekOCR
from dataset import SyntheticOCRDataset
from utils.colors import bcolors


def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return images, texts


def plot_loss(epoch_losses, loss_plot_path):
    """Save updated loss curve to disk after each epoch."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o", color="royalblue", label="Train Loss")
    ax.set_title("Training Loss (live)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    # Annotate last value
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
    BATCH_SIZE = 4          # Physical batch size
    ACCUMULATION_STEPS = 8  # Effective batch = 32 samples
    EPOCHS = 30

    CHECKPOINT_DIR = "src/train/checkpoints"
    METRICS_DIR = "src/train/training_metrics"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{bcolors.HEADER}Starting Training on {device}{bcolors.ENDC}")

    # --- Model ---
    model = LightSeekOCR(verbose=False).to(device)

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

    # Freeze all GPT-2 first
    for param in model.decoder.model.parameters():
        param.requires_grad = False

    # Unfreeze last 4 transformer blocks + final layer norm
    for i in [-1, -2, -3, -4]:
        for param in model.decoder.model.transformer.h[i].parameters():
            param.requires_grad = True
    for param in model.decoder.model.transformer.ln_f.parameters():
        param.requires_grad = True
    print("  - GPT-2: Frozen except last 4 blocks + ln_f")

    # Trainable: Compressor, channel projection, visual projection
    for param in model.encoder.compressor.parameters():
        param.requires_grad = True
    for param in model.encoder.channel_projection.parameters():
        param.requires_grad = True
    for param in model.decoder.visual_projection.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"  - Trainable: {trainable_params:,} / {all_params:,} ({trainable_params/all_params:.1%})")

    # --- Optimizer with differential learning rates ---
    gpt2_blocks_params = [
        p for i in [-1, -2, -3, -4]
        for p in model.decoder.model.transformer.h[i].parameters()
    ]
    optimizer = optim.AdamW([
        {"params": model.encoder.compressor.parameters(),       "lr": 1e-4},
        {"params": model.encoder.channel_projection.parameters(), "lr": 1e-4},
        {"params": model.decoder.visual_projection.parameters(), "lr": 1e-4},
        {"params": gpt2_blocks_params,                           "lr": 5e-5},
        {"params": model.decoder.model.transformer.ln_f.parameters(), "lr": 5e-5},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- Dataset ---
    print(f"\n{bcolors.OKBLUE}Initializing Dataset...{bcolors.ENDC}")
    dataset = SyntheticOCRDataset(length=1000)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    print(f"  - Fonts available: {dataset.available_fonts}")

    # --- Setup dirs ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    loss_plot_path = os.path.join(METRICS_DIR, "loss_curve.png")
    print(f"\n{bcolors.HEADER}Training Start!{bcolors.ENDC}")
    print(f"  Loss curve saved live to: {loss_plot_path}\n")

    epoch_losses = []
    step = 0  # global sample counter for accumulation

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_samples = 0
        optimizer.zero_grad()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, texts in progress_bar:
            # Process every sample in the batch (fix: was only using images[0])
            for image, text in zip(images, texts):
                features = model.encoder.extract_features(image)

                compressed = features["compressed_features"]   # (1, 768, 16, 16)
                global_f   = features["global_features"]       # (1, 256, 768)
                local_f    = compressed.flatten(2).permute(0, 2, 1)  # (1, 256, 768)

                tokenizer = model.decoder.tokenizer
                text_inputs = tokenizer(
                    text + tokenizer.eos_token,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)

                batch_size = text_inputs.input_ids.shape[0]
                visual_padding = torch.full(
                    (batch_size, 512), -100, dtype=torch.long, device=device
                )
                labels = torch.cat([visual_padding, text_inputs.input_ids], dim=1)

                outputs = model.decoder(
                    local_features=local_f,
                    global_features=global_f,
                    text_input_ids=text_inputs.input_ids,
                    text_attention_mask=text_inputs.attention_mask,
                    labels=labels,
                )

                loss = outputs.loss / ACCUMULATION_STEPS
                loss.backward()

                epoch_loss += outputs.loss.item()
                n_samples += 1
                step += 1

                if step % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            progress_bar.set_postfix({
                "loss": f"{epoch_loss / max(n_samples, 1):.4f}",
                "lr_enc": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Final optimizer step if leftover gradients
        if step % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        avg_loss = epoch_loss / max(n_samples, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth"),
        )

        # Update loss curve on disk after every epoch
        plot_loss(epoch_losses, loss_plot_path)

    print(f"\n{bcolors.OKGREEN}Training complete. Loss curve: {loss_plot_path}{bcolors.ENDC}")


if __name__ == "__main__":
    train()
