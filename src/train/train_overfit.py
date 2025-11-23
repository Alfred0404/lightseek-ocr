"""
Overfitting Test Script for LightSeek-OCR
Trains on a SINGLE sample to verify model capacity and gradient flow.
"""

import os
import torch
import torch.optim as optim
import sys

# Add src to path
# Current file is in src/train, so we need to add src (parent dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LightSeekOCR import LightSeekOCR
from dataset import SyntheticOCRDataset
from utils.colors import bcolors


def train_overfit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{bcolors.HEADER}Starting Overfit Test on {device}{bcolors.ENDC}")

    # Initialize Model
    model = LightSeekOCR(verbose=False).to(device)
    model.train()

    # --- Freezing Strategy ---
    # Freeze SAM (Encoder)
    for param in model.encoder.sam_extractor.model.parameters():
        param.requires_grad = False

    # Freeze CLIP (Encoder)
    for param in model.encoder.clip_processor.model.parameters():
        param.requires_grad = False

    # Freeze GPT-2 (Decoder) - OPTIONAL: You might want to freeze it or not
    # For this specific request "train compressor", we should freeze GPT-2 too?
    # The user said "ni clip, ni sam, ni gpt2". So we freeze GPT-2.
    for param in model.decoder.model.parameters():
        param.requires_grad = False

    # Ensure Compressor and Projection are trainable
    for param in model.encoder.compressor.parameters():
        param.requires_grad = True
    for param in model.encoder.channel_projection.parameters():
        param.requires_grad = True
    # Visual Projection in Decoder MUST be trainable to adapt features to GPT-2
    for param in model.decoder.visual_projection.parameters():
        param.requires_grad = True

    # Check trainable params
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable Parameters: {trainable_params:,} / {all_params:,} ({trainable_params/all_params:.1%})"
    )

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    # Get Data
    dataset = SyntheticOCRDataset(length=1)
    image, text = dataset[0]

    print(f"Target Text: '{text}'")

    # Training Loop
    print("Step | Loss")
    print("-" * 20)

    loss = 1
    step = 0

    loss_treshold = 0.1
    loss_history = []

    while loss > loss_treshold:
        optimizer.zero_grad()

        # 1. Extract Features
        # We need to run the full extraction because we are training the compressor!
        # Gradients must flow back through extract_features -> compressor
        features = model.encoder.extract_features(image)
        compressed = features["compressed_features"]
        global_f = features["global_features"]
        local_f = compressed.flatten(2).permute(0, 2, 1)

        # 2. Tokenize
        tokenizer = model.decoder.tokenizer
        text_with_eos = text + tokenizer.eos_token
        text_inputs = tokenizer(
            text_with_eos, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # 3. Labels
        batch_size = text_inputs.input_ids.shape[0]
        visual_padding = torch.full(
            (batch_size, 512), -100, dtype=torch.long, device=device
        )
        labels = torch.cat([visual_padding, text_inputs.input_ids], dim=1)

        # 4. Forward
        outputs = model.decoder(
            local_features=local_f,
            global_features=global_f,
            text_input_ids=text_inputs.input_ids,
            text_attention_mask=text_inputs.attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"{step:4d} | {loss.item():.4f}")

        loss_history.append(loss.item())

    print("-" * 20)
    print(f"Final Loss: {loss.item():.4f}")

    # Test Generation
    print("\nTesting Generation...")
    model.eval()
    with torch.no_grad():
        result = model.predict_from_image(image)
    print(f"Input: '{text}'")
    print(f"Output: '{result['generated_text']}'")

    # Plot loss history
    import matplotlib.pyplot as plt

    plt.plot(loss_history)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.show()


if __name__ == "__main__":
    train_overfit()
