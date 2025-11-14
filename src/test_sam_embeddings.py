from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor

# Load model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Load image
img_url = "./output_image.png"
raw_image = Image.open(img_url).convert("RGB")
print(f"Image size: {raw_image.size}")

# Define input points for SAM
input_points = [[[450, 600]]]  # 2D localization of a window

# Process inputs and move to device
inputs = processor(raw_image, input_points=input_points, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate masks
with torch.no_grad():
    outputs = model(**inputs)

# Post-process masks
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu(),
)
scores = outputs.iou_scores

print("Masks shape:", masks[0].shape)
print("Scores:", scores[0])
print("Test completed successfully.")

# Visualize segmentation masks
fig, axes = plt.subplots(1, 4, figsize=(15, 5))

# Original image
axes[0].imshow(raw_image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Show the 3 predicted masks with different IoU scores
mask_array = masks[0].squeeze().cpu().numpy()  # Shape: (num_masks, H, W)
scores_cpu = scores.cpu().squeeze().numpy()  # Convert scores to numpy

for idx in range(min(3, mask_array.shape[0])):
    ax = axes[idx + 1]

    # Show mask overlay on image
    ax.imshow(raw_image)
    ax.imshow(mask_array[idx], alpha=0.5, cmap="viridis")
    ax.set_title(f"Mask {idx+1}\nIoU: {scores_cpu[idx]:.3f}")
    ax.axis("off")

    # Add red point for input location
    ax.plot(input_points[0][0][0], input_points[0][0][1], "ro", markersize=10)

plt.suptitle("SAM Segmentation Masks", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("sam_segmentation_result.png", dpi=150, bbox_inches="tight")
print("\n✓ Visualization saved to: sam_segmentation_result.png")
plt.show()
