import numpy as np
from PIL import Image
import os


def extract_patches(img_array, patch_size=16, stride=16):
    h, w, c = img_array.shape
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_array[y : y + patch_size, x : x + patch_size, :]
            patches.append(patch)

    return np.array(patches)


def main():
    patch_size = 32
    image_path = "image.png"
    output_dir = f"patches_{patch_size}px"

    os.makedirs(output_dir, exist_ok=True)

    img = np.array(Image.open(image_path))
    print("Image shape:", img.shape)

    patches = extract_patches(img, patch_size=patch_size, stride=patch_size)
    print("Total patches:", len(patches))

    for idx, patch in enumerate(patches):
        Image.fromarray(patch).save(os.path.join(output_dir, f"patch_{idx}.png"))

    print(f"Saved {len(patches)} patches in: {output_dir}/")


if __name__ == "__main__":
    main()
