import os
from src.dataset import SyntheticOCRDataset


def generate_samples():
    output_dir = "res/debug_samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = SyntheticOCRDataset(length=5)
    print(f"Generating 5 samples to {output_dir}...")

    for i in range(5):
        image, text = dataset[i]
        save_path = os.path.join(output_dir, f"sample_{i}_{text.replace(' ', '_')}.png")
        image.save(save_path)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    generate_samples()
