"""
Complete Pipeline Runner
Runs the full LightSeek-OCR pipeline on a test sentence.
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from LightSeekOCR import LightSeekOCR
from utils.colors import bcolors


def main():
    print(f"{bcolors.HEADER}Starting Complete Pipeline Test{bcolors.ENDC}")

    # Initialize the full model
    ocr = LightSeekOCR(verbose=True)

    # Test Sentence
    test_text = "The quick brown fox jumps over the lazy dog."

    print(f"\n{bcolors.BOLD}Input Text:{bcolors.ENDC} {test_text}")
    print("-" * 50)

    # Run Prediction
    # 1. Renders text to image
    # 2. Extracts SAM features (local)
    # 3. Extracts CLIP features (global)
    # 4. Decodes features back to text via GPT-2
    result = ocr.predict(test_text, max_new_tokens=20)

    print("-" * 50)
    print(f"{bcolors.OKGREEN}Pipeline Finished!{bcolors.ENDC}")
    print("-" * 50)
    print(f"{bcolors.BOLD}Original Text :{bcolors.ENDC} {result['original_text']}")
    print(f"{bcolors.BOLD}Decoded Text  :{bcolors.ENDC} {result['generated_text']}")
    print("-" * 50)

    print(
        f"\n{bcolors.WARNING}Note: Since the decoder (GPT-2) has NOT been trained on these visual features yet,"
    )
    print(
        f"the output text will be random or hallucinated. This is expected behavior.{bcolors.ENDC}"
    )

    # Save the rendered image to verify that part worked
    image_path = "pipeline_test_image.png"
    result["encoder_results"]["image"].save(image_path)
    print(f"\nRendered image saved to: {image_path}")


if __name__ == "__main__":
    main()
