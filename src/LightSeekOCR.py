"""
LightSeekOCR
The complete Optical Character Recognition pipeline.
Combines DeepEncoder (Text->Visual) and DeepDecoder (Visual->Text).
"""

import torch
import torch.nn as nn
from transformers import infer_device

from DeepEncoder import DeepEncoder
from DeepDecoder import DeepDecoder
from utils.colors import bcolors


class LightSeekOCR(nn.Module):
    """
    LightSeek-OCR: A lightweight implementation of DeepSeek-OCR architecture.

    Pipeline:
    1. Input Text (Ground Truth) -> Rendered Image
    2. Image -> DeepEncoder -> Visual Features (Local + Global)
    3. Visual Features -> DeepDecoder -> Predicted Text
    """

    def __init__(
        self,
        encoder_name="facebook/sam-vit-base",
        decoder_name="gpt2",
        device=None,
        verbose=True,
    ):
        super().__init__()
        self.device = device if device is not None else infer_device()
        self.verbose = verbose

        if self.verbose:
            print("=" * 70)
            print(f"{bcolors.HEADER}Initializing LightSeek-OCR Pipeline{bcolors.ENDC}")
            print("=" * 70)

        # Initialize Encoder
        self.encoder = DeepEncoder(
            sam_model_name=encoder_name, device=self.device, verbose=verbose
        )

        # Initialize Decoder
        self.decoder = DeepDecoder(
            model_name=decoder_name, device=self.device, verbose=verbose
        )

        if self.verbose:
            print("\n" + "=" * 70)
            print(f"{bcolors.OKGREEN}LightSeek-OCR Pipeline Ready!{bcolors.ENDC}")
            print("=" * 70 + "\n")

    def predict_from_image(self, image, max_new_tokens=50) -> dict:
        """
        Run the pipeline from an image (Visual Features -> Text)

        Args:
            image: PIL Image
            max_new_tokens: Maximum tokens to generate

        Returns:
            dict containing:
                - generated_text: Output from decoder
                - visual_features: Intermediate features
        """
        # 1. Extract Features
        features = self.encoder.extract_features(image)
        compressed = features["compressed_features"]
        global_f = features["global_features"]
        local_f = compressed.flatten(2).permute(0, 2, 1)

        # 2. Decode
        if self.verbose:
            print(f"{bcolors.OKBLUE}Decoding visual features...{bcolors.ENDC}")

        generated_text = self.decoder.decode(
            local_features=local_f,
            global_features=global_f,
            max_new_tokens=max_new_tokens,
        )

        return {
            "generated_text": generated_text[0],
            "encoder_results": features,
        }

    def predict(self, text: str, max_new_tokens=50) -> dict:
        """
        Run the full pipeline: Text -> Image -> Features -> Text

        Args:
            text: Input text (ground truth) to render and reconstruct
            max_new_tokens: Maximum tokens to generate

        Returns:
            dict containing:
                - original_text: Input text
                - generated_text: Output from decoder
                - visual_features: Intermediate features
        """
        if self.verbose:
            print(f"{bcolors.OKBLUE}Running Pipeline on: '{text}'{bcolors.ENDC}")

        # 1. Render Text to Image
        image = self.encoder.text_to_image(text)

        # 2. Predict from Image
        result = self.predict_from_image(image, max_new_tokens)
        result["original_text"] = text
        result["image"] = image

        return result


if __name__ == "__main__":
    # Simple test
    ocr = LightSeekOCR(verbose=True)
    result = ocr.predict("Hello LightSeek")
    print(f"\nOriginal: {result['original_text']}")
    print(f"Generated: {result['generated_text']}")
