"""
LightSeekOCR
The complete OCR pipeline.
Combines DeepEncoder (Image → Visual Features) and DeepDecoder (Visual Features → Text).
"""

import torch
import torch.nn as nn
from transformers import infer_device

from DeepEncoder import DeepEncoder
from DeepDecoder import DeepDecoder
from utils.colors import bcolors


class LightSeekOCR(nn.Module):
    """
    LightSeek-OCR: Lightweight reproduction of the DeepSeek-OCR architecture.

    Pipeline:
      PIL Image → DeepEncoder → (local_features, global_features) → DeepDecoder → Transcription
    """

    def __init__(
        self,
        encoder_name: str = "facebook/sam-vit-base",
        decoder_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device=None,
        verbose: bool = True,
    ):
        super().__init__()
        self.device = device if device is not None else infer_device()
        self.verbose = verbose

        if self.verbose:
            print("=" * 70)
            print(f"{bcolors.HEADER}Initializing LightSeek-OCR Pipeline{bcolors.ENDC}")
            print("=" * 70)

        self.encoder = DeepEncoder(
            sam_model_name=encoder_name, device=self.device, verbose=verbose
        )
        self.decoder = DeepDecoder(
            model_name=decoder_name, device=self.device, verbose=verbose
        )

        if self.verbose:
            print("\n" + "=" * 70)
            print(f"{bcolors.OKGREEN}LightSeek-OCR Pipeline Ready!{bcolors.ENDC}")
            print("=" * 70 + "\n")

    def predict_from_image(self, image, max_new_tokens: int = 200) -> dict:
        """
        Run the OCR pipeline on a PIL Image.

        Args:
            image          : PIL Image
            max_new_tokens : Max tokens to generate

        Returns:
            dict with keys: generated_text, encoder_results
        """
        features = self.encoder.extract_features(image)
        compressed = features["compressed_features"]          # (1, 768, 16, 16)
        global_f = features["global_features"]                # (1, 256, 768)
        local_f = compressed.flatten(2).permute(0, 2, 1)     # (1, 256, 768)

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


if __name__ == "__main__":
    ocr = LightSeekOCR(verbose=True)
    from PIL import Image
    img = Image.new("RGB", (1024, 1024), (255, 255, 255))
    result = ocr.predict_from_image(img)
    print(f"Generated: {result['generated_text']}")
