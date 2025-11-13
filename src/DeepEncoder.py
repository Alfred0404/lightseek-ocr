"""
DeepEncoder
Complete LightSeek-OCR encoding pipeline:
1. Text to image rendering
2. Image to SAM feature extraction (local features)
3. Feature map compression
4. CLIP vision encoder processing (global features)
"""

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import infer_device

from SAMFeatureExtractor import SAMFeatureExtractor
from CLIPVisionProcessor import CLIPVisionProcessor
from Conv2DCompressor import Conv2DCompressor


class DeepEncoder:
    """
    End-to-end encoder for LightSeek-OCR

    Transforms text into visual features ready for OCR decoding:
    - Local features (SAM): Fine-grained spatial details (B, 256, 64, 64)
    - Global features (CLIP): Semantic token sequence (B, N, 768)
    """

    def __init__(
        self,
        sam_model_name="facebook/sam-vit-base",
        clip_model_name="openai/clip-vit-base-patch32",
        device=None,
        verbose=True,
    ):
        """
        Initialize the DeepEncoder pipeline

        Args:
            sam_model_name: HuggingFace model name for SAM
            clip_model_name: HuggingFace model name for CLIP
            device: torch device (auto-detected if None)
            verbose: Print progress information
        """
        self.device = device if device is not None else infer_device()
        self.verbose = verbose

        if self.verbose:
            print("=" * 70)
            print("Initializing DeepEncoder")
            print("=" * 70)
            print(f"Device: {self.device}\n")

        # Initialize components
        self.sam_extractor = SAMFeatureExtractor(
            model_name=sam_model_name, device=self.device
        )
        self.clip_processor = CLIPVisionProcessor(
            model_name=clip_model_name, device=self.device
        )

        # Initialize compressor to match CLIP's expected dimension
        self.compressor = Conv2DCompressor(
            in_channels=256,  # SAM output channels
            out_channels=self.clip_processor.vision_hidden_size,  # CLIP vision hidden size (768)
        ).to(self.device)

        if self.verbose:
            print("\nDeepEncoder initialized successfully!\n")

    def text_to_image(
        self,
        text: str,
        image_size=(1024, 1024),
        bg_color=(255, 255, 255),
        text_color=(0, 0, 0),
        font_size=24,
    ) -> Image.Image:
        """
        Render text onto an image

        Args:
            text: Text to render
            image_size: (width, height) tuple
            bg_color: Background RGB color
            text_color: Text RGB color
            font_size: Font size in points

        Returns:
            PIL Image with rendered text
        """
        image = Image.new("RGB", image_size, color=bg_color)
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", size=font_size)
        except IOError:
            font = ImageFont.load_default()

        text_position = (10, 10)
        draw.text(text_position, text, fill=text_color, font=font)
        return image

    def encode(self, text: str) -> dict:
        """
        Run the complete encoding pipeline: text → visual features

        Args:
            text: Input text to encode

        Returns:
            dict with keys:
                - image: Rendered PIL Image
                - local_features: SAM features (B, 256, 64, 64)
                - global_features: CLIP sequence (B, N, 768)
                - compressed_features: Intermediate compressed map (B, 768, H, W)
        """
        if self.verbose:
            print("=" * 70)
            print("DeepEncoder Pipeline")
            print("=" * 70)
            print(f"Input text: {text}\n")

        # Step 1: Text to Image
        if self.verbose:
            print("[1/4] Rendering text to image...")
        pil_image = self.text_to_image(text)
        if self.verbose:
            print(f"  Image size: {pil_image.size}\n")

        # Step 2: SAM Feature Extraction (Local Features)
        if self.verbose:
            print("[2/4] Extracting SAM features (local)...")
        local_features = self.sam_extractor.extract(pil_image)
        if self.verbose:
            print(f"  Local features shape: {local_features.shape}\n")

        # Step 3: Feature Compression
        if self.verbose:
            print("[3/4] Compressing features...")
        compressed_map = self.compressor(local_features)
        if self.verbose:
            print(f"  Compressed map shape: {compressed_map.shape}\n")

        # Step 4: CLIP Processing (Global Features)
        if self.verbose:
            print("[4/4] Processing through CLIP (global)...")
        global_features = self.clip_processor.process_compressed_features(
            compressed_map
        )
        if self.verbose:
            print(f"  Global features shape: {global_features.shape}\n")

        if self.verbose:
            print("=" * 70)
            print("Encoding completed successfully!")
            print("=" * 70)
            print("\nVisual Features for Decoder:")
            print(
                f"  - Local features (SAM):   {local_features.shape}  # Fine-grained spatial"
            )
            print(
                f"  - Global features (CLIP): {global_features.shape}  # Semantic tokens"
            )
            print()

        return {
            "image": pil_image,
            "local_features": local_features,
            "global_features": global_features,
            "compressed_features": compressed_map,
        }


if __name__ == "__main__":
    # Example usage
    sample_text = "A scenic view of mountains during sunrise"

    # Initialize encoder
    encoder = DeepEncoder(verbose=True)

    # Encode text to visual features
    results = encoder.encode(sample_text)

    # Access outputs
    print("\nOutput Summary:")
    print(f"  Local features (SAM):   {results['local_features'].shape}")
    print(f"  Global features (CLIP): {results['global_features'].shape}")
    print(f"\nSample global features (first 5 tokens, first 5 dims):")
    print(results["global_features"][0, :5, :5])

    # Save the rendered image
    results["image"].save("output_image.png")
    print("\nRendered image saved to: output_image.png")
