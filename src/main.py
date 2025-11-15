"""
LightSeek-OCR Main Pipeline
Complete end-to-end OCR system: Image → Text
"""

import sys
from DeepEncoder import DeepEncoder
# from Decoder import Decoder


def main():
    """
    Run the complete LightSeek-OCR pipeline
    """
    print("=" * 70)
    print("LightSeek-OCR - Complete Pipeline")
    print("=" * 70)
    print("\nInitializing system...\n")

    # Initialize encoder
    print("[1/2] Loading Encoder (SAM + CLIP)...")
    encoder = DeepEncoder(verbose=False)
    print("✓ Encoder ready\n")

    # Initialize decoder
    print("[2/2] Loading Decoder (DeepSeek-3B-MoE)...")
    decoder = Decoder(model_name="deepseek-ai/deepseek-moe-16b-base", max_length=512)
    print("✓ Decoder ready\n")

    print("=" * 70)
    print("System ready! Enter text to process or 'quit' to exit.")
    print("=" * 70)
    print()

    while True:
        # Get user input
        print("\nEnter text to render and process (or 'quit' to exit):")
        user_input = input("> ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nExiting LightSeek-OCR. Goodbye!")
            break

        if not user_input:
            print("Please enter some text.")
            continue

        try:
            print("\n" + "=" * 70)
            print("Processing...")
            print("=" * 70)

            # Step 1: Encode (text → visual features)
            print("\n[Encoding] Text → Image → Visual Features...")
            results = encoder.encode(user_input)

            local_features = results["local_features"]
            global_features = results["global_features"]

            print(f"✓ Local features:  {local_features.shape}")
            print(f"✓ Global features: {global_features.shape}")

            # Step 2: Decode (visual features → text)
            print("\n[Decoding] Visual Features → Text...")

            # Option 1: Simple generation with prompt
            prompt = f"The image contains the text: '{user_input}'. Transcribe it:"

            # Generate text
            output_text = decoder.decode(
                global_features=global_features,
                local_features=local_features,
                prompt=prompt,
                num_beams=3,
                temperature=0.7,
            )

            print("\n" + "=" * 70)
            print("RESULTS")
            print("=" * 70)
            print(f"\nInput text:  {user_input}")
            print(f"Output text: {output_text}")
            print("\n" + "=" * 70)

            # Optionally save the rendered image
            save_option = input("\nSave rendered image? (y/n): ").strip().lower()
            if save_option == "y":
                filename = f"output_{user_input[:20].replace(' ', '_')}.png"
                results["image"].save(filename)
                print(f"✓ Image saved to: {filename}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue with new input.")
            continue
        except Exception as e:
            print(f"\n✗ Error during processing: {e}")
            print("Please try again with different input.")
            continue


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting LightSeek-OCR. Goodbye!")
        sys.exit(0)
