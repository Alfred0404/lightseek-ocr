import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from DeepDecoder import DeepDecoder


def test_decoder_init():
    print("Testing Initialization...")
    decoder = DeepDecoder(verbose=True)
    assert decoder.model is not None
    assert decoder.tokenizer is not None
    print("Initialization OK")
    return decoder


def test_forward(decoder):
    print("\nTesting Forward Pass...")
    B = 2
    local_f = torch.randn(B, 256, 768).to(decoder.device)
    global_f = torch.randn(B, 256, 768).to(decoder.device)

    text = ["Hello world", "Testing decoder"]
    inputs = decoder.tokenizer(text, return_tensors="pt", padding=True).to(
        decoder.device
    )

    outputs = decoder(
        local_features=local_f,
        global_features=global_f,
        text_input_ids=inputs.input_ids,
        text_attention_mask=inputs.attention_mask,
    )

    # Logits shape should be (B, Visual+Text_Len, Vocab)
    # Visual len = 512
    seq_len = inputs.input_ids.shape[1]
    expected_len = 512 + seq_len

    print(f"Logits shape: {outputs.logits.shape}")


import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from DeepDecoder import DeepDecoder


def test_decoder_init():
    print("Testing Initialization...")
    decoder = DeepDecoder(verbose=True)
    assert decoder.model is not None
    assert decoder.tokenizer is not None
    print("Initialization OK")
    return decoder


def test_forward(decoder):
    print("\nTesting Forward Pass...")
    B = 2
    local_f = torch.randn(B, 256, 768).to(decoder.device)
    global_f = torch.randn(B, 256, 768).to(decoder.device)

    text = ["Hello world", "Testing decoder"]
    inputs = decoder.tokenizer(text, return_tensors="pt", padding=True).to(
        decoder.device
    )

    outputs = decoder(
        local_features=local_f,
        global_features=global_f,
        text_input_ids=inputs.input_ids,
        text_attention_mask=inputs.attention_mask,
    )

    # Logits shape should be (B, Visual+Text_Len, Vocab)
    # Visual len = 512
    seq_len = inputs.input_ids.shape[1]
    expected_len = 512 + seq_len

    print(f"Logits shape: {outputs.logits.shape}")
    assert outputs.logits.shape[0] == B
    assert outputs.logits.shape[1] == expected_len
    assert outputs.logits.shape[2] == decoder.model.config.vocab_size
    print("Forward Pass OK")


def test_decode(decoder):
    print("\nTesting Decoding...")
    B = 1
    local_f = torch.randn(B, 256, 768).to(decoder.device)
    global_f = torch.randn(B, 256, 768).to(decoder.device)

    generated = decoder.decode(local_f, global_f, max_new_tokens=10)
    print(f"Decoded output: {generated}")
    assert isinstance(generated, list)
    assert len(generated) == B
    print("Decoding OK")


if __name__ == "__main__":
    decoder = test_decoder_init()
    test_forward(decoder)
    test_decode(decoder)
    print("\nALL TESTS PASSED")
