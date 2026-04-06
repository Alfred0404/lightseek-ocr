"""
LoRA configuration for SmolLM2-1.7B-Instruct fine-tuning.
Centralises the config so train.py and any eval script use the same setup.
"""

from peft import LoraConfig, TaskType


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
