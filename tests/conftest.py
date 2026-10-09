"""Shared pytest fixtures for LoRAForge test suite."""

import pytest
from loraforge.core.config import (
    GPUSpec,
    LoRAConfig,
    ModelArchitectureSpec,
    QuantizationConfig,
    QuantizationType,
    TrainingConfig,
)
from loraforge.core.hardware import PRESET_GPUS
from loraforge.core.models import PRESET_MODELS


@pytest.fixture
def llama3_8b_spec() -> ModelArchitectureSpec:
    """Standard Meta Llama 3.1 8B Instruct model spec."""
    return PRESET_MODELS["meta-llama/Llama-3.1-8B-Instruct"]


@pytest.fixture
def rtx_3060_spec() -> GPUSpec:
    """Consumer GPU: RTX 3060 12GB spec."""
    return PRESET_GPUS["rtx-3060-12gb"]


@pytest.fixture
def rtx_4090_spec() -> GPUSpec:
    """Workstation GPU: RTX 4090 24GB spec."""
    return PRESET_GPUS["rtx-4090-24gb"]


@pytest.fixture
def default_lora_config() -> LoRAConfig:
    """Standard rank-16 all-linear LoRA configuration."""
    return LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


@pytest.fixture
def qlora_nf4_config() -> QuantizationConfig:
    """Standard 4-bit NF4 double quantization configuration."""
    return QuantizationConfig(
        quant_type=QuantizationType.NF4,
        double_quant=True,
        compute_dtype="bfloat16",
    )


@pytest.fixture
def standard_training_config() -> TrainingConfig:
    """Standard consumer training hyperparameters."""
    return TrainingConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_seq_length=2048,
        gradient_checkpointing=True,
    )
