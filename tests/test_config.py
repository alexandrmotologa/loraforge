"""Unit tests for configuration schemas and serialization."""

import pytest
from pydantic import ValidationError

from loraforge.core.config import (
    DatasetConfig,
    DatasetFormat,
    ExportConfig,
    ExportTarget,
    FineTuningJobConfig,
    GPUSpec,
    LoRAConfig,
    ModelArchitectureSpec,
    OptimizerType,
    QuantizationConfig,
    QuantizationType,
    TrainingConfig,
)


def test_quantization_config_validation():
    cfg = QuantizationConfig(quant_type=QuantizationType.NF4, compute_dtype="bfloat16")
    assert cfg.quant_type == QuantizationType.NF4
    assert cfg.double_quant is True

    with pytest.raises(ValidationError):
        QuantizationConfig(compute_dtype="int4")


def test_lora_config_validation():
    cfg = LoRAConfig(r=32, lora_alpha=64, bias="none")
    assert cfg.r == 32
    assert cfg.lora_alpha == 64

    with pytest.raises(ValidationError):
        LoRAConfig(r=0)  # ge=1

    with pytest.raises(ValidationError):
        LoRAConfig(bias="invalid_bias")


def test_training_config_effective_batch():
    cfg = TrainingConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
    )
    assert cfg.effective_batch_size == 8


def test_dataset_config_splits():
    cfg = DatasetConfig(
        path="data.jsonl",
        format=DatasetFormat.SHAREGPT,
        train_split=0.9,
        val_split=0.1,
    )
    assert cfg.train_split == 0.9

    with pytest.raises(ValidationError):
        DatasetConfig(
            path="data.jsonl",
            train_split=0.8,
            val_split=0.3,  # sum > 1.0
        )


def test_gpu_spec_properties():
    gpu = GPUSpec(
        name="RTX Test",
        total_vram_mb=12288,
        compute_capability_major=8,
        compute_capability_minor=6,
        safe_vram_headroom_mb=768,
    )
    assert gpu.compute_capability == 8.6
    assert gpu.usable_vram_mb == 12288 - 768


def test_job_config_yaml_json_serialization(tmp_path, llama3_8b_spec, rtx_3060_spec):
    cfg = FineTuningJobConfig(
        job_name="test-run",
        model=llama3_8b_spec,
        gpu=rtx_3060_spec,
        dataset=DatasetConfig(path="train.jsonl", format=DatasetFormat.ALPACA),
    )

    yaml_str = cfg.to_yaml()
    assert "test-run" in yaml_str
    assert "meta-llama/Llama-3.1-8B-Instruct" in yaml_str

    loaded_from_yaml = FineTuningJobConfig.from_yaml(yaml_str)
    assert loaded_from_yaml.job_name == "test-run"
    assert loaded_from_yaml.model.num_layers == 32

    json_str = cfg.to_json()
    loaded_from_json = FineTuningJobConfig.from_json(json_str)
    assert loaded_from_json.model.hidden_size == 4096

    # Test file saving
    save_path = tmp_path / "config.yaml"
    cfg.save_yaml(save_path)
    assert save_path.is_file()
    loaded_from_file = FineTuningJobConfig.from_yaml(save_path)
    assert loaded_from_file.job_name == "test-run"
