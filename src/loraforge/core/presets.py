"""Turnkey fine-tuning presets for specialized AI tasks.

Provides tuned hyperparameters for:
- Mathematical & CoT reasoning (DeepSeek-R1 style)
- Software engineering & code generation
- Agentic tool use & JSON structured outputs
- General conversational instruction following
"""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel

from loraforge.core.config import (
    AttentionImplementation,
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
from loraforge.core.hardware import PRESET_GPUS
from loraforge.core.models import PRESET_MODELS


class FineTuningPreset(BaseModel):
    """Specification of a task-specific fine-tuning preset."""

    name: str
    description: str
    recommended_model: str
    lora: LoRAConfig
    training: TrainingConfig
    quantization: QuantizationConfig


PRESETS: Dict[str, FineTuningPreset] = {
    "reasoning": FineTuningPreset(
        name="reasoning",
        description="Chain-of-thought mathematical and logical deduction (DeepSeek-R1 style).",
        recommended_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        lora=LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        training=TrainingConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            max_seq_length=4096,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            optimizer=OptimizerType.PAGED_ADAMW_8BIT,
            attention_impl=AttentionImplementation.FLASH_ATTENTION_2,
            gradient_checkpointing=True,
        ),
        quantization=QuantizationConfig(quant_type=QuantizationType.NF4, double_quant=True),
    ),
    "coding": FineTuningPreset(
        name="coding",
        description="Software development, syntactical completion, and bug diagnosis.",
        recommended_model="Qwen/Qwen2.5-7B-Instruct",
        lora=LoRAConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        training=TrainingConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_seq_length=4096,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optimizer=OptimizerType.PAGED_ADAMW_8BIT,
            attention_impl=AttentionImplementation.FLASH_ATTENTION_2,
            gradient_checkpointing=True,
        ),
        quantization=QuantizationConfig(quant_type=QuantizationType.NF4, double_quant=True),
    ),
    "agentic": FineTuningPreset(
        name="agentic",
        description="Function calling, external tool usage, and structured JSON output schema conformance.",
        recommended_model="meta-llama/Llama-3.1-8B-Instruct",
        lora=LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        training=TrainingConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_seq_length=3072,
            learning_rate=1.5e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optimizer=OptimizerType.PAGED_ADAMW_8BIT,
            attention_impl=AttentionImplementation.FLASH_ATTENTION_2,
            gradient_checkpointing=True,
        ),
        quantization=QuantizationConfig(quant_type=QuantizationType.NF4, double_quant=True),
    ),
    "chat": FineTuningPreset(
        name="chat",
        description="General dialogue, conversational instruction following, and domain personalization.",
        recommended_model="meta-llama/Llama-3.1-8B-Instruct",
        lora=LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        training=TrainingConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_seq_length=2048,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            optimizer=OptimizerType.PAGED_ADAMW_8BIT,
            attention_impl=AttentionImplementation.FLASH_ATTENTION_2,
            gradient_checkpointing=True,
        ),
        quantization=QuantizationConfig(quant_type=QuantizationType.NF4, double_quant=True),
    ),
}


def get_preset(name: str) -> Optional[FineTuningPreset]:
    """Look up a task preset by name."""
    norm = name.lower().strip()
    return PRESETS.get(norm)


def build_config_from_preset(
    preset_name: str,
    gpu_name: str = "rtx-3060-12gb",
    dataset_path: str = "train.jsonl",
    dataset_format: DatasetFormat = DatasetFormat.SHAREGPT,
    job_name: Optional[str] = None,
) -> FineTuningJobConfig:
    """Build a ready-to-run FineTuningJobConfig from a task preset."""
    preset = get_preset(preset_name)
    if not preset:
        preset = PRESETS["chat"]

    model_spec = PRESET_MODELS.get(preset.recommended_model, PRESET_MODELS["meta-llama/Llama-3.1-8B-Instruct"])
    gpu_spec = PRESET_GPUS.get(gpu_name, PRESET_GPUS["rtx-3060-12gb"])

    return FineTuningJobConfig(
        job_name=job_name or f"{preset.name}-experiment",
        model=model_spec,
        lora=preset.lora,
        quantization=preset.quantization,
        training=preset.training,
        dataset=DatasetConfig(path=dataset_path, format=dataset_format),
        gpu=gpu_spec,
        export=ExportConfig(target=ExportTarget.INFEROPS_BUNDLE),
    )
