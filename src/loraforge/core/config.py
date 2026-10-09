"""Core configuration models and schemas for LoRAForge.

Defines Pydantic models for quantization parameters, LoRA matrices, training
hyperparameters, model architectures, GPU targets, datasets, and unified fine-tuning jobs.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class QuantizationType(str, Enum):
    """Supported weight quantization strategies for local fine-tuning."""

    NONE = "none"  # Full precision (16-bit FP16 or BF16)
    INT8 = "int8"  # 8-bit quantization
    NF4 = "nf4"    # NormalFloat4 (QLoRA standard)
    FP4 = "fp4"    # Float4


class OptimizerType(str, Enum):
    """Supported optimizer implementations."""

    ADAMW_TORCH = "adamw_torch"
    ADAMW_8BIT = "adamw_8bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_ADAMW_32BIT = "paged_adamw_32bit"
    LION_8BIT = "lion_8bit"
    SGD = "sgd"


class AttentionImplementation(str, Enum):
    """Attention kernel implementations."""

    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"
    EAGER = "eager"


class DistributedStrategy(str, Enum):
    """Distributed training and memory sharding strategies."""

    NONE = "none"                # Single GPU execution
    DDP = "ddp"                  # Standard DistributedDataParallel
    FSDP_FULL_SHARD = "fsdp"     # PyTorch Fully Sharded Data Parallel
    ZERO_1 = "zero_1"            # DeepSpeed ZeRO-1 (Optimizer state partitioning)
    ZERO_2 = "zero_2"            # DeepSpeed ZeRO-2 (Optimizer + Gradient partitioning)
    ZERO_3 = "zero_3"            # DeepSpeed ZeRO-3 (Full parameter, gradient, optimizer partitioning)


class DatasetFormat(str, Enum):
    """Supported input dataset schema formats."""

    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CHATML = "chatml"
    OPENAI_MESSAGES = "openai_messages"
    RAW_TEXT = "raw_text"


class ExportTarget(str, Enum):
    """Export artifacts for post-training deployment."""

    SAFETENSORS_16BIT = "safetensors_16bit"
    GGUF = "gguf"
    INFEROPS_BUNDLE = "inferops_bundle"


class QuantizationConfig(BaseModel):
    """Configuration for BitsAndBytes quantization."""

    quant_type: QuantizationType = Field(
        default=QuantizationType.NF4,
        description="Quantization algorithm for frozen base model weights.",
    )
    double_quant: bool = Field(
        default=True,
        description="Compress second-level quantization constants to save ~0.37 bits/param.",
    )
    compute_dtype: str = Field(
        default="bfloat16",
        description="Floating-point compute data type during forward/backward passes.",
    )
    quantize_lm_head: bool = Field(
        default=False,
        description="Whether to quantize the language modeling head (keeping 16-bit preserves token distribution).",
    )

    @field_validator("compute_dtype")
    @classmethod
    def validate_compute_dtype(cls, v: str) -> str:
        valid = {"bfloat16", "float16", "float32"}
        if v not in valid:
            raise ValueError(f"compute_dtype must be one of {valid}, got '{v}'")
        return v


class LoRAConfig(BaseModel):
    """Configuration for Low-Rank Adaptation (LoRA) adapters."""

    r: int = Field(
        default=16,
        ge=1,
        le=512,
        description="LoRA decomposition rank dimension.",
    )
    lora_alpha: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="LoRA scaling factor alpha (effective scale is alpha / r).",
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Dropout probability applied to LoRA adapter input activations.",
    )
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Names of linear weight projections to attach adapters to.",
    )
    bias: str = Field(
        default="none",
        description="Trainable bias configuration ('none', 'all', 'lora_only').",
    )
    task_type: str = Field(
        default="CAUSAL_LM",
        description="Target model task type.",
    )
    modules_to_save: List[str] = Field(
        default_factory=list,
        description="Full weight modules to unfreeze and train completely alongside adapters.",
    )

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v: str) -> str:
        valid = {"none", "all", "lora_only"}
        if v not in valid:
            raise ValueError(f"bias must be one of {valid}, got '{v}'")
        return v


class TrainingConfig(BaseModel):
    """Hyperparameters and runtime settings for training loop execution."""

    per_device_train_batch_size: int = Field(
        default=1,
        ge=1,
        description="Micro-batch size per GPU forward pass.",
    )
    gradient_accumulation_steps: int = Field(
        default=8,
        ge=1,
        description="Number of backward passes to accumulate before optimizer step.",
    )
    max_seq_length: int = Field(
        default=2048,
        ge=64,
        le=131072,
        description="Maximum sequence token length for attention window.",
    )
    learning_rate: float = Field(
        default=2e-4,
        gt=0.0,
        le=0.1,
        description="Initial learning rate for LoRA weights.",
    )
    lr_scheduler_type: str = Field(
        default="cosine",
        description="Learning rate schedule decay strategy.",
    )
    warmup_ratio: float = Field(
        default=0.03,
        ge=0.0,
        le=0.5,
        description="Ratio of total steps dedicated to linear warmup.",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Decoupled weight decay coefficient.",
    )
    optimizer: OptimizerType = Field(
        default=OptimizerType.PAGED_ADAMW_8BIT,
        description="Optimizer selection.",
    )
    attention_impl: AttentionImplementation = Field(
        default=AttentionImplementation.FLASH_ATTENTION_2,
        description="Attention mechanism implementation.",
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description="Whether to discard intermediate activations and recompute on backward pass.",
    )
    fp16: bool = Field(
        default=False,
        description="Use FP16 mixed precision.",
    )
    bf16: bool = Field(
        default=True,
        description="Use BF16 mixed precision (requires Ampere or newer).",
    )
    logging_steps: int = Field(
        default=1,
        ge=1,
        description="Frequency of telemetry logging in training steps.",
    )
    save_steps: int = Field(
        default=50,
        ge=1,
        description="Frequency of checkpoint saving in training steps.",
    )
    save_total_limit: int = Field(
        default=2,
        ge=1,
        description="Maximum number of historical checkpoints to retain on disk.",
    )
    num_train_epochs: int = Field(
        default=3,
        ge=1,
        description="Total training epochs over dataset.",
    )
    max_steps: int = Field(
        default=-1,
        description="Maximum training steps (-1 uses num_train_epochs).",
    )
    seed: int = Field(
        default=42,
        description="Random seed for data shuffling and initialization.",
    )
    dataloader_num_workers: int = Field(
        default=0,
        ge=0,
        description="Subprocess workers for data loading.",
    )
    distributed_strategy: DistributedStrategy = Field(
        default=DistributedStrategy.NONE,
        description="Distributed parallelism and memory sharding strategy.",
    )
    num_gpus: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Number of identical GPUs participating in the training run.",
    )
    eval_steps: int = Field(
        default=25,
        ge=1,
        description="Frequency of evaluation steps on validation split.",
    )
    early_stopping_patience: Optional[int] = Field(
        default=3,
        ge=1,
        description="Evaluation intervals without val_loss improvement before early termination.",
    )
    early_stopping_threshold: float = Field(
        default=0.005,
        ge=0.0,
        description="Minimum reduction in val_loss required to qualify as improvement.",
    )

    @property
    def effective_batch_size(self) -> int:
        """Returns effective batch size (micro_batch * gradient_accumulation * num_gpus)."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps * self.num_gpus


class DatasetConfig(BaseModel):
    """Specification of training dataset sources and preprocessing rules."""

    path: str = Field(
        description="Local path or HuggingFace hub identifier for the dataset.",
    )
    format: DatasetFormat = Field(
        default=DatasetFormat.SHAREGPT,
        description="Data structure format for instruction/chat parsing.",
    )
    train_split: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Fraction of dataset allocated to training.",
    )
    val_split: float = Field(
        default=0.05,
        ge=0.0,
        lt=1.0,
        description="Fraction of dataset allocated to evaluation.",
    )
    packing: bool = Field(
        default=True,
        description="Whether to pack multiple short samples into max_seq_length blocks without padding.",
    )
    max_samples: Optional[int] = Field(
        default=None,
        description="Optional cap on processed samples for quick verification runs.",
    )

    @model_validator(mode="after")
    def validate_splits(self) -> DatasetConfig:
        if round(self.train_split + self.val_split, 4) > 1.0:
            raise ValueError(
                f"train_split ({self.train_split}) + val_split ({self.val_split}) cannot exceed 1.0"
            )
        return self


class ModelArchitectureSpec(BaseModel):
    """Analytical structural dimensions of a transformer model."""

    name: str = Field(
        description="Model identifier, e.g. 'meta-llama/Llama-3.1-8B-Instruct'.",
    )
    num_parameters: int = Field(
        ge=100_000,
        description="Total parameter count of base model.",
    )
    num_layers: int = Field(
        ge=1,
        description="Number of transformer decoder blocks (L).",
    )
    hidden_size: int = Field(
        ge=128,
        description="Hidden layer dimension (d_model).",
    )
    intermediate_size: int = Field(
        ge=128,
        description="Feed-forward intermediate dimension (d_mlp).",
    )
    num_attention_heads: int = Field(
        ge=1,
        description="Number of query attention heads (h_q).",
    )
    num_key_value_heads: int = Field(
        ge=1,
        description="Number of key/value heads for GQA/MQA (h_kv).",
    )
    vocab_size: int = Field(
        ge=1000,
        description="Vocabulary size (V).",
    )
    max_position_embeddings: int = Field(
        default=8192,
        ge=512,
        description="Native context window size.",
    )
    tie_word_embeddings: bool = Field(
        default=False,
        description="Whether embedding layer shares weights with lm_head.",
    )

    @property
    def head_dim(self) -> int:
        """Returns dimension per attention head."""
        return self.hidden_size // self.num_attention_heads


class GPUSpec(BaseModel):
    """Hardware capabilities and memory limits of target accelerator."""

    name: str = Field(
        description="GPU model name, e.g. 'NVIDIA GeForce RTX 3060'.",
    )
    total_vram_mb: int = Field(
        ge=2048,
        description="Total physical video memory in megabytes.",
    )
    compute_capability_major: int = Field(
        ge=6,
        description="CUDA compute capability major version (e.g. 8 for Ampere).",
    )
    compute_capability_minor: int = Field(
        ge=0,
        description="CUDA compute capability minor version (e.g. 6 for RTX 3060).",
    )
    supports_bf16: bool = Field(
        default=True,
        description="Whether hardware natively executes BFloat16 math.",
    )
    supports_flash_attn_2: bool = Field(
        default=True,
        description="Whether hardware architecture supports FlashAttention-2 kernels.",
    )
    safe_vram_headroom_mb: int = Field(
        default=768,
        ge=128,
        description="Safety buffer margin reserved for OS display server and driver context.",
    )

    @property
    def compute_capability(self) -> float:
        return float(f"{self.compute_capability_major}.{self.compute_capability_minor}")

    @property
    def usable_vram_mb(self) -> float:
        """Available VRAM after subtracting driver safety margin."""
        return max(0, self.total_vram_mb - self.safe_vram_headroom_mb)


class ExportConfig(BaseModel):
    """Post-training artifact export and InferOps bridge configuration."""

    target: ExportTarget = Field(
        default=ExportTarget.INFEROPS_BUNDLE,
        description="Target export format.",
    )
    output_dir: str = Field(
        default="./exports",
        description="Destination directory for exported weights and manifests.",
    )
    merge_lora: bool = Field(
        default=True,
        description="Merge adapter weights back into full precision base model weights.",
    )
    gguf_quantization: str = Field(
        default="Q4_K_M",
        description="GGUF quantization scheme if target is GGUF (e.g. Q4_K_M, Q8_0).",
    )
    generate_inferops_manifest: bool = Field(
        default=True,
        description="Generate YAML model definition ready for InferOps server deployment.",
    )
    inferops_model_id: Optional[str] = Field(
        default=None,
        description="Custom model identifier string for InferOps registry.",
    )


class FineTuningJobConfig(BaseModel):
    """Unified configuration contract for a complete LoRAForge fine-tuning job."""

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for the fine-tuning run.",
    )
    job_name: str = Field(
        default="loraforge-experiment",
        description="Human-readable job label.",
    )
    model: ModelArchitectureSpec = Field(
        description="Base model architecture specification.",
    )
    lora: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description="LoRA adapter hyperparameters.",
    )
    quantization: QuantizationConfig = Field(
        default_factory=QuantizationConfig,
        description="Base model quantization scheme.",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training hyperparameters and loop settings.",
    )
    dataset: DatasetConfig = Field(
        description="Dataset source and preprocessing parameters.",
    )
    gpu: GPUSpec = Field(
        description="Target GPU specification.",
    )
    export: ExportConfig = Field(
        default_factory=ExportConfig,
        description="Export and packaging configuration.",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def to_yaml(self) -> str:
        """Serialize configuration to clean YAML string."""
        return yaml.dump(self.model_dump(mode="json"), sort_keys=False, default_flow_style=False)

    def save_yaml(self, path: Path | str) -> None:
        """Write configuration to a YAML file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_yaml(cls, path_or_content: Path | str) -> FineTuningJobConfig:
        """Load configuration from a YAML file path or YAML content string."""
        text = str(path_or_content)
        path = Path(text)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(text)
        return cls(**data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.model_dump(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> FineTuningJobConfig:
        """Parse configuration from JSON string."""
        data = json.loads(json_str)
        return cls(**data)
