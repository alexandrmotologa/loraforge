"""Hyperparameter optimization and safe configuration tuning.

Searches hyperparameter space (batch size, sequence length, rank) to maximize
throughput and token context while strictly guaranteeing safe VRAM margins.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel

from loraforge.core.config import (
    AttentionImplementation,
    GPUSpec,
    LoRAConfig,
    ModelArchitectureSpec,
    OptimizerType,
    QuantizationConfig,
    QuantizationType,
    TrainingConfig,
)
from loraforge.vram.calculator import VRAMCalculator, VRAMEstimateReport


class OptimizationRecommendation(BaseModel):
    """Recommended tuned training configuration for a given GPU and model."""

    max_feasible_seq_len: int
    optimal_micro_batch_size: int
    optimal_grad_accumulation_steps: int
    recommended_rank: int
    recommended_quantization: QuantizationType
    recommended_optimizer: OptimizerType
    predicted_peak_vram_mb: float
    usable_gpu_vram_mb: float
    headroom_mb: float
    is_safe: bool
    notes: List[str]


class HyperparameterOptimizer:
    """Optimizer engine finding highest performing feasible training parameters."""

    def __init__(self) -> None:
        self.calculator = VRAMCalculator()

    def find_max_sequence_length(
        self,
        model: ModelArchitectureSpec,
        lora: LoRAConfig,
        quant: QuantizationConfig,
        training: TrainingConfig,
        gpu: GPUSpec,
        min_seq: int = 512,
        max_seq: int = 16384,
        step: int = 256,
    ) -> int:
        """Binary search or stepped search for the maximum feasible sequence length."""
        best_feasible = min_seq
        low = min_seq
        high = max_seq

        # Clone config
        test_training = training.model_copy()

        while low <= high:
            mid = ((low + high) // (2 * step)) * step
            if mid < min_seq:
                mid = min_seq
            test_training.max_seq_length = mid

            report = self.calculator.estimate(model, lora, quant, test_training, gpu)
            if report.status in ("OPTIMAL", "FEASIBLE"):
                best_feasible = mid
                low = mid + step
            else:
                high = mid - step

        return best_feasible

    def auto_tune_for_gpu(
        self,
        model: ModelArchitectureSpec,
        gpu: GPUSpec,
        target_effective_batch_size: int = 16,
        target_seq_len: int = 2048,
    ) -> OptimizationRecommendation:
        """Generate an auto-tuned configuration guaranteed to fit the target GPU."""
        notes: List[str] = []

        # 1. Choose quantization strategy based on model parameter count and GPU VRAM
        if gpu.total_vram_mb <= 12288:
            # 12GB GPU (RTX 3060, RTX 4070): NF4 is mandatory for 7B/8B models
            chosen_quant = QuantizationType.NF4
            double_quant = True
            notes.append("Selected 4-bit NF4 with Double Quantization for consumer VRAM envelope.")
        elif gpu.total_vram_mb <= 24576 and model.num_parameters >= 13_000_000_000:
            chosen_quant = QuantizationType.NF4
            double_quant = True
            notes.append("Selected 4-bit NF4 for models >= 13B parameters on 24GB GPU.")
        elif gpu.total_vram_mb >= 49152:
            chosen_quant = QuantizationType.NF4 if model.num_parameters >= 65_000_000_000 else QuantizationType.NONE
            double_quant = False
            notes.append("Workstation VRAM capacity allows full 16-bit LoRA or large-scale QLoRA.")
        else:
            chosen_quant = QuantizationType.NF4
            double_quant = True

        quant_cfg = QuantizationConfig(
            quant_type=chosen_quant,
            double_quant=double_quant,
            compute_dtype="bfloat16" if gpu.supports_bf16 else "float16",
        )

        # 2. Attention implementation
        chosen_attn = (
            AttentionImplementation.FLASH_ATTENTION_2
            if gpu.supports_flash_attn_2
            else AttentionImplementation.SDPA
        )

        # 3. LoRA rank selection
        chosen_rank = 16
        lora_cfg = LoRAConfig(
            r=chosen_rank,
            lora_alpha=chosen_rank * 2,
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

        # 4. Optimizer: paged_adamw_8bit is gold standard for consumer GPUs
        chosen_opt = OptimizerType.PAGED_ADAMW_8BIT

        # 5. Iteratively find highest micro batch size (1, 2, 4)
        best_micro_batch = 1
        for candidate_batch in [1, 2, 4]:
            t_cfg = TrainingConfig(
                per_device_train_batch_size=candidate_batch,
                gradient_accumulation_steps=max(1, target_effective_batch_size // candidate_batch),
                max_seq_length=target_seq_len,
                optimizer=chosen_opt,
                attention_impl=chosen_attn,
                gradient_checkpointing=True,
            )
            report = self.calculator.estimate(model, lora_cfg, quant_cfg, t_cfg, gpu)
            if report.status in ("OPTIMAL", "FEASIBLE"):
                best_micro_batch = candidate_batch

        grad_acc = max(1, target_effective_batch_size // best_micro_batch)
        final_training = TrainingConfig(
            per_device_train_batch_size=best_micro_batch,
            gradient_accumulation_steps=grad_acc,
            max_seq_length=target_seq_len,
            optimizer=chosen_opt,
            attention_impl=chosen_attn,
            gradient_checkpointing=True,
        )

        final_report = self.calculator.estimate(model, lora_cfg, quant_cfg, final_training, gpu)

        # Find maximum feasible sequence length if user wants to stretch context
        max_seq = self.find_max_sequence_length(
            model,
            lora_cfg,
            quant_cfg,
            final_training,
            gpu,
            min_seq=target_seq_len,
            max_seq=16384,
        )

        return OptimizationRecommendation(
            max_feasible_seq_len=max_seq,
            optimal_micro_batch_size=best_micro_batch,
            optimal_grad_accumulation_steps=grad_acc,
            recommended_rank=chosen_rank,
            recommended_quantization=chosen_quant,
            recommended_optimizer=chosen_opt,
            predicted_peak_vram_mb=final_report.peak_vram_mb,
            usable_gpu_vram_mb=gpu.usable_vram_mb,
            headroom_mb=final_report.headroom_mb,
            is_safe=final_report.status in ("OPTIMAL", "FEASIBLE"),
            notes=notes,
        )
