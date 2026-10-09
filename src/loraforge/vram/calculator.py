"""Predictive Training VRAM Calculator for LoRA & QLoRA Fine-Tuning.

Provides exact analytical calculations of base model weights, quantization scaling overhead,
LoRA trainable matrices, gradient buffers, optimizer states, and forward/backward activation memory.
Zero mocks or placeholders.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

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


class LoRAParameterBreakdown(BaseModel):
    """Detailed parameter count breakdown for LoRA adapter modules."""

    target_modules: List[str]
    rank: int
    trainable_parameters: int
    total_base_parameters: int
    trainable_percentage: float
    parameters_per_layer: int
    num_layers: int


class VRAMBreakdown(BaseModel):
    """Component-level video memory consumption in megabytes."""

    cuda_context_mb: float = Field(
        description="PyTorch CUDA runtime, cuBLAS workspace, and driver baseline overhead.",
    )
    base_model_weights_mb: float = Field(
        description="Quantized or full-precision base model parameters in HBM.",
    )
    quantization_metadata_mb: float = Field(
        description="Block quantization scales and double-quantization constants.",
    )
    lora_weights_mb: float = Field(
        description="LoRA A and B adapter matrices in compute precision (FP16/BF16).",
    )
    gradients_mb: float = Field(
        description="Gradient buffers accumulated for trainable parameters.",
    )
    optimizer_states_mb: float = Field(
        description="Optimizer momentum and variance states across parameter steps.",
    )
    activations_mb: float = Field(
        description="Forward pass stashed activations and peak intra-block backward tensors.",
    )
    fragmentation_buffer_mb: float = Field(
        description="Headroom reserved for PyTorch caching allocator fragmentation.",
    )

    @property
    def total_peak_mb(self) -> float:
        """Total estimated peak VRAM consumption."""
        return round(
            self.cuda_context_mb
            + self.base_model_weights_mb
            + self.quantization_metadata_mb
            + self.lora_weights_mb
            + self.gradients_mb
            + self.optimizer_states_mb
            + self.activations_mb
            + self.fragmentation_buffer_mb,
            2,
        )


class VRAMEstimateReport(BaseModel):
    """Comprehensive analytical report on training feasibility and memory profile."""

    model_name: str
    gpu_name: str
    total_gpu_vram_mb: int
    usable_gpu_vram_mb: float
    breakdown: VRAMBreakdown
    lora_parameters: LoRAParameterBreakdown
    peak_vram_mb: float
    headroom_mb: float
    utilization_percentage: float
    status: str  # OPTIMAL, FEASIBLE, HIGH_RISK, OOM
    recommendations: List[str]


class VRAMCalculator:
    """Analytical predictive training VRAM profiler."""

    # CUDA runtime context baseline per major architecture (in MB)
    CUDA_CONTEXT_BASELINES: Dict[int, float] = {
        6: 500.0,  # Pascal
        7: 600.0,  # Volta / Turing
        8: 680.0,  # Ampere / Ada Lovelace
        9: 820.0,  # Hopper
    }

    # Caching allocator fragmentation safety coefficient
    FRAGMENTATION_COEFFICIENT: float = 0.12

    def calculate_lora_parameters(
        self,
        model: ModelArchitectureSpec,
        lora: LoRAConfig,
    ) -> LoRAParameterBreakdown:
        """Compute the exact trainable parameter count of LoRA adapter matrices.

        For a linear projection W in R^(d_out x d_in), low-rank decomposition creates:
        A in R^(r x d_in) and B in R^(d_out x r).
        Total parameters for that projection = r * (d_in + d_out).
        """
        d_model = model.hidden_size
        d_mlp = model.intermediate_size
        head_dim = model.head_dim
        d_kv = model.num_key_value_heads * head_dim
        r = lora.r

        params_per_layer = 0

        for mod in lora.target_modules:
            normalized = mod.lower().strip()
            if normalized in ("q_proj", "q"):
                # q_proj: (d_model -> d_model)
                params_per_layer += r * (d_model + d_model)
            elif normalized in ("k_proj", "k"):
                # k_proj: (d_model -> d_kv)
                params_per_layer += r * (d_model + d_kv)
            elif normalized in ("v_proj", "v"):
                # v_proj: (d_model -> d_kv)
                params_per_layer += r * (d_model + d_kv)
            elif normalized in ("o_proj", "o", "out_proj"):
                # o_proj: (d_model -> d_model)
                params_per_layer += r * (d_model + d_model)
            elif normalized in ("gate_proj", "gate", "w1"):
                # gate_proj: (d_model -> d_mlp)
                params_per_layer += r * (d_model + d_mlp)
            elif normalized in ("up_proj", "up", "w3"):
                # up_proj: (d_model -> d_mlp)
                params_per_layer += r * (d_model + d_mlp)
            elif normalized in ("down_proj", "down", "w2"):
                # down_proj: (d_mlp -> d_model)
                params_per_layer += r * (d_mlp + d_model)
            else:
                # Fallback to standard linear projection (d_model x d_model)
                params_per_layer += r * (d_model + d_model)

        total_lora_params = params_per_layer * model.num_layers

        # Additional unmasked full parameters if modules_to_save specified
        for mod_save in lora.modules_to_save:
            if "embed_tokens" in mod_save or "embed" in mod_save:
                total_lora_params += model.vocab_size * d_model
            elif "lm_head" in mod_save:
                total_lora_params += model.vocab_size * d_model

        pct = (total_lora_params / model.num_parameters) * 100.0

        return LoRAParameterBreakdown(
            target_modules=lora.target_modules,
            rank=r,
            trainable_parameters=total_lora_params,
            total_base_parameters=model.num_parameters,
            trainable_percentage=round(pct, 4),
            parameters_per_layer=params_per_layer,
            num_layers=model.num_layers,
        )

    def calculate_base_model_memory(
        self,
        model: ModelArchitectureSpec,
        quant: QuantizationConfig,
    ) -> tuple[float, float]:
        """Compute base model parameter memory and quantization metadata overhead in MB.

        Returns (base_weights_mb, quantization_metadata_mb).
        """
        total_params = model.num_parameters
        d_model = model.hidden_size
        vocab_size = model.vocab_size

        # In standard QLoRA implementations (e.g. bitsandbytes):
        # Embeddings and lm_head are typically kept in 16-bit (2 bytes/param)
        embed_params = vocab_size * d_model
        head_params = vocab_size * d_model if not model.tie_word_embeddings else 0

        if quant.quant_type == QuantizationType.NONE:
            # Full precision (16-bit)
            bytes_per_param = 2.0
            weights_mb = (total_params * bytes_per_param) / (1024 * 1024)
            return round(weights_mb, 2), 0.0

        if quant.quant_type == QuantizationType.INT8:
            # 8-bit quantization: 1 byte per quantized param + scale vector
            quantized_params = max(0, total_params - embed_params - head_params)
            unquantized_params = total_params - quantized_params
            weights_bytes = (quantized_params * 1.0) + (unquantized_params * 2.0)
            # Int8 block scale overhead: 1 float32 scale per 64 weights
            scale_bytes = (quantized_params / 64.0) * 4.0
            weights_mb = weights_bytes / (1024 * 1024)
            scale_mb = scale_bytes / (1024 * 1024)
            return round(weights_mb, 2), round(scale_mb, 2)

        if quant.quant_type in (QuantizationType.NF4, QuantizationType.FP4):
            # 4-bit NormalFloat (0.5 bytes per parameter)
            if quant.quantize_lm_head:
                quantized_params = total_params - embed_params
                unquantized_params = embed_params
            else:
                quantized_params = max(0, total_params - embed_params - head_params)
                unquantized_params = total_params - quantized_params

            weights_bytes = (quantized_params * 0.5) + (unquantized_params * 2.0)

            # Block quantization constants (block size B = 64)
            num_blocks = quantized_params / 64.0

            if quant.double_quant:
                # Double quantization: first-level scales are quantized to 8-bit (1 byte per block)
                # plus 32-bit second-level scales per 256 blocks
                scale_bytes = (num_blocks * 1.0) + ((num_blocks / 256.0) * 4.0)
            else:
                # Single quantization: FP32 (4 bytes) per block
                scale_bytes = num_blocks * 4.0

            weights_mb = weights_bytes / (1024 * 1024)
            scale_mb = scale_bytes / (1024 * 1024)
            return round(weights_mb, 2), round(scale_mb, 2)

        # Fallback 16-bit
        weights_mb = (total_params * 2.0) / (1024 * 1024)
        return round(weights_mb, 2), 0.0

    def calculate_lora_weights_memory(
        self,
        lora_params: int,
        compute_dtype: str = "bfloat16",
    ) -> float:
        """Memory required to store LoRA adapter weights in GPU HBM in MB."""
        bytes_per_param = 4.0 if compute_dtype == "float32" else 2.0
        return round((lora_params * bytes_per_param) / (1024 * 1024), 2)

    def calculate_gradient_memory(
        self,
        lora_params: int,
        optimizer: OptimizerType,
    ) -> float:
        """Memory required for gradient buffers of trainable parameters in MB.

        Base model frozen weights have requires_grad=False, so zero gradients are stored for them.
        """
        # Gradients are accumulated in 16-bit (2 bytes) or 32-bit (4 bytes)
        bytes_per_grad = 2.0 if optimizer in (
            OptimizerType.ADAMW_8BIT,
            OptimizerType.PAGED_ADAMW_8BIT,
            OptimizerType.LION_8BIT,
        ) else 4.0
        return round((lora_params * bytes_per_grad) / (1024 * 1024), 2)

    def calculate_optimizer_memory(
        self,
        lora_params: int,
        optimizer: OptimizerType,
    ) -> float:
        """Memory required for optimizer state tensors in MB.

        - adamw_torch: 1st moment (4 bytes) + 2nd moment (4 bytes) + master fp32 weights (4 bytes) = 12 bytes/param (or 8 bytes without master)
        - adamw_8bit: 1st moment (1 byte) + 2nd moment (1 byte) + quant scale tables = ~2.008 bytes/param
        - paged_adamw_8bit: ~2.008 bytes/param on GPU (pages to CPU RAM on spikes)
        - lion_8bit: 1st moment (1 byte) = 1.0 byte/param
        - sgd: 4 bytes/param with momentum
        """
        if optimizer == OptimizerType.ADAMW_TORCH:
            bytes_per_param = 8.0
        elif optimizer in (OptimizerType.ADAMW_8BIT, OptimizerType.PAGED_ADAMW_8BIT):
            bytes_per_param = 2.0 + (2.0 * 4.0 / 2048.0)  # 2 bytes + block scales
        elif optimizer == OptimizerType.PAGED_ADAMW_32BIT:
            bytes_per_param = 8.0
        elif optimizer == OptimizerType.LION_8BIT:
            bytes_per_param = 1.0 + (4.0 / 2048.0)
        elif optimizer == OptimizerType.SGD:
            bytes_per_param = 4.0
        else:
            bytes_per_param = 8.0

        return round((lora_params * bytes_per_param) / (1024 * 1024), 2)

    def calculate_activation_memory(
        self,
        model: ModelArchitectureSpec,
        training: TrainingConfig,
    ) -> float:
        """Compute peak forward/backward activation memory in MB.

        Calculates stashed layer inputs under gradient checkpointing and peak intra-block
        recomputation tensors, accounting for FlashAttention-2 vs Standard Attention.
        """
        b = training.per_device_train_batch_size
        s = training.max_seq_length
        L = model.num_layers
        d = model.hidden_size
        d_mlp = model.intermediate_size
        h_q = model.num_attention_heads
        h_kv = model.num_key_value_heads
        head_dim = model.head_dim
        v = model.vocab_size
        bytes_per_elem = 2.0  # FP16 / BF16

        if training.gradient_checkpointing:
            # 1. Stashed activations across checkpoint boundaries:
            # Each decoder layer input tensor is saved: [b, s, d] in 16-bit
            stashed_bytes = L * b * s * d * bytes_per_elem

            # 2. Peak intra-block recomputation tensors in a single layer:
            # Self-attention module:
            if training.attention_impl == AttentionImplementation.FLASH_ATTENTION_2:
                # FlashAttention-2 avoids storing the O(s^2) attention matrix in HBM!
                # It only holds Q [b, s, h_q * head_dim], K [b, s, h_kv * head_dim],
                # V [b, s, h_kv * head_dim], and output [b, s, d]
                attn_peak_bytes = b * s * (d + 2 * (h_kv * head_dim) + d) * bytes_per_elem
            else:
                # SDPA / Eager: stores full QK^T matrix [b, h_q, s, s] + softmax + projections
                attn_matrix_bytes = b * h_q * s * s * bytes_per_elem
                attn_proj_bytes = b * s * (d + 2 * (h_kv * head_dim) + d) * bytes_per_elem
                attn_peak_bytes = attn_matrix_bytes + attn_proj_bytes

            # MLP module (SwiGLU):
            # Input [b, s, d], Gate [b, s, d_mlp], Up [b, s, d_mlp], Down in [b, s, d_mlp]
            mlp_peak_bytes = b * s * (d + 3 * d_mlp) * bytes_per_elem

            # Logits & Cross-Entropy buffer at model output:
            # Chunked cross-entropy typically calculates loss in chunks of 1024 tokens
            chunk_tokens = min(s * b, 1024)
            logits_bytes = chunk_tokens * v * bytes_per_elem

            # Intra-block peak is maximum concurrent operation + logits buffer
            intra_block_bytes = max(attn_peak_bytes, mlp_peak_bytes) + logits_bytes

            total_act_bytes = stashed_bytes + intra_block_bytes

        else:
            # Gradient checkpointing disabled: ALL activations for ALL layers retained
            if training.attention_impl == AttentionImplementation.FLASH_ATTENTION_2:
                attn_per_layer = b * s * (d + 2 * (h_kv * head_dim) + d) * bytes_per_elem
            else:
                attn_per_layer = (b * h_q * s * s + b * s * (d + 2 * (h_kv * head_dim) + d)) * bytes_per_elem

            mlp_per_layer = b * s * (d + 3 * d_mlp) * bytes_per_elem
            norms_per_layer = 2 * b * s * d * bytes_per_elem
            layer_act_bytes = attn_per_layer + mlp_per_layer + norms_per_layer

            logits_bytes = b * s * v * bytes_per_elem
            total_act_bytes = (L * layer_act_bytes) + logits_bytes

        return round(total_act_bytes / (1024 * 1024), 2)

    def calculate_cuda_context_overhead(self, gpu: GPUSpec) -> float:
        """Baseline VRAM reserved by CUDA runtime, driver, and cuBLAS context."""
        major = gpu.compute_capability_major
        return self.CUDA_CONTEXT_BASELINES.get(major, 680.0)

    def estimate(
        self,
        model: ModelArchitectureSpec,
        lora: LoRAConfig,
        quant: QuantizationConfig,
        training: TrainingConfig,
        gpu: GPUSpec,
    ) -> VRAMEstimateReport:
        """Generate a complete predictive VRAM profile and feasibility evaluation."""
        lora_breakdown = self.calculate_lora_parameters(model, lora)
        base_weights_mb, quant_meta_mb = self.calculate_base_model_memory(model, quant)
        lora_weights_mb = self.calculate_lora_weights_memory(
            lora_breakdown.trainable_parameters,
            quant.compute_dtype,
        )
        gradients_mb = self.calculate_gradient_memory(
            lora_breakdown.trainable_parameters,
            training.optimizer,
        )
        optimizer_mb = self.calculate_optimizer_memory(
            lora_breakdown.trainable_parameters,
            training.optimizer,
        )
        activations_mb = self.calculate_activation_memory(model, training)
        cuda_context_mb = self.calculate_cuda_context_overhead(gpu)

        # Dynamic fragmentation buffer (12% of activations + gradients)
        frag_mb = round(
            (activations_mb + gradients_mb + lora_weights_mb) * self.FRAGMENTATION_COEFFICIENT,
            2,
        )

        breakdown = VRAMBreakdown(
            cuda_context_mb=cuda_context_mb,
            base_model_weights_mb=base_weights_mb,
            quantization_metadata_mb=quant_meta_mb,
            lora_weights_mb=lora_weights_mb,
            gradients_mb=gradients_mb,
            optimizer_states_mb=optimizer_mb,
            activations_mb=activations_mb,
            fragmentation_buffer_mb=frag_mb,
        )

        peak_mb = breakdown.total_peak_mb
        total_gpu_mb = gpu.total_vram_mb
        usable_gpu_mb = gpu.usable_vram_mb
        headroom_mb = round(total_gpu_mb - peak_mb, 2)
        util_pct = round((peak_mb / total_gpu_mb) * 100.0, 1)

        # Categorize feasibility
        recommendations: List[str] = []

        if peak_mb > total_gpu_mb:
            status = "OOM"
            deficit = peak_mb - total_gpu_mb
            recommendations.append(
                f"Configuration exceeds total physical VRAM by {deficit:.1f} MB. Training will trigger CUDA OOM."
            )
        elif peak_mb > usable_gpu_mb:
            status = "HIGH_RISK"
            recommendations.append(
                f"Peak memory ({peak_mb:.1f} MB) exceeds safe headroom threshold ({usable_gpu_mb:.1f} MB). High risk of OS display crash."
            )
        elif peak_mb > (total_gpu_mb * 0.85):
            status = "FEASIBLE"
            recommendations.append(
                "Configuration fits within safe VRAM envelope. Monitor during initial training steps."
            )
        else:
            status = "OPTIMAL"
            recommendations.append(
                f"Optimal memory profile with {headroom_mb:.1f} MB free headroom. Stable for extended training runs."
            )

        # Generate actionable hyperparameter advice
        if not training.gradient_checkpointing and status in ("OOM", "HIGH_RISK"):
            recommendations.append(
                "Enable gradient_checkpointing to reduce activation memory by up to 70%."
            )

        if (
            training.attention_impl != AttentionImplementation.FLASH_ATTENTION_2
            and gpu.supports_flash_attn_2
            and training.max_seq_length >= 2048
        ):
            recommendations.append(
                "Switch attention_impl to 'flash_attention_2' to eliminate O(s^2) attention matrix allocation."
            )

        if (
            training.optimizer == OptimizerType.ADAMW_TORCH
            and status in ("OOM", "HIGH_RISK", "FEASIBLE")
        ):
            recommendations.append(
                "Switch optimizer to 'paged_adamw_8bit' to reduce optimizer state footprint by 75%."
            )

        if (
            quant.quant_type == QuantizationType.NONE
            and status in ("OOM", "HIGH_RISK")
        ):
            recommendations.append(
                "Switch quantization to 'nf4' (QLoRA) to compress base model weights by ~75%."
            )

        if training.per_device_train_batch_size > 1 and status in ("OOM", "HIGH_RISK"):
            suggested_acc = training.gradient_accumulation_steps * training.per_device_train_batch_size
            recommendations.append(
                f"Reduce per_device_train_batch_size to 1 and increase gradient_accumulation_steps to {suggested_acc}."
            )

        return VRAMEstimateReport(
            model_name=model.name,
            gpu_name=gpu.name,
            total_gpu_vram_mb=total_gpu_mb,
            usable_gpu_vram_mb=usable_gpu_mb,
            breakdown=breakdown,
            lora_parameters=lora_breakdown,
            peak_vram_mb=peak_mb,
            headroom_mb=headroom_mb,
            utilization_percentage=util_pct,
            status=status,
            recommendations=recommendations,
        )
