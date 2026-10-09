"""Mathematical verification tests for predictive training VRAM calculations."""

import pytest
from loraforge.core.config import (
    AttentionImplementation,
    LoRAConfig,
    OptimizerType,
    QuantizationConfig,
    QuantizationType,
    TrainingConfig,
)
from loraforge.vram.calculator import VRAMCalculator
from loraforge.vram.optimizer import HyperparameterOptimizer


def test_lora_parameter_math(llama3_8b_spec, default_lora_config):
    calc = VRAMCalculator()
    bd = calc.calculate_lora_parameters(llama3_8b_spec, default_lora_config)

    # For Llama-3-8B:
    # d_model = 4096, d_mlp = 14336, head_dim = 128, d_kv = 8 * 128 = 1024
    # r = 16
    # q_proj: 16 * (4096 + 4096) = 131,072
    # k_proj: 16 * (4096 + 1024) = 81,920
    # v_proj: 16 * (4096 + 1024) = 81,920
    # o_proj: 16 * (4096 + 4096) = 131,072
    # gate_proj: 16 * (4096 + 14336) = 294,912
    # up_proj: 16 * (4096 + 14336) = 294,912
    # down_proj: 16 * (14336 + 4096) = 294,912
    # Sum per layer = 1,310,720
    # 32 layers = 41,943,040 params
    assert bd.parameters_per_layer == 1_310_720
    assert bd.trainable_parameters == 41_943_040
    assert bd.trainable_percentage < 1.0  # < 1% of 8B


def test_base_model_quantization_memory(llama3_8b_spec):
    calc = VRAMCalculator()

    # 16-bit Full precision: ~8.03B params * 2 bytes = ~16,060 MB / 1024 = ~15,316 MB
    weights_fp16, meta_fp16 = calc.calculate_base_model_memory(
        llama3_8b_spec,
        QuantizationConfig(quant_type=QuantizationType.NONE),
    )
    assert meta_fp16 == 0.0
    assert 14000 < weights_fp16 < 16500

    # 4-bit NF4 with Double Quantization:
    # Should compress weights to ~4.5 - 5.5 GB
    weights_nf4, meta_nf4 = calc.calculate_base_model_memory(
        llama3_8b_spec,
        QuantizationConfig(quant_type=QuantizationType.NF4, double_quant=True),
    )
    assert meta_nf4 > 0.0  # Double quant block scales present
    assert weights_nf4 < weights_fp16 * 0.40  # Massive reduction


def test_optimizer_memory_scaling(llama3_8b_spec, default_lora_config):
    calc = VRAMCalculator()
    bd = calc.calculate_lora_parameters(llama3_8b_spec, default_lora_config)
    n_params = bd.trainable_parameters

    opt_torch = calc.calculate_optimizer_memory(n_params, OptimizerType.ADAMW_TORCH)
    opt_8bit = calc.calculate_optimizer_memory(n_params, OptimizerType.PAGED_ADAMW_8BIT)
    opt_lion = calc.calculate_optimizer_memory(n_params, OptimizerType.LION_8BIT)

    # 8-bit AdamW should consume ~1/4 of FP32 AdamW
    assert opt_8bit < opt_torch * 0.30
    # Lion 8-bit should consume ~half of 8-bit AdamW
    assert opt_lion < opt_8bit * 0.60


def test_activation_memory_gradient_checkpointing(llama3_8b_spec):
    calc = VRAMCalculator()

    t_ckpt = TrainingConfig(
        per_device_train_batch_size=1,
        max_seq_length=2048,
        gradient_checkpointing=True,
        attention_impl=AttentionImplementation.FLASH_ATTENTION_2,
    )
    act_ckpt = calc.calculate_activation_memory(llama3_8b_spec, t_ckpt)

    t_no_ckpt = TrainingConfig(
        per_device_train_batch_size=1,
        max_seq_length=2048,
        gradient_checkpointing=False,
        attention_impl=AttentionImplementation.FLASH_ATTENTION_2,
    )
    act_no_ckpt = calc.calculate_activation_memory(llama3_8b_spec, t_no_ckpt)

    # Gradient checkpointing should reduce activation memory substantially
    assert act_ckpt < act_no_ckpt * 0.30


def test_flash_attention_vs_sdpa_activations(llama3_8b_spec):
    calc = VRAMCalculator()

    t_flash = TrainingConfig(
        per_device_train_batch_size=1,
        max_seq_length=4096,
        gradient_checkpointing=True,
        attention_impl=AttentionImplementation.FLASH_ATTENTION_2,
    )
    act_flash = calc.calculate_activation_memory(llama3_8b_spec, t_flash)

    t_sdpa = TrainingConfig(
        per_device_train_batch_size=1,
        max_seq_length=4096,
        gradient_checkpointing=True,
        attention_impl=AttentionImplementation.SDPA,
    )
    act_sdpa = calc.calculate_activation_memory(llama3_8b_spec, t_sdpa)

    # At 4096 tokens, FlashAttention eliminates O(s^2) attention matrix
    assert act_flash < act_sdpa


def test_full_estimate_feasibility_rtx_3060(
    llama3_8b_spec,
    rtx_3060_spec,
    default_lora_config,
    qlora_nf4_config,
    standard_training_config,
):
    calc = VRAMCalculator()
    report = calc.estimate(
        llama3_8b_spec,
        default_lora_config,
        qlora_nf4_config,
        standard_training_config,
        rtx_3060_spec,
    )

    # Llama 3.1 8B QLoRA with rank 16 on RTX 3060 12GB should fit comfortably (< 11,500 MB)
    assert report.status in ("OPTIMAL", "FEASIBLE")
    assert report.peak_vram_mb < rtx_3060_spec.total_vram_mb
    assert report.headroom_mb > 500.0


def test_oom_detection(
    llama3_8b_spec,
    rtx_3060_spec,
    default_lora_config,
    standard_training_config,
):
    calc = VRAMCalculator()
    # Unquantized 16-bit base model requires ~15GB weights alone on a 12GB GPU
    unquant_cfg = QuantizationConfig(quant_type=QuantizationType.NONE)
    report = calc.estimate(
        llama3_8b_spec,
        default_lora_config,
        unquant_cfg,
        standard_training_config,
        rtx_3060_spec,
    )

    assert report.status == "OOM"
    assert report.peak_vram_mb > rtx_3060_spec.total_vram_mb
    assert any("exceeds total physical VRAM" in r for r in report.recommendations)


def test_hyperparameter_optimizer_autotune(llama3_8b_spec, rtx_3060_spec):
    opt = HyperparameterOptimizer()
    rec = opt.auto_tune_for_gpu(
        llama3_8b_spec,
        rtx_3060_spec,
        target_effective_batch_size=16,
        target_seq_len=2048,
    )

    assert rec.is_safe is True
    assert rec.optimal_micro_batch_size >= 1
    assert rec.recommended_quantization == QuantizationType.NF4
    assert rec.headroom_mb > 0

def test_multi_gpu_sharding_zero3(llama3_8b_spec, rtx_3060_spec, default_lora_config, qlora_nf4_config):
    from loraforge.core.config import DistributedStrategy

    calc = VRAMCalculator()

    # 1. Single GPU baseline
    t_single = TrainingConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_gpus=1,
        distributed_strategy=DistributedStrategy.NONE,
    )
    rep_single = calc.estimate(llama3_8b_spec, default_lora_config, qlora_nf4_config, t_single, rtx_3060_spec)

    # 2. Dual GPU ZeRO-3
    t_zero3 = TrainingConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_gpus=2,
        distributed_strategy=DistributedStrategy.ZERO_3,
    )
    rep_zero3 = calc.estimate(llama3_8b_spec, default_lora_config, qlora_nf4_config, t_zero3, rtx_3060_spec)

    # In ZeRO-3, base model weights per GPU should be roughly cut in half!
    assert rep_zero3.breakdown.base_model_weights_mb < rep_single.breakdown.base_model_weights_mb * 0.60
    assert rep_zero3.breakdown.optimizer_states_mb < rep_single.breakdown.optimizer_states_mb * 0.60
    assert rep_zero3.breakdown.gradients_mb < rep_single.breakdown.gradients_mb * 0.60
    assert rep_zero3.effective_batch_size == 16  # 1 * 8 * 2
    assert rep_zero3.total_cluster_vram_mb == 24576


def test_multi_gpu_ddp_comm_buffers(llama3_8b_spec, rtx_3060_spec, default_lora_config, qlora_nf4_config):
    from loraforge.core.config import DistributedStrategy

    calc = VRAMCalculator()
    t_ddp = TrainingConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_gpus=4,
        distributed_strategy=DistributedStrategy.DDP,
    )
    rep_ddp = calc.estimate(llama3_8b_spec, default_lora_config, qlora_nf4_config, t_ddp, rtx_3060_spec)

    assert rep_ddp.breakdown.communication_buffers_mb > 0
    assert rep_ddp.num_gpus == 4
    assert rep_ddp.effective_batch_size == 32
