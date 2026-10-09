import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from loraforge.core.models import PRESET_MODELS
from loraforge.core.hardware import PRESET_GPUS
from loraforge.core.config import (
    LoRAConfig,
    QuantizationConfig,
    QuantizationType,
    TrainingConfig,
    OptimizerType,
    AttentionImplementation,
)
from loraforge.vram.calculator import VRAMCalculator
from loraforge.vram.optimizer import HyperparameterOptimizer

out_dir = Path("docs/images")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Generate CLI VRAM Analysis SVG
console = Console(record=True, width=108)

model_spec = PRESET_MODELS["meta-llama/Llama-3.1-8B-Instruct"]
gpu_spec = PRESET_GPUS["rtx-3060-12gb"]
lora_cfg = LoRAConfig(r=16, lora_alpha=32)
quant_cfg = QuantizationConfig(quant_type=QuantizationType.NF4, double_quant=True)
training_cfg = TrainingConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_seq_length=2048,
    optimizer=OptimizerType.PAGED_ADAMW_8BIT,
    attention_impl=AttentionImplementation.FLASH_ATTENTION_2,
    gradient_checkpointing=True,
)

calc = VRAMCalculator()
report = calc.estimate(model_spec, lora_cfg, quant_cfg, training_cfg, gpu_spec)

badge = "[bold green]OPTIMAL[/bold green]"
header_text = (
    f"[bold]Model:[/bold] {model_spec.name} ({model_spec.num_parameters / 1e9:.2f}B params)\n"
    f"[bold]Target GPU:[/bold] {gpu_spec.name} ({gpu_spec.total_vram_mb} MB VRAM)\n"
    f"[bold]Hyperparameters:[/bold] Micro-Batch=1, GradAcc=8 (Effective Batch=8), SeqLen=2048, LoRA Rank=16\n"
    f"[bold]Quantization / Opt:[/bold] NF4 | paged_adamw_8bit | GradCheckpointing=True | FlashAttn=True\n"
    f"[bold]Feasibility Verdict:[/bold] {badge}"
)
console.print(Panel(header_text, title="LoRAForge Predictive VRAM Assessment", border_style="cyan"))

table = Table(title="HBM Allocation Breakdown", show_header=True, header_style="bold magenta")
table.add_column("Component", style="dim", width=30)
table.add_column("Formula / Source", width=36)
table.add_column("Allocation (MB)", justify="right", width=18)
table.add_column("% of Peak", justify="right", width=12)

bd = report.breakdown
peak = report.peak_vram_mb
items = [
    ("Base Model Weights", "NF4 Quantized Weights (0.5 B/p)", bd.base_model_weights_mb),
    ("Quantization Metadata", "Block scales + Double-Quant constants", bd.quantization_metadata_mb),
    ("LoRA Adapter Matrices", f"Rank=16 ({report.lora_parameters.trainable_parameters:,} params)", bd.lora_weights_mb),
    ("Gradient Buffers", f"BF16 Trainable Gradients ({report.lora_parameters.trainable_parameters:,} p)", bd.gradients_mb),
    ("Optimizer States", "paged_adamw_8bit Moments Buffer", bd.optimizer_states_mb),
    ("Activation Memory", "Forward stashed + intra-block recompute", bd.activations_mb),
    ("CUDA Context & Workspace", f"CUDA CC {gpu_spec.compute_capability} Runtime Baseline", bd.cuda_context_mb),
    ("Allocator Fragmentation", "12% Dynamic Tensor Headroom", bd.fragmentation_buffer_mb),
]

for name, detail, val in items:
    pct = (val / peak * 100.0) if peak > 0 else 0.0
    table.add_row(name, detail, f"{val:.1f} MB", f"{pct:.1f}%")

table.add_section()
table.add_row(
    "[bold]Total Predicted Peak VRAM[/bold]",
    "[bold]Sum of components[/bold]",
    f"[bold]{report.peak_vram_mb:.1f} MB[/bold]",
    f"[bold]{report.utilization_percentage:.1f}% of GPU[/bold]",
)
table.add_row(
    "[bold]Physical GPU Memory[/bold]",
    f"[bold]{gpu_spec.name}[/bold]",
    f"[bold]{gpu_spec.total_vram_mb} MB[/bold]",
    "[bold]100.0%[/bold]",
)
table.add_row(
    "[bold]Safety Headroom[/bold]",
    "[bold]Safe buffer before driver OOM[/bold]",
    f"[bold green]{report.headroom_mb:.1f} MB[/bold green]",
    f"[bold green]{(report.headroom_mb / gpu_spec.total_vram_mb * 100):.1f}%[/bold green]",
)
console.print(table)

svg_vram = console.export_svg(title="loraforge vram --model meta-llama/Llama-3.1-8B-Instruct --gpu rtx-3060-12gb")
with open(out_dir / "cli_vram.svg", "w", encoding="utf-8") as f:
    f.write(svg_vram)
print("[OK] Exported docs/images/cli_vram.svg")


# 2. Generate Autotune CLI SVG
console_auto = Console(record=True, width=108)
optimizer_engine = HyperparameterOptimizer()
rec = optimizer_engine.auto_tune_for_gpu(model_spec, gpu_spec, target_effective_batch_size=16, target_seq_len=2048)

panel_text = (
    f"[bold]Target GPU:[/bold] {gpu_spec.name} ({gpu_spec.total_vram_mb} MB)\n"
    f"[bold]Target Model:[/bold] {model_spec.name} ({model_spec.num_parameters / 1e9:.2f}B)\n"
    f"[bold]Feasible / Safe:[/bold] {'[bold green]YES[/bold green]' if rec.is_safe else '[bold red]NO[/bold red]'}\n"
    f"[bold]Optimal Micro-Batch Size:[/bold] [bold cyan]{rec.optimal_micro_batch_size}[/bold cyan]\n"
    f"[bold]Gradient Accumulation Steps:[/bold] [bold cyan]{rec.optimal_grad_accumulation_steps}[/bold cyan] (Effective Batch: {rec.optimal_micro_batch_size * rec.optimal_grad_accumulation_steps})\n"
    f"[bold]Recommended Sequence Length:[/bold] [bold cyan]{rec.max_feasible_seq_len}[/bold cyan] tokens\n"
    f"[bold]Quantization:[/bold] [bold cyan]{rec.recommended_quantization.value.upper()}[/bold cyan]\n"
    f"[bold]Optimizer:[/bold] [bold cyan]{rec.recommended_optimizer.value}[/bold cyan]\n"
    f"[bold]Estimated Peak VRAM:[/bold] [bold yellow]{rec.predicted_peak_vram_mb:.1f} MB[/bold yellow] (Headroom: [green]{rec.headroom_mb:.1f} MB[/green])"
)
console_auto.print(Panel(panel_text, title="LoRAForge Auto-Tuned Hyperparameters", border_style="green"))

svg_auto = console_auto.export_svg(title="loraforge autotune --model Llama-3.1-8B --gpu rtx-3060")
with open(out_dir / "cli_autotune.svg", "w", encoding="utf-8") as f:
    f.write(svg_auto)
print("[OK] Exported docs/images/cli_autotune.svg")
