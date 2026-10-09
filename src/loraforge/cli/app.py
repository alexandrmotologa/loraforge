"""Rich command-line interface for LoRAForge using Typer.

Provides commands for predictive VRAM profiling, hardware inspection,
model architecture exploration, dataset packing audits, training supervision,
and starting the web control plane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from loraforge.core.config import (
    AttentionImplementation,
    DatasetConfig,
    DatasetFormat,
    ExportConfig,
    FineTuningJobConfig,
    GPUSpec,
    LoRAConfig,
    ModelArchitectureSpec,
    OptimizerType,
    QuantizationConfig,
    QuantizationType,
    TrainingConfig,
)
from loraforge.core.hardware import (
    PRESET_GPUS,
    detect_system_gpus,
    get_current_telemetry,
    get_gpu_preset,
)
from loraforge.core.models import PRESET_MODELS, get_model_spec
from loraforge.data.formats import DatasetParser
from loraforge.data.validator import DatasetValidator
from loraforge.export.inferops import InferOpsExporter
from loraforge.supervisor.runner import JobStatus, TrainingSupervisor
from loraforge.vram.calculator import VRAMCalculator
from loraforge.vram.optimizer import HyperparameterOptimizer

app = typer.Typer(
    name="loraforge",
    help="Bare-metal local fine-tuning control plane and QLoRA supervisor for consumer & workstation GPUs.",
    add_completion=False,
)
data_app = typer.Typer(help="Dataset inspection, validation, and sequence packing commands.")
app.add_typer(data_app, name="data")

console = Console(width=160)


@app.command("vram")
@app.command("estimate")
def estimate_vram(
    model: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct",
        "--model",
        "-m",
        help="Base model architecture name or local path.",
    ),
    gpu: str = typer.Option(
        "rtx-3060-12gb",
        "--gpu",
        "-g",
        help="Target GPU preset (e.g. rtx-3060-12gb, rtx-4070-12gb, rtx-3090-24gb, rtx-4090-24gb).",
    ),
    seq_len: int = typer.Option(
        2048,
        "--seq-len",
        "-s",
        help="Maximum training sequence length.",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Per-device micro-batch size.",
    ),
    grad_acc: int = typer.Option(
        8,
        "--grad-acc",
        "-a",
        help="Gradient accumulation steps.",
    ),
    rank: int = typer.Option(
        16,
        "--rank",
        "-r",
        help="LoRA decomposition rank dimension.",
    ),
    quant: str = typer.Option(
        "nf4",
        "--quant",
        "-q",
        help="Base model quantization strategy ('nf4', 'fp4', 'int8', 'none').",
    ),
    optimizer: str = typer.Option(
        "paged_adamw_8bit",
        "--optimizer",
        "-o",
        help="Optimizer type ('paged_adamw_8bit', 'adamw_8bit', 'adamw_torch', 'lion_8bit').",
    ),
    grad_checkpointing: bool = typer.Option(
        True,
        "--grad-ckpt/--no-grad-ckpt",
        help="Enable activation gradient checkpointing.",
    ),
    flash_attn: bool = typer.Option(
        True,
        "--flash-attn/--no-flash-attn",
        help="Enable FlashAttention-2 memory optimization.",
    ),
) -> None:
    """Predict exact training VRAM consumption and evaluate GPU feasibility."""
    model_spec = get_model_spec(model)
    if not model_spec:
        console.print(f"[bold red]Error:[/bold red] Unknown model '{model}'. Use 'loraforge models' to list available presets.")
        raise typer.Exit(code=1)

    gpu_spec = get_gpu_preset(gpu)
    if not gpu_spec:
        console.print(f"[bold red]Error:[/bold red] Unknown GPU target '{gpu}'. Use 'loraforge hardware' to inspect presets.")
        raise typer.Exit(code=1)

    lora_cfg = LoRAConfig(r=rank, lora_alpha=rank * 2)
    quant_cfg = QuantizationConfig(
        quant_type=QuantizationType(quant),
        double_quant=True,
    )
    training_cfg = TrainingConfig(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        max_seq_length=seq_len,
        optimizer=OptimizerType(optimizer),
        attention_impl=AttentionImplementation.FLASH_ATTENTION_2 if flash_attn else AttentionImplementation.SDPA,
        gradient_checkpointing=grad_checkpointing,
    )

    calc = VRAMCalculator()
    report = calc.estimate(model_spec, lora_cfg, quant_cfg, training_cfg, gpu_spec)

    # Render summary header
    status_colors = {
        "OPTIMAL": "bold green",
        "FEASIBLE": "bold cyan",
        "HIGH_RISK": "bold yellow",
        "OOM": "bold white on red",
    }
    badge = f"[{status_colors.get(report.status, 'bold white')}]{report.status}[/{status_colors.get(report.status, 'bold white')}]"

    header_text = (
        f"[bold]Model:[/bold] {model_spec.name} ({model_spec.num_parameters / 1e9:.2f}B params)\n"
        f"[bold]Target GPU:[/bold] {gpu_spec.name} ({gpu_spec.total_vram_mb} MB)\n"
        f"[bold]Hyperparameters:[/bold] Micro-Batch={batch_size}, GradAcc={grad_acc} (Effective Batch={batch_size * grad_acc}), SeqLen={seq_len}, LoRA Rank={rank}\n"
        f"[bold]Quantization / Opt:[/bold] {quant.upper()} | {optimizer} | GradCheckpointing={grad_checkpointing} | FlashAttn={flash_attn}\n"
        f"[bold]Feasibility Verdict:[/bold] {badge}"
    )
    console.print(Panel(header_text, title="LoRAForge Predictive VRAM Assessment", border_style="cyan"))

    # Table of VRAM breakdown
    table = Table(title="HBM Allocation Breakdown", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="dim", width=32)
    table.add_column("Formula / Source", width=32)
    table.add_column("Allocation (MB)", justify="right", width=18)
    table.add_column("% of Peak", justify="right", width=12)

    bd = report.breakdown
    peak = report.peak_vram_mb

    items = [
        ("Base Model Weights", f"{quant.upper()} Quantized Weights", bd.base_model_weights_mb),
        ("Quantization Metadata", "Block scales + Double-Quant constants", bd.quantization_metadata_mb),
        ("LoRA Adapter Matrices", f"Rank={rank} ({report.lora_parameters.trainable_parameters:,} params)", bd.lora_weights_mb),
        ("Gradient Buffers", f"FP16/BF16 Trainable Gradients ({report.lora_parameters.trainable_parameters:,} p)", bd.gradients_mb),
        ("Optimizer States", f"{optimizer} Moments Buffer", bd.optimizer_states_mb),
        ("Activation Memory", f"Checkpointing={grad_checkpointing}, FlashAttn={flash_attn}", bd.activations_mb),
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
        f"Available: {gpu_spec.usable_vram_mb:.1f} MB safe",
        f"[bold]{gpu_spec.total_vram_mb:.1f} MB[/bold]",
        "100.0%",
    )
    table.add_row(
        "[bold]Free Headroom[/bold]",
        "Remaining margin before hardware OOM",
        f"[bold {'green' if report.headroom_mb > 0 else 'red'}]{report.headroom_mb:.1f} MB[/bold {'green' if report.headroom_mb > 0 else 'red'}]",
        f"{(report.headroom_mb / gpu_spec.total_vram_mb * 100.0):.1f}%",
    )

    console.print(table)

    # Print recommendations
    if report.recommendations:
        console.print("\n[bold yellow]Advisor Recommendations:[/bold yellow]")
        for rec in report.recommendations:
            console.print(f"  • {rec}")
    console.print()


@app.command("presets")
def list_presets() -> None:
    """List available turnkey task presets for specialized fine-tuning."""
    from loraforge.core.presets import PRESETS

    table = Table(title="Turnkey Fine-Tuning Task Presets", show_header=True)
    table.add_column("Preset", style="bold cyan")
    table.add_column("Default Model", style="dim")
    table.add_column("Rank / Alpha", justify="center")
    table.add_column("SeqLen", justify="right")
    table.add_column("Description")

    for k, p in PRESETS.items():
        table.add_row(
            k,
            p.recommended_model.split("/")[-1],
            f"r={p.lora.r} / α={p.lora.lora_alpha}",
            f"{p.training.max_seq_length} toks",
            p.description,
        )

    console.print(table)


@app.command("init")
def init_project(
    preset: str = typer.Option("chat", "--preset", "-p", help="Task preset: reasoning, coding, agentic, chat."),
    gpu: str = typer.Option("rtx-3060-12gb", "--gpu", "-g", help="Target GPU preset."),
    dataset: str = typer.Option("train.jsonl", "--dataset", "-d", help="Path to training dataset."),
    out: str = typer.Option("loraforge.yaml", "--out", "-o", help="Output configuration file."),
) -> None:
    """Initialize a reproducible fine-tuning project configuration from a preset."""
    from loraforge.core.presets import build_config_from_preset

    cfg = build_config_from_preset(preset_name=preset, gpu_name=gpu, dataset_path=dataset)
    target = Path(out)
    cfg.save_yaml(target)

    console.print(Panel(
        f"[bold green]Created fine-tuning configuration:[/bold green] [underline]{target.resolve()}[/underline]\n\n"
        f"[bold]Task Preset:[/bold] {preset}\n"
        f"[bold]Base Model:[/bold] {cfg.model.name}\n"
        f"[bold]Target GPU:[/bold] {cfg.gpu.name} ({cfg.gpu.total_vram_mb} MB)\n"
        f"[bold]LoRA Rank / Alpha:[/bold] r={cfg.lora.r}, alpha={cfg.lora.lora_alpha}\n"
        f"[bold]Max Sequence Length:[/bold] {cfg.training.max_seq_length} tokens\n\n"
        f"[dim]Next step: review VRAM budget with 'loraforge vram' or run 'loraforge train'.[/dim]",
        title="Project Initialization Wizard",
        border_style="green",
    ))


@app.command("autotune")
def autotune_config(
    model: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct", "--model", "-m"),
    gpu: str = typer.Option("rtx-3060-12gb", "--gpu", "-g"),
    effective_batch: int = typer.Option(16, "--effective-batch", "-e"),
    target_seq: int = typer.Option(2048, "--seq-len", "-s"),
) -> None:
    """Find the optimal safe hyperparameter configuration for a model on a target GPU."""
    model_spec = get_model_spec(model)
    if not model_spec:
        console.print(f"[bold red]Error:[/bold red] Unknown model '{model}'.")
        raise typer.Exit(code=1)

    gpu_spec = get_gpu_preset(gpu)
    if not gpu_spec:
        console.print(f"[bold red]Error:[/bold red] Unknown GPU preset '{gpu}'.")
        raise typer.Exit(code=1)

    optimizer = HyperparameterOptimizer()
    rec = optimizer.auto_tune_for_gpu(
        model_spec,
        gpu_spec,
        target_effective_batch_size=effective_batch,
        target_seq_len=target_seq,
    )

    table = Table(title=f"Auto-Tuned Training Configuration for {gpu_spec.name}", show_header=True)
    table.add_column("Parameter", style="dim")
    table.add_column("Recommended Value", style="bold cyan")

    table.add_row("Micro Batch Size", str(rec.optimal_micro_batch_size))
    table.add_row("Gradient Accumulation Steps", str(rec.optimal_grad_accumulation_steps))
    table.add_row("Effective Batch Size", str(rec.optimal_micro_batch_size * rec.optimal_grad_accumulation_steps))
    table.add_row("Target Sequence Length", f"{target_seq} tokens")
    table.add_row("Maximum Feasible Context", f"{rec.max_feasible_seq_len} tokens")
    table.add_row("LoRA Rank (r)", str(rec.recommended_rank))
    table.add_row("Quantization Type", rec.recommended_quantization.value.upper())
    table.add_row("Optimizer", rec.recommended_optimizer.value)
    table.add_row("Predicted Peak VRAM", f"{rec.predicted_peak_vram_mb:.1f} MB")
    table.add_row("Safe Headroom", f"{rec.headroom_mb:.1f} MB")

    console.print(table)
    if rec.notes:
        for n in rec.notes:
            console.print(f"  • {n}")


@app.command("hardware")
def hardware_info() -> None:
    """Inspect local GPU devices via NVML/SMI and list supported reference presets."""
    snaps = get_current_telemetry()
    if snaps:
        table = Table(title="Detected Physical Accelerators", show_header=True)
        table.add_column("Index", justify="center")
        table.add_column("Device Name", style="bold")
        table.add_column("Total VRAM", justify="right")
        table.add_column("Used VRAM", justify="right")
        table.add_column("Free VRAM", justify="right")
        table.add_column("GPU Util", justify="right")
        table.add_column("Temp", justify="right")
        table.add_column("Power", justify="right")

        for s in snaps:
            table.add_row(
                str(s.index),
                s.name,
                f"{s.vram_total_mb:.1f} MB",
                f"{s.vram_used_mb:.1f} MB",
                f"{s.vram_free_mb:.1f} MB",
                f"{s.gpu_utilization_percent:.1f}%",
                f"{s.temperature_c:.0f}°C" if s.temperature_c is not None else "N/A",
                f"{s.power_draw_w:.1f}W" if s.power_draw_w is not None else "N/A",
            )
        console.print(table)
    else:
        console.print("[dim]No local NVIDIA accelerator active. Running in workstation simulation mode.[/dim]\n")

    # Reference presets table
    preset_table = Table(title="Reference Consumer & Workstation GPU Presets", show_header=True)
    preset_table.add_column("Preset Key", style="cyan")
    preset_table.add_column("GPU Model", style="bold")
    preset_table.add_column("VRAM Capacity", justify="right")
    preset_table.add_column("Compute Arch", justify="center")
    preset_table.add_column("BF16", justify="center")
    preset_table.add_column("FlashAttn-2", justify="center")

    for k, spec in PRESET_GPUS.items():
        preset_table.add_row(
            k,
            spec.name,
            f"{spec.total_vram_mb // 1024} GB",
            f"CC {spec.compute_capability}",
            "[green]Yes[/green]" if spec.supports_bf16 else "[red]No[/red]",
            "[green]Yes[/green]" if spec.supports_flash_attn_2 else "[red]No[/red]",
        )

    console.print(preset_table)


@app.command("models")
def list_models() -> None:
    """List built-in foundation model architecture specifications."""
    table = Table(title="Built-in Foundation Model Architecture Specifications", show_header=True)
    table.add_column("Model Identifier", style="bold cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Layers (L)", justify="center")
    table.add_column("Hidden (d)", justify="right")
    table.add_column("MLP (d_mlp)", justify="right")
    table.add_column("Heads (q / kv)", justify="center")
    table.add_column("Vocab", justify="right")

    for name, m in PRESET_MODELS.items():
        table.add_row(
            name,
            f"{m.num_parameters / 1e9:.2f}B",
            str(m.num_layers),
            str(m.hidden_size),
            str(m.intermediate_size),
            f"{m.num_attention_heads} / {m.num_key_value_heads}",
            f"{m.vocab_size:,}",
        )

    console.print(table)


@data_app.command("audit")
def audit_dataset(
    path: str = typer.Argument(..., help="Path to JSON or JSONL dataset file."),
    fmt: str = typer.Option("sharegpt", "--format", "-f", help="Format: sharegpt, alpaca, chatml, openai_messages."),
    seq_len: int = typer.Option(2048, "--seq-len", "-s", help="Target sequence length."),
) -> None:
    """Audit token length distribution and sequence packing feasibility of a dataset."""
    file_path = Path(path)
    if not file_path.is_file():
        console.print(f"[bold red]Error:[/bold red] File not found: {path}")
        raise typer.Exit(code=1)

    samples = DatasetParser.load_jsonl(file_path, DatasetFormat(fmt))
    validator = DatasetValidator()
    report = validator.audit(samples, target_seq_len=seq_len)

    dist = report.distribution
    console.print(Panel(
        f"[bold]Total Samples:[/bold] {dist.total_samples:,}\n"
        f"[bold]Total Estimated Tokens:[/bold] {dist.total_tokens:,}\n"
        f"[bold]Min / Mean / Median Length:[/bold] {dist.min_length} / {dist.mean_length:.1f} / {dist.median_length:.1f} tokens\n"
        f"[bold]P90 / P95 / P99 Length:[/bold] {dist.p90_length:.0f} / {dist.p95_length:.0f} / {dist.p99_length:.0f} tokens\n"
        f"[bold]Max Sample Length:[/bold] {dist.max_length:,} tokens\n"
        f"[bold]Truncated Samples (> {seq_len} toks):[/bold] {dist.truncated_samples} ({dist.truncated_percent:.1f}%)\n"
        f"[bold]Recommended Sequence Length:[/bold] {report.recommended_max_seq_len} tokens\n"
        f"[bold]Estimated Training Steps (3 epochs, batch 8):[/bold] {report.estimated_total_steps:,} steps",
        title=f"Dataset Audit: {file_path.name}",
        border_style="cyan",
    ))

    if report.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for w in report.warnings:
            console.print(f"  • {w}")


@data_app.command("clean")
def clean_dataset(
    path: str = typer.Argument(..., help="Path to input dataset file."),
    output: str = typer.Option(..., "--output", "-o", help="Path to write sanitized JSONL dataset."),
    fmt: str = typer.Option("sharegpt", "--format", "-f", help="Format: sharegpt, alpaca, chatml, openai_messages."),
) -> None:
    """Sanitize dataset: remove duplicates, strip invisible control characters, and prune empty turns."""
    from loraforge.data.cleaner import DatasetCleaner

    in_file = Path(path)
    if not in_file.is_file():
        console.print(f"[bold red]Error:[/bold red] File not found: {path}")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]Sanitizing dataset:[/bold cyan] {in_file.name}...")
    report = DatasetCleaner.clean(in_file, output, fmt=DatasetFormat(fmt))

    table = Table(title="Dataset Sanitization Summary", show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Count / Value", style="bold cyan")

    table.add_row("Total Input Samples", f"{report.total_input_samples:,}")
    table.add_row("Retained High-Quality Samples", f"{report.retained_samples:,} ({report.retention_rate_pct:.1f}%)")
    table.add_row("Duplicate Samples Dropped", f"{report.duplicate_samples_dropped:,}")
    table.add_row("Degenerate / Single-Turn Dropped", f"{report.degenerate_samples_dropped:,}")
    table.add_row("Whitespace-Trimmed Turns", f"{report.whitespace_trimmed_turns:,}")
    table.add_row("Sanitized Output File", report.output_path)

    console.print(table)


@data_app.command("preview")
def preview_dataset_template(
    path: str = typer.Argument(..., help="Path to input dataset file."),
    template: str = typer.Option("llama3", "--template", "-t", help="Template: llama3, chatml, mistral, deepseek, gemma2, phi3."),
    samples: int = typer.Option(1, "--samples", "-n", help="Number of samples to preview."),
    fmt: str = typer.Option("sharegpt", "--format", "-f", help="Input format."),
) -> None:
    """Preview formatted prompt using a foundation model chat template."""
    from loraforge.data.templates import get_template

    in_file = Path(path)
    if not in_file.is_file():
        console.print(f"[bold red]Error:[/bold red] File not found: {path}")
        raise typer.Exit(code=1)

    tmpl = get_template(template)
    if not tmpl:
        console.print(f"[bold red]Error:[/bold red] Unknown template '{template}'.")
        raise typer.Exit(code=1)

    loaded = DatasetParser.load_jsonl(in_file, DatasetFormat(fmt), max_samples=samples)
    for idx, sample in enumerate(loaded, 1):
        formatted = tmpl.format(sample)
        console.print(Panel(formatted, title=f"Sample #{idx} [{tmpl.name.upper()}]", border_style="cyan"))


@app.command("train")
def train_job(
    model: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct", "--model", "-m"),
    gpu: str = typer.Option("rtx-3060-12gb", "--gpu", "-g"),
    dataset: str = typer.Option("data/train.jsonl", "--dataset", "-d", help="Path to JSONL dataset file."),
    fmt: str = typer.Option("sharegpt", "--format", "-f", help="Format: sharegpt, alpaca, chatml, openai_messages."),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to loraforge.yaml config file."),
    steps: int = typer.Option(25, "--steps", help="Training steps."),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Run simulated supervisor loop instead of bare-metal subprocess."),
) -> None:
    """Launch a supervised fine-tuning run with real-time telemetry."""
    if config_file:
        cfg_path = Path(config_file)
        if not cfg_path.is_file():
            console.print(f"[bold red]Error:[/bold red] Configuration file not found: {config_file}")
            raise typer.Exit(code=1)
        cfg = FineTuningJobConfig.from_yaml(cfg_path.read_text(encoding="utf-8"))
    else:
        model_spec = get_model_spec(model)
        gpu_spec = get_gpu_preset(gpu)

        if not model_spec or not gpu_spec:
            console.print("[bold red]Invalid model or GPU specified.[/bold red]")
            raise typer.Exit(code=1)

        cfg = FineTuningJobConfig(
            job_name=f"ft-{model_spec.name.split('/')[-1].lower()}",
            model=model_spec,
            gpu=gpu_spec,
            dataset=DatasetConfig(path=dataset, format=DatasetFormat(fmt)),
        )

    supervisor = TrainingSupervisor()
    console.print(f"[bold cyan]Launching fine-tuning supervisor for {cfg.job_name}...[/bold cyan]")

    def print_metric(m):
        console.print(f"  [Step {m.step:03d}/{steps}] loss: [bold yellow]{m.loss:.4f}[/bold yellow] | lr: {m.learning_rate:.2e} | vram: [cyan]{m.vram_used_mb:.1f} MB[/cyan]")

    supervisor.event_bus.subscribe(print_metric)

    if dry_run:
        console.print("[dim]Mode: Dry-run simulation enabled.[/dim]")
        state = supervisor.run_dry_run_simulation(cfg, total_steps=steps, step_delay=0.03)
    else:
        console.print("[bold green]Mode: Bare-metal PyTorch subprocess execution.[/bold green]")
        state = supervisor.run_training_job(cfg, max_steps=steps)

    if state.status == JobStatus.COMPLETED:
        console.print(f"[bold green]Training COMPLETED![/bold green] Peak VRAM: {state.peak_vram_recorded_mb:.1f} MB")
        console.print(f"Adapter saved to: [underline]{state.output_checkpoint_dir}[/underline]")
    elif state.status == JobStatus.OOM_ABORTED:
        console.print(f"[bold red]Training ABORTED (OOM Watchdog)![/bold red] {state.error_message}")
        raise typer.Exit(code=2)
    elif state.status == JobStatus.FAILED:
        console.print(f"[bold red]Training FAILED![/bold red] {state.error_message}")
        raise typer.Exit(code=1)


@app.command("export")
def export_job(
    model: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct", "--model", "-m"),
    gpu: str = typer.Option("rtx-3060-12gb", "--gpu", "-g"),
    output_dir: str = typer.Option("./exports/llama3-loraforge", "--out", "-o"),
    engine: str = typer.Option("vllm", "--engine", "-e", help="InferOps engine: vllm, sglang, or llama-cpp."),
    deploy: bool = typer.Option(False, "--deploy", "-d", help="Directly dispatch model registration to running InferOps server."),
    inferops_url: str = typer.Option("http://127.0.0.1:8000", "--inferops-url", help="InferOps server base URL."),
    job_name: Optional[str] = typer.Option(None, "--job-name", "-j", help="Custom deployment job name."),
) -> None:
    """Generate InferOps deployment manifest and adapter merge blueprint."""
    model_spec = get_model_spec(model)
    gpu_spec = get_gpu_preset(gpu)
    if not model_spec or not gpu_spec:
        console.print("[bold red]Invalid model or GPU specified.[/bold red]")
        raise typer.Exit(code=1)

    effective_name = job_name or f"deploy-{model_spec.name.split('/')[-1].lower()}"
    cfg = FineTuningJobConfig(
        job_name=effective_name,
        model=model_spec,
        gpu=gpu_spec,
        dataset=DatasetConfig(path="train.jsonl", format=DatasetFormat.SHAREGPT),
    )

    manifest_path = InferOpsExporter.export_bundle(cfg, output_dir, engine=engine)

    deploy_status_msg = ""
    if deploy:
        result = InferOpsExporter.deploy_to_server(manifest_path, inferops_url=inferops_url)
        if result.get("status") == "deployed_live":
            deploy_status_msg = f"\n[bold green]✓ Live Dispatched to InferOps:[/bold green] registered on {inferops_url}"
        else:
            deploy_status_msg = f"\n[bold yellow]! InferOps server not active at {inferops_url}[/bold yellow] (ready for launch via script)"

    console.print(Panel(
        f"[bold green]InferOps Service Manifest Exported Successfully![/bold green]\n\n"
        f"[bold]Manifest Path:[/bold] {manifest_path.resolve()}\n"
        f"[bold]Target Engine:[/bold] {engine}{deploy_status_msg}\n\n"
        f"[bold]To launch in InferOps:[/bold]\n"
        f"[cyan]inferops serve --manifest {manifest_path} --engine {engine} --port 8000[/cyan]",
        title="InferOps Integration Bridge",
        border_style="green",
    ))


@app.command("cloud")
def export_cloud(
    model: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct", "--model", "-m"),
    gpu: str = typer.Option("rtx-4090-24gb", "--gpu", "-g"),
    output_dir: str = typer.Option("./cloud_deploy", "--out", "-o"),
    provider: str = typer.Option("runpod", "--provider", "-p", help="Target cloud provider: runpod, vastai."),
) -> None:
    """Generate turnkey Dockerfile and bootstrap entrypoint for RunPod / Vast.ai GPUs."""
    from loraforge.export.cloud import CloudDeploymentGenerator

    model_spec = get_model_spec(model)
    gpu_spec = get_gpu_preset(gpu)

    cfg = FineTuningJobConfig(
        job_name=f"cloud-{model_spec.name.split('/')[-1].lower()}",
        model=model_spec,
        gpu=gpu_spec,
        dataset=DatasetConfig(path="dataset.jsonl", format=DatasetFormat.SHAREGPT),
    )

    pkg = CloudDeploymentGenerator.generate(cfg, output_dir, provider=provider)
    console.print(Panel(
        f"[bold green]Cloud GPU Deployment Package Created![/bold green]\n\n"
        f"• [bold]Target Cloud Provider:[/bold] {pkg.provider.upper()}\n"
        f"• [bold]Dockerfile:[/bold] {pkg.dockerfile_path}\n"
        f"• [bold]Bootstrap Entrypoint:[/bold] {pkg.entrypoint_script_path}\n"
        f"• [bold]Training Config:[/bold] {pkg.config_yaml_path}\n\n"
        f"[dim]Build & launch container: 'docker build -t loraforge-runner {output_dir}'[/dim]",
        title="Cloud Cluster Exporter",
        border_style="cyan",
    ))


@app.command("merge")
def merge_adapters(
    adapters: str = typer.Option(..., "--adapters", "-a", help="Comma-separated paths to adapter folders or files."),
    output_dir: str = typer.Option("./merged_adapters", "--out", "-o", help="Directory where merged adapter will be saved."),
    method: str = typer.Option("ties", "--method", "-m", help="Merge method: linear, ties, dare, svd."),
    density: float = typer.Option(0.2, "--density", help="Top-k density fraction kept for TIES trimming (0.01 - 1.0)."),
    drop_rate: float = typer.Option(0.5, "--drop-rate", help="Bernoulli drop rate for DARE (0.0 - 0.99)."),
    weights: Optional[str] = typer.Option(None, "--weights", "-w", help="Comma-separated floats for adapter blend weights."),
    target_rank: Optional[int] = typer.Option(None, "--target-rank", "-r", help="Target rank for SVD truncation compression."),
) -> None:
    """Consolidate multiple LoRA adapters using TIES, DARE, SVD rank truncation, or Linear blending."""
    from loraforge.export.merger import AdapterMerger, MergeMethod, MultiLoraMergeConfig

    paths = [p.strip() for p in adapters.split(",") if p.strip()]
    if not paths:
        console.print("[bold red]Error:[/bold red] At least one adapter path must be provided.")
        raise typer.Exit(code=1)

    parsed_weights = None
    if weights:
        try:
            parsed_weights = [float(w.strip()) for w in weights.split(",") if w.strip()]
        except ValueError:
            console.print("[bold red]Error:[/bold red] Weights must be comma-separated numbers.")
            raise typer.Exit(code=1)

    try:
        merge_method = MergeMethod(method.lower())
    except ValueError:
        console.print(f"[bold red]Error:[/bold red] Unsupported merge method '{method}'. Choose from: linear, ties, dare, svd.")
        raise typer.Exit(code=1)

    cfg = MultiLoraMergeConfig(
        method=merge_method,
        adapter_paths=paths,
        weights=parsed_weights,
        density=density,
        drop_rate=drop_rate,
        target_rank=target_rank,
        output_dir=output_dir,
    )

    console.print(f"[bold cyan]Merging {len(paths)} adapters using {cfg.method.value.upper()}...[/bold cyan]")
    report = AdapterMerger.merge_adapters_from_files(paths, cfg)

    table = Table(title="Multi-LoRA Adapter Consolidation Summary", show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold cyan")

    table.add_row("Algorithm", report.method.value.upper())
    table.add_row("Adapters Merged", str(report.num_adapters))
    table.add_row("Tensors Consolidated", str(report.tensors_merged))
    table.add_row("Total Parameters", f"{report.total_parameters:,}")
    table.add_row("Parameter Density / Retention", f"{report.density * 100.0:.1f}%")
    if report.preserved_variance is not None:
        table.add_row("Preserved SVD Variance", f"{report.preserved_variance * 100.0:.2f}%")
    table.add_row("Output Safetensors", report.output_file)

    console.print(table)
    console.print(f"[bold green]✓ Ready for deployment in InferOps![/bold green]\n")


@app.command("serve")
def serve_dashboard(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host address."),
    port: int = typer.Option(8765, "--port", "-p", help="Bind port number."),
) -> None:
    """Start the LoRAForge FastAPI control plane and reactive Web Dashboard."""
    from loraforge.web.app import create_app

    console.print(Panel(
        f"[bold green]Starting LoRAForge Web Control Plane[/bold green]\n\n"
        f"• Dashboard URL: [bold cyan]http://{host}:{port}[/bold cyan]\n"
        f"• API Documentation: [bold cyan]http://{host}:{port}/docs[/bold cyan]\n"
        f"• Inference Companion: [bold]InferOps[/bold]\n\n"
        f"[dim]Press Ctrl+C to terminate server.[/dim]",
        title="LoRAForge Supervisor Control Plane",
        border_style="cyan",
    ))
    web_app = create_app()
    uvicorn.run(web_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
