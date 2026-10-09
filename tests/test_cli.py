"""Integration tests for LoRAForge CLI commands using Typer CliRunner."""

from typer.testing import CliRunner
from loraforge.cli.app import app

runner = CliRunner()


def test_cli_models_command():
    result = runner.invoke(app, ["models"], env={"COLUMNS": "160"})
    assert result.exit_code == 0
    assert "8.03B" in result.output
    assert "7.25B" in result.output


def test_cli_hardware_command():
    result = runner.invoke(app, ["hardware"])
    assert result.exit_code == 0
    assert "3060" in result.output
    assert "4090" in result.output


def test_cli_vram_command_success():
    result = runner.invoke(app, [
        "vram",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--gpu", "rtx-3060-12gb",
        "--seq-len", "2048",
        "--batch-size", "1",
        "--rank", "16",
        "--quant", "nf4",
    ])
    assert result.exit_code == 0
    assert "Total Predicted Peak VRAM" in result.output
    assert "FEASIBLE" in result.output or "OPTIMAL" in result.output


def test_cli_vram_unknown_model():
    result = runner.invoke(app, [
        "vram",
        "--model", "non-existent-model",
    ])
    assert result.exit_code != 0
    assert "Unknown model" in result.output


def test_cli_autotune_command():
    result = runner.invoke(app, [
        "autotune",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--gpu", "rtx-3060-12gb",
    ])
    assert result.exit_code == 0
    assert "Auto-Tuned Training Configuration" in result.output
    assert "Micro Batch Size" in result.output


def test_cli_export_command(tmp_path):
    out_dir = tmp_path / "cli_export"
    result = runner.invoke(app, [
        "export",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--out", str(out_dir),
        "--engine", "vllm",
    ])
    assert result.exit_code == 0
    assert "InferOps Service Manifest Exported Successfully" in result.output
    assert (out_dir / "inferops_manifest.yaml").is_file()


def test_cli_train_simulation(tmp_path):
    result = runner.invoke(app, [
        "train",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--gpu", "rtx-3060-12gb",
        "--steps", "5",
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "Launching fine-tuning supervisor" in result.output
    assert "Training COMPLETED" in result.output
