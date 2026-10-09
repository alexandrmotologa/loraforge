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


def test_cli_presets_command():
    result = runner.invoke(app, ["presets"])
    assert result.exit_code == 0
    assert "reasoning" in result.output
    assert "coding" in result.output
    assert "agentic" in result.output
    assert "chat" in result.output


def test_cli_init_command(tmp_path):
    out_file = tmp_path / "custom_init.yaml"
    result = runner.invoke(app, [
        "init",
        "--preset", "reasoning",
        "--gpu", "rtx-4090-24gb",
        "--out", str(out_file),
    ])
    assert result.exit_code == 0
    assert "Created fine-tuning configuration" in result.output
    assert out_file.is_file()


def test_cli_cloud_command(tmp_path):
    out_dir = tmp_path / "cloud_cli_out"
    result = runner.invoke(app, [
        "cloud",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--gpu", "rtx-4090-24gb",
        "--out", str(out_dir),
        "--provider", "runpod",
    ])
    assert result.exit_code == 0
    assert "Cloud GPU Deployment Package Created" in result.output
    assert (out_dir / "Dockerfile").is_file()


def test_cli_data_subcommands(tmp_path):
    import json
    data_file = tmp_path / "test_data.jsonl"
    clean_out = tmp_path / "cleaned_data.jsonl"

    sample = {
        "conversations": [
            {"from": "human", "value": "How does LoRA work?"},
            {"from": "gpt", "value": "It decomposes weight updates into low-rank matrices."},
        ]
    }
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")

    # 1. Audit
    audit_res = runner.invoke(app, ["data", "audit", str(data_file), "--format", "sharegpt"])
    assert audit_res.exit_code == 0
    assert "Dataset Audit" in audit_res.output
    assert "Total Samples" in audit_res.output

    # 2. Clean
    clean_res = runner.invoke(app, ["data", "clean", str(data_file), "--output", str(clean_out), "--format", "sharegpt"])
    assert clean_res.exit_code == 0
    assert "Dataset Sanitization Summary" in clean_res.output
    assert clean_out.is_file()

    # 3. Preview
    preview_res = runner.invoke(app, ["data", "preview", str(data_file), "--template", "llama3", "--format", "sharegpt"])
    assert preview_res.exit_code == 0
    assert "LLAMA3" in preview_res.output


def test_cli_train_with_config_file(tmp_path):
    from loraforge.core.presets import build_config_from_preset
    cfg = build_config_from_preset("chat")
    cfg_file = tmp_path / "loraforge_job.yaml"
    cfg_file.write_text(cfg.to_yaml(), encoding="utf-8")

    result = runner.invoke(app, [
        "train",
        "--config", str(cfg_file),
        "--steps", "2",
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "Launching fine-tuning supervisor" in result.output
    assert "Training COMPLETED" in result.output


def test_cli_train_invalid_inputs(tmp_path):
    # Non-existent config
    res_bad_cfg = runner.invoke(app, ["train", "--config", str(tmp_path / "missing.yaml")])
    assert res_bad_cfg.exit_code != 0
    assert "Configuration file not found" in res_bad_cfg.output

    # Invalid model
    res_bad_model = runner.invoke(app, ["train", "--model", "nonexistent-model"])
    assert res_bad_model.exit_code != 0
    assert "Invalid model or GPU" in res_bad_model.output


def test_cli_train_bare_metal_dispatch(monkeypatch):
    from loraforge.supervisor.runner import JobState, JobStatus, TrainingSupervisor

    def mock_run_training_job(self, config, max_steps=None):
        return JobState(
            job_id=config.job_id,
            status=JobStatus.COMPLETED,
            total_steps=max_steps or 2,
            peak_vram_recorded_mb=5400.0,
            output_checkpoint_dir="./test_ckpt",
        )

    monkeypatch.setattr(TrainingSupervisor, "run_training_job", mock_run_training_job)

    result = runner.invoke(app, [
        "train",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--gpu", "rtx-3060-12gb",
        "--no-dry-run",
        "--steps", "2",
    ])
    assert result.exit_code == 0
    assert "Bare-metal PyTorch subprocess execution" in result.output
    assert "Training COMPLETED" in result.output


