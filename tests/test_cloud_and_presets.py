"""Tests for fine-tuning presets, cloud deployment generation, and related Web API endpoints."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from loraforge.core.config import DatasetFormat
from loraforge.core.presets import PRESETS, build_config_from_preset, get_preset
from loraforge.export.cloud import CloudDeploymentGenerator, CloudExportPackage
from loraforge.web.app import create_app

client = TestClient(create_app())


def test_presets_structure_and_lookup():
    for name in ("reasoning", "coding", "agentic", "chat"):
        preset = get_preset(name)
        assert preset is not None
        assert preset.name == name
        assert preset.lora.r >= 16
        assert preset.lora.lora_alpha >= 32
        assert preset.training.max_seq_length >= 2048
        assert len(preset.lora.target_modules) > 0

    assert get_preset("non_existent_preset") is None


def test_build_config_from_preset(tmp_path: Path):
    cfg = build_config_from_preset(
        preset_name="coding",
        gpu_name="rtx-3060-12gb",
        dataset_path="test_dataset.jsonl",
        job_name="my-coding-run",
    )

    assert cfg.job_name == "my-coding-run"
    assert "Qwen2.5-7B" in cfg.model.name
    assert "RTX 3060" in cfg.gpu.name
    assert cfg.lora.r == 32

    # Fallback to chat preset if unknown
    cfg_fallback = build_config_from_preset(preset_name="unknown-preset")
    assert "Llama-3.1-8B" in cfg_fallback.model.name

    yaml_file = tmp_path / "job.yaml"
    cfg.save_yaml(yaml_file)
    assert yaml_file.is_file()


def test_cloud_deployment_generator(tmp_path: Path):
    cfg = build_config_from_preset("reasoning", gpu_name="rtx-4090-24gb")
    out_dir = tmp_path / "cloud_output"

    pkg: CloudExportPackage = CloudDeploymentGenerator.generate(cfg, out_dir, provider="runpod")

    assert pkg.provider == "runpod"
    assert Path(pkg.dockerfile_path).is_file()
    assert Path(pkg.entrypoint_script_path).is_file()
    assert Path(pkg.config_yaml_path).is_file()

    dockerfile_content = Path(pkg.dockerfile_path).read_text(encoding="utf-8")
    assert "FROM nvidia/cuda:12.4.1" in dockerfile_content
    assert "loraforge" in dockerfile_content

    entrypoint_content = Path(pkg.entrypoint_script_path).read_text(encoding="utf-8")
    assert "loraforge train" in entrypoint_content
    assert "loraforge export" in entrypoint_content


def test_api_presets():
    res = client.get("/api/presets")
    assert res.status_code == 200
    data = res.json()
    assert "reasoning" in data
    assert "coding" in data
    assert "agentic" in data
    assert "chat" in data


def test_api_config_yaml():
    payload = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "rtx-3060-12gb",
        "num_gpus": 2,
        "distributed_strategy": "zero_3",
        "seq_len": 4096,
        "batch_size": 1,
        "grad_acc": 8,
        "rank": 32,
        "quant": "nf4",
        "optimizer": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "flash_attention": True,
    }
    res = client.post("/api/config/yaml", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "yaml" in data
    assert "distributed_strategy: zero_3" in data["yaml"]
    assert "num_gpus: 2" in data["yaml"]


def test_api_export_cloud(tmp_path: Path):
    payload = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "rtx-4090-24gb",
    }
    res = client.post("/api/export/cloud", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "runpod"
    assert "dockerfile_path" in data
    assert "entrypoint_script_path" in data


def test_api_job_cancellation_and_not_found():
    # Not found job
    res = client.get("/api/jobs/non-existent-id")
    assert res.status_code == 404

    # Cancel non-running job returns false
    res_cancel = client.post("/api/jobs/non-existent-id/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["cancelled"] is False
