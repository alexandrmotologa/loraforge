"""Integration tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
from loraforge.web.app import create_app

client = TestClient(create_app())


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "loraforge"


def test_api_hardware():
    res = client.get("/api/hardware")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert "rtx-3060-12gb" in data["presets"]


def test_api_models():
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "meta-llama/Llama-3.1-8B-Instruct" in data


def test_api_vram_estimate():
    payload = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "rtx-3060-12gb",
        "seq_len": 2048,
        "batch_size": 1,
        "grad_acc": 8,
        "rank": 16,
        "quant": "nf4",
        "optimizer": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "flash_attention": True,
    }
    res = client.post("/api/vram/estimate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "peak_vram_mb" in data
    assert "breakdown" in data
    assert data["status"] in ("OPTIMAL", "FEASIBLE")


def test_api_vram_estimate_unknown_model():
    payload = {
        "model_name": "invalid-model",
        "gpu_name": "rtx-3060-12gb",
    }
    res = client.post("/api/vram/estimate", json=payload)
    assert res.status_code == 404


def test_api_vram_autotune():
    payload = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "rtx-3060-12gb",
        "effective_batch": 16,
        "target_seq_len": 2048,
    }
    res = client.post("/api/vram/autotune", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "optimal_micro_batch_size" in data
    assert "recommended_quantization" in data


def test_api_jobs_launch_and_list():
    payload = {
        "job_name": "api-test-job",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "rtx-3060-12gb",
        "total_steps": 5,
    }
    launch_res = client.post("/api/jobs", json=payload)
    assert launch_res.status_code == 200
    job_data = launch_res.json()
    job_id = job_data["job_id"]

    list_res = client.get("/api/jobs")
    assert list_res.status_code == 200
    jobs = list_res.json()
    assert any(j["job_id"] == job_id for j in jobs)


def test_api_export_inferops(tmp_path):
    payload = {
        "job_name": "export-api-test",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "rtx-3060-12gb",
        "engine": "vllm",
        "output_dir": str(tmp_path / "api_inferops"),
    }
    res = client.post("/api/export/inferops", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "inferops serve" in data["inferops_command"]


def test_api_dashboard_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "LoRAForge" in res.text
    assert "InferOps Companion" in res.text
