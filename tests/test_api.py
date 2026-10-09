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


def test_api_error_branches():
    # 1. Invalid GPU in estimate
    res1 = client.post("/api/vram/estimate", json={
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "invalid-gpu",
    })
    assert res1.status_code == 404

    # 2. Invalid model in launch_job
    res2 = client.post("/api/jobs", json={
        "job_name": "err-job",
        "model_name": "unknown-model",
        "gpu_name": "rtx-3060-12gb",
    })
    assert res2.status_code == 404

    # 3. Invalid GPU in launch_job
    res3 = client.post("/api/jobs", json={
        "job_name": "err-job",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "unknown-gpu",
    })
    assert res3.status_code == 404

    # 4. Invalid model in export_inferops
    res4 = client.post("/api/export/inferops", json={
        "job_name": "err-job",
        "model_name": "unknown-model",
        "gpu_name": "rtx-3060-12gb",
    })
    assert res4.status_code == 404

    # 5. Invalid GPU in export_inferops
    res5 = client.post("/api/export/inferops", json={
        "job_name": "err-job",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "unknown-gpu",
    })
    assert res5.status_code == 404


def test_api_websocket_telemetry():
    with client.websocket_connect("/ws/telemetry") as websocket:
        assert websocket is not None


def test_api_jobs_launch_options():
    payload_dry = {
        "job_name": "api-dry-test",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "rtx-3060-12gb",
        "total_steps": 2,
        "dry_run": True,
        "dataset_path": "custom_train.jsonl",
        "dataset_format": "alpaca",
    }
    res_dry = client.post("/api/jobs", json=payload_dry)
    assert res_dry.status_code == 200
    assert res_dry.json()["job_id"] is not None

    payload_bare = {
        "job_name": "api-bare-test",
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "gpu_name": "rtx-3060-12gb",
        "total_steps": 2,
        "dry_run": False,
    }
    res_bare = client.post("/api/jobs", json=payload_bare)
    assert res_bare.status_code == 200
    assert res_bare.json()["job_id"] is not None


