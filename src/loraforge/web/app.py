"""FastAPI application factory and REST/WebSocket control plane for LoRAForge."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
from loraforge.export.inferops import InferOpsExporter
from loraforge.supervisor.runner import JobState, TrainingSupervisor
from loraforge.supervisor.telemetry import StepMetrics
from loraforge.vram.calculator import VRAMCalculator, VRAMEstimateReport
from loraforge.vram.optimizer import HyperparameterOptimizer, OptimizationRecommendation


class EstimateRequest(BaseModel):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    gpu_name: str = "rtx-3060-12gb"
    seq_len: int = 2048
    batch_size: int = 1
    grad_acc: int = 8
    rank: int = 16
    quant: str = "nf4"
    optimizer: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    flash_attention: bool = True


class AutoTuneRequest(BaseModel):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    gpu_name: str = "rtx-3060-12gb"
    effective_batch: int = 16
    target_seq_len: int = 2048


class LaunchJobRequest(BaseModel):
    job_name: str = "my-qlora-experiment"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    gpu_name: str = "rtx-3060-12gb"
    seq_len: int = 2048
    batch_size: int = 1
    grad_acc: int = 8
    rank: int = 16
    quant: str = "nf4"
    optimizer: str = "paged_adamw_8bit"
    total_steps: int = 50


class ExportInferOpsRequest(BaseModel):
    job_name: str = "my-qlora-experiment"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    gpu_name: str = "rtx-3060-12gb"
    engine: str = "vllm"
    output_dir: str = "./exports/inferops-model"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="LoRAForge Control Plane",
        description="Bare-metal local fine-tuning control plane and QLoRA supervisor for consumer and workstation GPUs. Companion to InferOps.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    supervisor = TrainingSupervisor()
    calc = VRAMCalculator()
    optimizer_engine = HyperparameterOptimizer()

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        """Serve embedded single-page dashboard."""
        index_file = static_dir / "index.html"
        if index_file.is_file():
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>LoRAForge Control Plane</h1><p>Static UI building...</p>"

    @app.get("/api/health")
    async def health_check():
        return {"status": "healthy", "service": "loraforge", "version": "0.1.0"}

    @app.get("/api/hardware")
    async def get_hardware():
        return {
            "telemetry": [s.model_dump() for s in get_current_telemetry()],
            "detected": [d.model_dump() for d in detect_system_gpus()],
            "presets": {k: v.model_dump() for k, v in PRESET_GPUS.items()},
        }

    @app.get("/api/models")
    async def get_models():
        return {k: v.model_dump() for k, v in PRESET_MODELS.items()}

    @app.post("/api/vram/estimate", response_model=VRAMEstimateReport)
    async def estimate_vram(req: EstimateRequest):
        model_spec = get_model_spec(req.model_name)
        if not model_spec:
            raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found.")

        gpu_spec = get_gpu_preset(req.gpu_name)
        if not gpu_spec:
            raise HTTPException(status_code=404, detail=f"GPU '{req.gpu_name}' not found.")

        lora_cfg = LoRAConfig(r=req.rank, lora_alpha=req.rank * 2)
        quant_cfg = QuantizationConfig(
            quant_type=QuantizationType(req.quant),
            double_quant=True,
        )
        training_cfg = TrainingConfig(
            per_device_train_batch_size=req.batch_size,
            gradient_accumulation_steps=req.grad_acc,
            max_seq_length=req.seq_len,
            optimizer=OptimizerType(req.optimizer),
            attention_impl=AttentionImplementation.FLASH_ATTENTION_2 if req.flash_attention else AttentionImplementation.SDPA,
            gradient_checkpointing=req.gradient_checkpointing,
        )

        return calc.estimate(model_spec, lora_cfg, quant_cfg, training_cfg, gpu_spec)

    @app.post("/api/vram/autotune", response_model=OptimizationRecommendation)
    async def autotune(req: AutoTuneRequest):
        model_spec = get_model_spec(req.model_name)
        if not model_spec:
            raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found.")

        gpu_spec = get_gpu_preset(req.gpu_name)
        if not gpu_spec:
            raise HTTPException(status_code=404, detail=f"GPU '{req.gpu_name}' not found.")

        return optimizer_engine.auto_tune_for_gpu(
            model_spec,
            gpu_spec,
            target_effective_batch_size=req.effective_batch,
            target_seq_len=req.target_seq_len,
        )

    @app.get("/api/jobs", response_model=List[JobState])
    async def list_jobs():
        return supervisor.list_jobs()

    @app.get("/api/jobs/{job_id}", response_model=JobState)
    async def get_job(job_id: str):
        job = supervisor.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return job

    @app.post("/api/jobs", response_model=JobState)
    async def launch_job(req: LaunchJobRequest):
        model_spec = get_model_spec(req.model_name)
        if not model_spec:
            raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found.")

        gpu_spec = get_gpu_preset(req.gpu_name)
        if not gpu_spec:
            raise HTTPException(status_code=404, detail=f"GPU '{req.gpu_name}' not found.")

        cfg = FineTuningJobConfig(
            job_name=req.job_name,
            model=model_spec,
            lora=LoRAConfig(r=req.rank, lora_alpha=req.rank * 2),
            quantization=QuantizationConfig(quant_type=QuantizationType(req.quant)),
            training=TrainingConfig(
                per_device_train_batch_size=req.batch_size,
                gradient_accumulation_steps=req.grad_acc,
                max_seq_length=req.seq_len,
                optimizer=OptimizerType(req.optimizer),
            ),
            gpu=gpu_spec,
            dataset=DatasetConfig(path="dataset.jsonl", format=DatasetFormat.SHAREGPT),
        )

        # Launch in background thread to avoid blocking FastAPI event loop
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None,
            supervisor.run_dry_run_simulation,
            cfg,
            req.total_steps,
            0.05,
        )
        await asyncio.sleep(0.05)
        job = supervisor.get_job(cfg.job_id)
        if not job:
            job = JobState(job_id=cfg.job_id, status="pending", total_steps=req.total_steps)
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        success = supervisor.cancel_job(job_id)
        return {"job_id": job_id, "cancelled": success}

    @app.post("/api/export/inferops")
    async def export_inferops(req: ExportInferOpsRequest):
        model_spec = get_model_spec(req.model_name)
        if not model_spec:
            raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found.")

        gpu_spec = get_gpu_preset(req.gpu_name)
        if not gpu_spec:
            raise HTTPException(status_code=404, detail=f"GPU '{req.gpu_name}' not found.")

        cfg = FineTuningJobConfig(
            job_name=req.job_name,
            model=model_spec,
            gpu=gpu_spec,
            dataset=DatasetConfig(path="dataset.jsonl", format=DatasetFormat.SHAREGPT),
        )

        manifest_path = InferOpsExporter.export_bundle(cfg, req.output_dir, engine=req.engine)
        return {
            "status": "success",
            "manifest_path": str(manifest_path.resolve()),
            "engine": req.engine,
            "inferops_command": f"inferops serve --manifest {manifest_path.resolve()} --engine {req.engine} --port 8000",
        }

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket):
        await websocket.accept()
        queue: asyncio.Queue[StepMetrics] = asyncio.Queue()

        def on_metric(m: StepMetrics):
            asyncio.run_coroutine_threadsafe(queue.put(m), asyncio.get_event_loop())

        supervisor.event_bus.subscribe(on_metric)
        try:
            while True:
                metric = await queue.get()
                await websocket.send_text(metric.model_dump_json())
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            supervisor.event_bus.unsubscribe(on_metric)

    return app
