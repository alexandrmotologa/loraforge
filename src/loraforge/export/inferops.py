"""InferOps integration bridge and model manifest exporter.

Generates native configuration definitions and deployment scripts to hand off
fine-tuned models directly to InferOps for multi-engine serving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field

from loraforge.core.config import FineTuningJobConfig


class InferOpsModelManifest(BaseModel):
    """InferOps-compatible service descriptor for fine-tuned models."""

    model_id: str
    display_name: str
    engine: str = "vllm"  # vllm, sglang, or llama-cpp
    model_path: str
    base_model_reference: str
    context_window: int = 8192
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    dtype: str = "bfloat16"
    scale_to_zero_idle_minutes: int = 15
    lora_adapters: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class InferOpsExporter:
    """Exports fine-tuned models and generates turnkey manifests for InferOps."""

    @staticmethod
    def generate_manifest(
        config: FineTuningJobConfig,
        model_artifact_path: str,
        engine: str = "vllm",
    ) -> InferOpsModelManifest:
        """Create an InferOps service definition based on training parameters."""
        model_id = config.export.inferops_model_id or f"{config.job_name}-ft"

        return InferOpsModelManifest(
            model_id=model_id,
            display_name=f"{config.job_name} (LoRAForge Fine-Tuned)",
            engine=engine,
            model_path=model_artifact_path,
            base_model_reference=config.model.name,
            context_window=min(config.model.max_position_embeddings, 16384),
            max_model_len=config.training.max_seq_length,
            gpu_memory_utilization=0.90,
            dtype="bfloat16" if config.training.bf16 else "float16",
            scale_to_zero_idle_minutes=15,
            tags=["fine-tuned", "loraforge", config.model.name.split("/")[-1]],
        )

    @classmethod
    def export_bundle(
        cls,
        config: FineTuningJobConfig,
        artifact_dir: Path | str,
        engine: str = "vllm",
    ) -> Path:
        """Write out the manifest YAML, JSON, and launcher script into export directory."""
        target_dir = Path(artifact_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        manifest = cls.generate_manifest(config, str(target_dir), engine=engine)

        # Write YAML manifest
        yaml_path = target_dir / "inferops_manifest.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(manifest.model_dump(), f, sort_keys=False)

        # Write JSON metadata
        json_path = target_dir / "inferops_manifest.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)

        # Write quickstart launcher script
        launcher_sh = target_dir / "serve_in_inferops.sh"
        with open(launcher_sh, "w", encoding="utf-8") as f:
            f.write(f"""#!/usr/bin/env bash
# Quickstart launcher to serve this fine-tuned checkpoint in InferOps
inferops serve \\
  --manifest "{yaml_path.resolve()}" \\
  --engine {engine} \\
  --port 8000
""")

        launcher_bat = target_dir / "serve_in_inferops.bat"
        with open(launcher_bat, "w", encoding="utf-8") as f:
            f.write(f"""@echo off
REM Quickstart launcher to serve this fine-tuned checkpoint in InferOps
inferops serve ^
  --manifest "{yaml_path.resolve()}" ^
  --engine {engine} ^
  --port 8000
""")

        return yaml_path
