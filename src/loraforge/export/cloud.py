"""Cloud deployment generator for RunPod, Vast.ai, and containerized GPU clusters.

Produces production Dockerfile definitions and bootstrap entrypoint scripts
for executing supervised fine-tuning jobs on cloud GPU instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel

from loraforge.core.config import FineTuningJobConfig


class CloudExportPackage(BaseModel):
    """Artifact metadata for generated cloud deployment manifests."""

    provider: str
    dockerfile_path: str
    entrypoint_script_path: str
    config_yaml_path: str


class CloudDeploymentGenerator:
    """Generates turnkey containerized execution scripts for remote GPUs."""

    DOCKERFILE_TEMPLATE = """# LoRAForge GPU Training Container
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3-pip \\
    python3-dev \\
    git \\
    curl \\
    ca-certificates \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \\
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install training accelerators
RUN pip install --no-cache-dir \\
    transformers>=4.40.0 \\
    peft>=0.10.0 \\
    bitsandbytes>=0.43.0 \\
    accelerate>=0.29.0 \\
    trl>=0.8.0 \\
    datasets>=2.18.0

# Install LoRAForge
RUN pip install --no-cache-dir loraforge

COPY . /workspace/

CMD ["bash", "/workspace/runpod_entrypoint.sh"]
"""

    ENTRYPOINT_TEMPLATE = """#!/usr/bin/env bash
set -e

echo "[LoRAForge Cloud] Initializing GPU fine-tuning job on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)..."

# Pre-flight VRAM inspection
loraforge vram --config /workspace/loraforge.yaml

# Execute fine-tuning supervisor
echo "[LoRAForge Cloud] Launching training supervisor..."
loraforge train --config /workspace/loraforge.yaml

echo "[LoRAForge Cloud] Exporting model artifacts and InferOps manifests..."
loraforge export --config /workspace/loraforge.yaml --out /workspace/exports

echo "[LoRAForge Cloud] Job completed successfully!"
"""

    @classmethod
    def generate(
        cls,
        config: FineTuningJobConfig,
        output_dir: Path | str,
        provider: str = "runpod",
    ) -> CloudExportPackage:
        """Generate Dockerfile, entrypoint script, and serialized config in output directory."""
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)

        dockerfile = target / "Dockerfile"
        dockerfile.write_text(cls.DOCKERFILE_TEMPLATE, encoding="utf-8")

        entrypoint = target / f"{provider}_entrypoint.sh"
        entrypoint.write_text(cls.ENTRYPOINT_TEMPLATE, encoding="utf-8")

        config_path = target / "loraforge.yaml"
        config.save_yaml(config_path)

        return CloudExportPackage(
            provider=provider,
            dockerfile_path=str(dockerfile.resolve()),
            entrypoint_script_path=str(entrypoint.resolve()),
            config_yaml_path=str(config_path.resolve()),
        )
