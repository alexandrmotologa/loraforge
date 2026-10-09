"""Hardware inspection, GPU capability detection, and real-time telemetry.

Supports direct NVML querying via ctypes, torch.cuda inspection, nvidia-smi fallback,
and preset specifications for popular consumer and workstation GPUs.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from loraforge.core.config import GPUSpec

# Curated reference specifications for consumer and workstation GPUs
PRESET_GPUS: Dict[str, GPUSpec] = {
    "rtx-3060-12gb": GPUSpec(
        name="NVIDIA GeForce RTX 3060 12GB",
        total_vram_mb=12288,
        compute_capability_major=8,
        compute_capability_minor=6,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=768,
    ),
    "rtx-4060-8gb": GPUSpec(
        name="NVIDIA GeForce RTX 4060 8GB",
        total_vram_mb=8192,
        compute_capability_major=8,
        compute_capability_minor=9,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=640,
    ),
    "rtx-4060ti-16gb": GPUSpec(
        name="NVIDIA GeForce RTX 4060 Ti 16GB",
        total_vram_mb=16384,
        compute_capability_major=8,
        compute_capability_minor=9,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=768,
    ),
    "rtx-4070-12gb": GPUSpec(
        name="NVIDIA GeForce RTX 4070 12GB",
        total_vram_mb=12288,
        compute_capability_major=8,
        compute_capability_minor=9,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=768,
    ),
    "rtx-4070ti-16gb": GPUSpec(
        name="NVIDIA GeForce RTX 4070 Ti SUPER 16GB",
        total_vram_mb=16384,
        compute_capability_major=8,
        compute_capability_minor=9,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=768,
    ),
    "rtx-3080-10gb": GPUSpec(
        name="NVIDIA GeForce RTX 3080 10GB",
        total_vram_mb=10240,
        compute_capability_major=8,
        compute_capability_minor=6,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=768,
    ),
    "rtx-3090-24gb": GPUSpec(
        name="NVIDIA GeForce RTX 3090 24GB",
        total_vram_mb=24576,
        compute_capability_major=8,
        compute_capability_minor=6,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=1024,
    ),
    "rtx-4090-24gb": GPUSpec(
        name="NVIDIA GeForce RTX 4090 24GB",
        total_vram_mb=24576,
        compute_capability_major=8,
        compute_capability_minor=9,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=1024,
    ),
    "rtx-6000-ada-48gb": GPUSpec(
        name="NVIDIA RTX 6000 Ada Generation 48GB",
        total_vram_mb=49152,
        compute_capability_major=8,
        compute_capability_minor=9,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=1536,
    ),
    "a100-80gb": GPUSpec(
        name="NVIDIA A100-SXM4-80GB",
        total_vram_mb=81920,
        compute_capability_major=8,
        compute_capability_minor=0,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=2048,
    ),
    "h100-80gb": GPUSpec(
        name="NVIDIA H100 PCIe 80GB",
        total_vram_mb=81920,
        compute_capability_major=9,
        compute_capability_minor=0,
        supports_bf16=True,
        supports_flash_attn_2=True,
        safe_vram_headroom_mb=2048,
    ),
}


class GPUTelemetrySnapshot(BaseModel):
    """Point-in-time hardware monitoring measurement."""

    index: int = 0
    name: str = "Unknown"
    vram_total_mb: float = 0.0
    vram_used_mb: float = 0.0
    vram_free_mb: float = 0.0
    vram_utilization_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None


def get_gpu_preset(gpu_key_or_name: str) -> Optional[GPUSpec]:
    """Lookup reference GPU specs by identifier or fuzzy name."""
    if gpu_key_or_name in PRESET_GPUS:
        return PRESET_GPUS[gpu_key_or_name]

    normalized = gpu_key_or_name.lower().replace(" ", "").replace("_", "-")
    for key, spec in PRESET_GPUS.items():
        if (
            key == normalized
            or key.replace("-", "") in normalized
            or normalized in key.replace("-", "")
            or normalized in spec.name.lower().replace(" ", "").replace("_", "-")
        ):
            return spec

    return None


def probe_nvidia_via_nvml() -> List[GPUTelemetrySnapshot]:
    """Attempt direct query of NVML library using ctypes without external package dependencies."""
    lib_names = ["nvml.dll", "libnvidia-ml.so.1", "libnvidia-ml.so"]
    nvml = None
    for name in lib_names:
        try:
            nvml = ctypes.CDLL(name)
            break
        except (OSError, Exception):
            continue

    if not nvml:
        return []

    results: List[GPUTelemetrySnapshot] = []
    try:
        # nvmlInit()
        init_res = nvml.nvmlInit_v2() if hasattr(nvml, "nvmlInit_v2") else nvml.nvmlInit()
        if init_res != 0:
            return []

        device_count = ctypes.c_uint()
        nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count))

        for i in range(device_count.value):
            handle = ctypes.c_void_p()
            nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle))

            name_buf = ctypes.create_string_buffer(64)
            nvml.nvmlDeviceGetName(handle, name_buf, 64)
            gpu_name = name_buf.value.decode("utf-8", errors="ignore")

            # Memory struct: total, free, used (c_ulonglong)
            class NVMLMemory(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong),
                ]

            mem = NVMLMemory()
            nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
            total_mb = mem.total / (1024 * 1024)
            used_mb = mem.used / (1024 * 1024)
            free_mb = mem.free / (1024 * 1024)
            util_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0

            # Temperature
            temp = ctypes.c_uint()
            temp_res = nvml.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(temp))
            temp_c = float(temp.value) if temp_res == 0 else None

            # Power
            power = ctypes.c_uint()
            p_res = nvml.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power))
            power_w = (power.value / 1000.0) if p_res == 0 else None

            results.append(
                GPUTelemetrySnapshot(
                    index=i,
                    name=gpu_name,
                    vram_total_mb=round(total_mb, 1),
                    vram_used_mb=round(used_mb, 1),
                    vram_free_mb=round(free_mb, 1),
                    vram_utilization_percent=round(util_pct, 1),
                    temperature_c=temp_c,
                    power_draw_w=power_w,
                )
            )

        nvml.nvmlShutdown()
    except Exception:
        pass

    return results


def probe_nvidia_via_smi() -> List[GPUTelemetrySnapshot]:
    """Parse nvidia-smi command-line output if available."""
    smi_bin = shutil.which("nvidia-smi")
    if not smi_bin:
        # Check standard Windows path
        win_path = r"C:\Windows\System32\nvidia-smi.exe"
        if os.path.exists(win_path):
            smi_bin = win_path
        else:
            return []

    try:
        cmd = [
            smi_bin,
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
        snaps: List[GPUTelemetrySnapshot] = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx = int(parts[0])
                name = parts[1]
                total_mb = float(parts[2])
                used_mb = float(parts[3])
                free_mb = float(parts[4])
                gpu_util = float(parts[5]) if len(parts) > 5 and parts[5] != "[N/A]" else 0.0
                temp = float(parts[6]) if len(parts) > 6 and parts[6] != "[N/A]" else None
                power = float(parts[7]) if len(parts) > 7 and parts[7] != "[N/A]" else None

                snaps.append(
                    GPUTelemetrySnapshot(
                        index=idx,
                        name=name,
                        vram_total_mb=total_mb,
                        vram_used_mb=used_mb,
                        vram_free_mb=free_mb,
                        vram_utilization_percent=round((used_mb / total_mb * 100.0), 1) if total_mb > 0 else 0.0,
                        gpu_utilization_percent=gpu_util,
                        temperature_c=temp,
                        power_draw_w=power,
                    )
                )
        return snaps
    except Exception:
        return []


def detect_system_gpus() -> List[GPUSpec]:
    """Inspect local hardware and return detected NVIDIA accelerators as GPUSpec objects."""
    # 1. Try NVML
    snaps = probe_nvidia_via_nvml()
    # 2. Try nvidia-smi if NVML was empty
    if not snaps:
        snaps = probe_nvidia_via_smi()

    specs: List[GPUSpec] = []
    for snap in snaps:
        # Match against known GPU presets for exact compute capabilities
        matched_preset = get_gpu_preset(snap.name)
        if matched_preset:
            specs.append(matched_preset)
        else:
            # Fallback heuristic
            specs.append(
                GPUSpec(
                    name=snap.name,
                    total_vram_mb=int(snap.vram_total_mb),
                    compute_capability_major=8,
                    compute_capability_minor=0,
                    supports_bf16=True,
                    supports_flash_attn_2=True,
                    safe_vram_headroom_mb=max(512, int(snap.vram_total_mb * 0.05)),
                )
            )

    return specs


def get_current_telemetry() -> List[GPUTelemetrySnapshot]:
    """Collect current GPU telemetry readings."""
    snaps = probe_nvidia_via_nvml()
    if snaps:
        return snaps
    return probe_nvidia_via_smi()
