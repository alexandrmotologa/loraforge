"""Unit tests for hardware presets and probe routines."""

from loraforge.core.hardware import (
    PRESET_GPUS,
    detect_system_gpus,
    get_current_telemetry,
    get_gpu_preset,
)


def test_preset_lookup():
    rtx_3060 = get_gpu_preset("rtx-3060-12gb")
    assert rtx_3060 is not None
    assert rtx_3060.total_vram_mb == 12288
    assert rtx_3060.compute_capability == 8.6

    rtx_4090 = get_gpu_preset("rtx-4090-24gb")
    assert rtx_4090 is not None
    assert rtx_4090.total_vram_mb == 24576
    assert rtx_4090.compute_capability == 8.9

    fuzzy = get_gpu_preset("rtx 3090")
    assert fuzzy is not None
    assert fuzzy.total_vram_mb == 24576

    unknown = get_gpu_preset("non_existent_gpu")
    assert unknown is None


def test_hardware_probe_runs_safely():
    # Should execute without throwing unexpected exceptions even in headless / non-NVIDIA environments
    telemetry = get_current_telemetry()
    assert isinstance(telemetry, list)

    detected = detect_system_gpus()
    assert isinstance(detected, list)
