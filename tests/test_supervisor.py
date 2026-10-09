"""Unit tests for supervisor watchdog, event bus, and training execution."""

import time
import pytest
from loraforge.core.config import DatasetConfig, FineTuningJobConfig
from loraforge.core.hardware import GPUTelemetrySnapshot
from loraforge.supervisor.runner import JobStatus, TrainingSupervisor
from loraforge.supervisor.telemetry import StepMetrics, TelemetryEventBus
from loraforge.supervisor.watchdog import MemoryWatchdog


def test_telemetry_event_bus():
    bus = TelemetryEventBus(history_limit=10)
    received = []

    def subscriber(metric: StepMetrics):
        received.append(metric)

    bus.subscribe(subscriber)

    m1 = StepMetrics(step=1, epoch=0.1, loss=2.5, learning_rate=1e-4)
    m2 = StepMetrics(step=2, epoch=0.2, loss=2.3, learning_rate=1e-4)
    bus.record(m1)
    bus.record(m2)

    assert len(received) == 2
    assert bus.get_latest().step == 2
    assert len(bus.get_history()) == 2

    bus.unsubscribe(subscriber)
    bus.record(StepMetrics(step=3, epoch=0.3, loss=2.1, learning_rate=1e-4))
    assert len(received) == 2  # No new calls after unsubscribe


def test_memory_watchdog_trip_conditions():
    watchdog = MemoryWatchdog(max_utilization_pct=95.0, min_free_vram_mb=500.0)
    tripped_events = []

    watchdog.on_trip(lambda s: tripped_events.append(s))

    # Safe snapshot
    safe_snap = GPUTelemetrySnapshot(
        name="RTX 3060",
        vram_total_mb=12288.0,
        vram_used_mb=8000.0,
        vram_free_mb=4288.0,
        vram_utilization_percent=65.1,
    )
    status_safe = watchdog.check(safe_snap)
    assert status_safe.is_tripped is False
    assert len(tripped_events) == 0

    # Critical snapshot (utilization > 95%)
    danger_snap = GPUTelemetrySnapshot(
        name="RTX 3060",
        vram_total_mb=12288.0,
        vram_used_mb=11800.0,
        vram_free_mb=488.0,
        vram_utilization_percent=96.0,
    )
    status_danger = watchdog.check(danger_snap)
    assert status_danger.is_tripped is True
    assert len(tripped_events) == 1
    assert "96.0%" in status_danger.trip_reason


def test_training_supervisor_simulation(tmp_path, llama3_8b_spec, rtx_3060_spec):
    supervisor = TrainingSupervisor(work_dir=str(tmp_path))

    cfg = FineTuningJobConfig(
        job_name="test-sim",
        model=llama3_8b_spec,
        gpu=rtx_3060_spec,
        dataset=DatasetConfig(path="dummy.jsonl"),
    )

    state = supervisor.run_dry_run_simulation(cfg, total_steps=5, step_delay=0.001)

    assert state.status == JobStatus.COMPLETED
    assert state.current_step == 5
    assert state.latest_loss is not None
    assert state.peak_vram_recorded_mb > 0
    assert (tmp_path / cfg.job_id / "checkpoints" / "adapter_config.json").is_file()
