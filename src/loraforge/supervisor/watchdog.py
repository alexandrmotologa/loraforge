"""Real-time memory watchdog and proactive OOM circuit breaker.

Monitors GPU VRAM pressure and rate of growth during training steps.
Trips an early circuit breaker before an unrecoverable CUDA out-of-memory crash occurs.
"""

from __future__ import annotations

import time
from typing import Callable, Deque, List, Optional
from collections import deque
from pydantic import BaseModel

from loraforge.core.hardware import GPUTelemetrySnapshot, get_current_telemetry


class WatchdogTripStatus(BaseModel):
    """Evaluation status from the memory watchdog."""

    is_tripped: bool
    current_vram_mb: float
    total_vram_mb: float
    utilization_pct: float
    vram_growth_rate_mb_per_sec: float
    trip_reason: Optional[str] = None


class MemoryWatchdog:
    """Continuously monitors VRAM telemetry to prevent hard driver OOMs."""

    def __init__(
        self,
        max_utilization_pct: float = 95.0,
        min_free_vram_mb: float = 384.0,
        window_size: int = 10,
    ) -> None:
        self.max_utilization_pct = max_utilization_pct
        self.min_free_vram_mb = min_free_vram_mb
        self.window_size = window_size
        self._history: Deque[tuple[float, float]] = deque(maxlen=window_size)  # (timestamp, used_mb)
        self._trip_callbacks: List[Callable[[WatchdogTripStatus], None]] = []
        self._is_tripped = False

    def on_trip(self, callback: Callable[[WatchdogTripStatus], None]) -> None:
        """Register an emergency intervention callback when the watchdog trips."""
        self._trip_callbacks.append(callback)

    def check(self, current_snapshot: Optional[GPUTelemetrySnapshot] = None) -> WatchdogTripStatus:
        """Inspect current VRAM utilization and evaluate trip conditions."""
        now = time.time()

        if current_snapshot is None:
            snaps = get_current_telemetry()
            if not snaps:
                # No GPU telemetry available (CPU / simulated environment)
                return WatchdogTripStatus(
                    is_tripped=False,
                    current_vram_mb=0.0,
                    total_vram_mb=0.0,
                    utilization_pct=0.0,
                    vram_growth_rate_mb_per_sec=0.0,
                )
            current_snapshot = snaps[0]

        used_mb = current_snapshot.vram_used_mb
        total_mb = current_snapshot.vram_total_mb
        free_mb = current_snapshot.vram_free_mb
        util_pct = current_snapshot.vram_utilization_percent

        self._history.append((now, used_mb))

        # Calculate rate of growth (mb/sec)
        growth_rate = 0.0
        if len(self._history) >= 2:
            dt = self._history[-1][0] - self._history[0][0]
            dm = self._history[-1][1] - self._history[0][1]
            if dt > 0:
                growth_rate = round(dm / dt, 2)

        is_tripped = False
        reason = None

        if total_mb > 0:
            if util_pct >= self.max_utilization_pct:
                is_tripped = True
                reason = f"VRAM utilization reached {util_pct:.1f}% (threshold: {self.max_utilization_pct:.1f}%)."
            elif free_mb <= self.min_free_vram_mb:
                is_tripped = True
                reason = f"Free VRAM dropped to {free_mb:.1f} MB (safety minimum: {self.min_free_vram_mb:.1f} MB)."
            elif growth_rate > 500.0 and free_mb < 1024.0:
                # Memory surging at >500MB/s with less than 1GB free
                is_tripped = True
                reason = f"Rapid memory spike ({growth_rate:.1f} MB/s) imminent OOM threshold."

        status = WatchdogTripStatus(
            is_tripped=is_tripped,
            current_vram_mb=used_mb,
            total_vram_mb=total_mb,
            utilization_pct=util_pct,
            vram_growth_rate_mb_per_sec=growth_rate,
            trip_reason=reason,
        )

        if is_tripped and not self._is_tripped:
            self._is_tripped = True
            for cb in self._trip_callbacks:
                try:
                    cb(status)
                except Exception:
                    pass

        return status

    def reset(self) -> None:
        """Reset trip status for a new training phase."""
        self._is_tripped = False
        self._history.clear()
