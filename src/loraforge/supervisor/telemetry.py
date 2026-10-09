"""Real-time training metrics, telemetry event bus, and step event streaming."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional
from pydantic import BaseModel, Field


class StepMetrics(BaseModel):
    """Snapshot of fine-tuning telemetry for an individual optimization step."""

    step: int
    epoch: float
    loss: float
    val_loss: Optional[float] = None
    perplexity: Optional[float] = None
    is_best_checkpoint: bool = False
    learning_rate: float
    grad_norm: Optional[float] = None
    tokens_per_second: float = 0.0
    vram_used_mb: float = 0.0
    vram_peak_mb: float = 0.0
    temperature_c: Optional[float] = None
    elapsed_seconds: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class TelemetryEventBus:
    """Thread-safe telemetry buffer and subscriber dispatcher for WebSocket / CLI feeds."""

    def __init__(self, history_limit: int = 1000) -> None:
        self.history_limit = history_limit
        self._history: Deque[StepMetrics] = deque(maxlen=history_limit)
        self._subscribers: List[Callable[[StepMetrics], None]] = []

    def record(self, metric: StepMetrics) -> None:
        """Push a new step metric and notify all active listeners."""
        self._history.append(metric)
        for listener in self._subscribers:
            try:
                listener(metric)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[StepMetrics], None]) -> None:
        """Register a real-time event listener."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[StepMetrics], None]) -> None:
        """Remove a previously registered listener."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def get_history(self) -> List[StepMetrics]:
        """Return full recorded step history in chronological order."""
        return list(self._history)

    def get_latest(self) -> Optional[StepMetrics]:
        """Return the most recent metric entry."""
        return self._history[-1] if self._history else None
