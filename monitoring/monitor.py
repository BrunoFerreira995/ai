"""Logging, drift, performance, resource, and alert monitoring."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import psutil


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure JSON-like application logging and return the monitor logger."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return logging.getLogger("ai.monitoring")


@dataclass(frozen=True)
class MonitoringConfig:
    drift_threshold: float = 0.2
    max_latency_ms: float = 500.0
    max_memory_percent: float = 90.0
    min_accuracy: float = 0.8


class DriftDetector:
    """Detect distribution changes with a simple Population Stability Index."""

    def __init__(self, baseline: Iterable[float], bins: int = 10):
        self.baseline = np.asarray(list(baseline), dtype=float)
        if self.baseline.size < 2 or bins < 2:
            raise ValueError("baseline needs at least 2 values and bins must be >= 2")
        self.bins = bins
        self.edges = np.histogram_bin_edges(self.baseline, bins=bins)
        self.expected = self._distribution(self.baseline)

    def _distribution(self, values: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(values, bins=self.edges)
        probabilities = counts / max(1, counts.sum())
        return np.clip(probabilities, 1e-6, None)

    def score(self, current: Iterable[float]) -> float:
        """Return PSI; values above roughly 0.2 commonly indicate drift."""
        observed = self._distribution(np.asarray(list(current), dtype=float))
        return float(np.sum((observed - self.expected) * np.log(observed / self.expected)))

    def has_drift(self, current: Iterable[float], threshold: float = 0.2) -> bool:
        return self.score(current) >= threshold


class PerformanceMonitor:
    """Record inference latency and summarize throughput."""

    def __init__(self):
        self.latencies_ms: list[float] = []
        self.requests = 0

    def record(self, latency_ms: float) -> None:
        self.latencies_ms.append(float(latency_ms))
        self.requests += 1

    def measure(self, function: Callable, *args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        self.record((time.perf_counter() - start) * 1000)
        return result

    def summary(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {"requests": 0, "avg_latency_ms": 0.0, "p95_latency_ms": 0.0, "throughput_per_second": 0.0}
        values = np.asarray(self.latencies_ms)
        return {
            "requests": float(self.requests),
            "avg_latency_ms": float(values.mean()),
            "p95_latency_ms": float(np.percentile(values, 95)),
            "throughput_per_second": float(1000 / values.mean()) if values.mean() else 0.0,
        }


class ResourceMonitor:
    """Read CPU, memory, and optional NVIDIA GPU utilization."""

    def snapshot(self) -> dict[str, float | None]:
        snapshot: dict[str, float | None] = {
            "cpu_percent": float(psutil.cpu_percent(interval=0.05)),
            "memory_percent": float(psutil.virtual_memory().percent),
            "gpu_percent": None,
        }
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            snapshot["gpu_percent"] = float(output.strip().splitlines()[0])
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
            pass
        return snapshot


class AlertManager:
    """Evaluate monitoring values and notify registered alert handlers."""

    def __init__(self, config: MonitoringConfig = MonitoringConfig(), handlers: Iterable[Callable[[dict], None]] = ()):
        self.config = config
        self.handlers = list(handlers)

    def evaluate(self, *, drift: float | None = None, latency_ms: float | None = None, memory_percent: float | None = None, accuracy: float | None = None) -> list[dict[str, object]]:
        alerts = []
        checks = [
            ("drift", drift, self.config.drift_threshold, lambda value, limit: value >= limit),
            ("latency_ms", latency_ms, self.config.max_latency_ms, lambda value, limit: value >= limit),
            ("memory_percent", memory_percent, self.config.max_memory_percent, lambda value, limit: value >= limit),
            ("accuracy", accuracy, self.config.min_accuracy, lambda value, limit: value < limit),
        ]
        for metric, value, limit, is_alert in checks:
            if value is not None and is_alert(value, limit):
                alert = {"metric": metric, "value": value, "threshold": limit, "message": f"{metric} threshold exceeded"}
                alerts.append(alert)
                for handler in self.handlers:
                    handler(alert)
        return alerts


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    """Write a structured monitoring event."""
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))
