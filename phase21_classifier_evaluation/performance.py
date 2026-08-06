"""Inference performance and TensorFlow profiling helpers."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import numpy as np
import tensorflow as tf


def benchmark_inference(model: tf.keras.Model, inputs: np.ndarray, repeats: int = 10, warmup: int = 2) -> dict[str, float | int]:
    """Measure model load-independent inference latency and throughput."""
    for _ in range(warmup):
        model.predict(inputs, verbose=0)
    tracemalloc.start()
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict(inputs, verbose=0)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latency_ms = elapsed * 1000 / repeats
    return {"batch_size": int(len(inputs)), "repeats": repeats, "latency_ms": latency_ms, "throughput_per_second": repeats * len(inputs) / elapsed, "peak_python_memory_mb": peak / 1024**2}


def benchmark_batch_sizes(model: tf.keras.Model, inputs: np.ndarray, batch_sizes: tuple[int, ...] = (1, 8, 32, 64), repeats: int = 3) -> dict[str, dict[str, float | int]]:
    """Compare inference performance for several batch sizes."""
    results = {}
    for batch_size in batch_sizes:
        batch = np.asarray(inputs)[:batch_size]
        if len(batch) == 0:
            continue
        results[str(batch_size)] = benchmark_inference(model, batch, repeats=repeats, warmup=1)
    return results


def available_devices() -> list[str]:
    """Return TensorFlow-visible CPU/GPU devices."""
    return [device.name for device in tf.config.list_physical_devices()]


def profile_model(model: tf.keras.Model, inputs: np.ndarray, log_dir: str | Path) -> dict[str, str]:
    """Capture a TensorFlow trace for TensorBoard profiling."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    tf.profiler.experimental.start(str(log_path))
    try:
        model.predict(inputs, verbose=0)
    finally:
        tf.profiler.experimental.stop()
    return {"profile_log_dir": str(log_path)}
