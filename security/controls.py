"""Authentication, authorization, rate limiting, encryption, and robustness tools."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import tensorflow as tf


class APIKeyAuth:
    """Hash API keys and authorize callers with optional roles."""

    def __init__(self, keys: dict[str, str]):
        self._keys = {self._hash(key): role for key, role in keys.items()}

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def authorize(self, key: str, required_role: str | None = None) -> bool:
        role = self._keys.get(self._hash(key))
        if role is None:
            return False
        return required_role is None or role in {required_role, "admin"}


class RateLimiter:
    """In-memory sliding-window rate limiter suitable for one API instance."""

    def __init__(self, limit: int = 60, window_seconds: int = 60):
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit and window_seconds must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_id: str, now: float | None = None) -> bool:
        timestamp = time.time() if now is None else now
        history = self.requests[client_id]
        while history and timestamp - history[0] >= self.window_seconds:
            history.popleft()
        if len(history) >= self.limit:
            return False
        history.append(timestamp)
        return True


def encrypt_file(source: str | Path, destination: str | Path, key: bytes) -> Path:
    """Encrypt a file with Fernet authenticated encryption."""
    from cryptography.fernet import Fernet

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Fernet(key).encrypt(Path(source).read_bytes()))
    return target


def decrypt_file(source: str | Path, destination: str | Path, key: bytes) -> Path:
    """Decrypt and authenticate a Fernet-encrypted file."""
    from cryptography.fernet import Fernet

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Fernet(key).decrypt(Path(source).read_bytes()))
    return target


def validate_input(values: np.ndarray, *, minimum: float = -100.0, maximum: float = 100.0) -> np.ndarray:
    """Reject NaN, infinity, and values outside the model's allowed range."""
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("input contains NaN or infinity")
    if np.any(array < minimum) or np.any(array > maximum):
        raise ValueError("input is outside the allowed range")
    return array


def fgsm_attack(model: tf.keras.Model, inputs: tf.Tensor, labels: tf.Tensor, epsilon: float = 0.01) -> tf.Tensor:
    """Generate a Fast Gradient Sign Method adversarial example."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs, training=False)
        loss = loss_fn(labels, predictions)
    gradient = tape.gradient(loss, inputs)
    return inputs + epsilon * tf.sign(gradient)
