"""Data validation, retraining orchestration, A/B testing, and model registry."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def validate_dataset(features: np.ndarray, labels: Iterable[int], *, min_samples: int = 2) -> dict[str, object]:
    """Validate shape, missing values, and label cardinality before training."""
    values = np.asarray(features)
    targets = np.asarray(list(labels))
    errors = []
    if values.ndim < 2:
        errors.append("features must have at least 2 dimensions")
    if len(values) != len(targets):
        errors.append("features and labels must have the same number of samples")
    if len(values) < min_samples:
        errors.append(f"dataset needs at least {min_samples} samples")
    if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
        errors.append("features contain NaN or infinite values")
    if targets.size and len(np.unique(targets)) < 2:
        errors.append("labels must contain at least two classes")
    if errors:
        raise ValueError("dataset validation failed: " + "; ".join(errors))
    return {
        "valid": True,
        "samples": int(len(values)),
        "feature_shape": list(values.shape[1:]),
        "classes": sorted(int(value) for value in np.unique(targets)),
    }


def should_retrain(
    samples_since_training: int,
    *,
    sample_threshold: int = 1000,
    drift_score: float | None = None,
    drift_threshold: float = 0.2,
) -> bool:
    """Trigger retraining after enough new data or detected distribution drift."""
    if sample_threshold <= 0 or drift_threshold < 0:
        raise ValueError("thresholds must be valid positive values")
    return samples_since_training >= sample_threshold or (
        drift_score is not None and drift_score >= drift_threshold
    )


class ModelRegistry:
    """Simple filesystem model registry with versions and a production alias."""

    def __init__(self, root: str | Path = "model_registry"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, model_directory: str | Path, metrics: dict[str, float], version: str | None = None) -> str:
        source = Path(model_directory)
        if not source.is_dir():
            raise FileNotFoundError(f"model directory not found: {source}")
        version = version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        destination = self.root / version
        if destination.exists():
            raise FileExistsError(f"model version already exists: {version}")
        shutil.copytree(source, destination)
        metadata = {"version": version, "metrics": metrics, "created_at": datetime.now(timezone.utc).isoformat()}
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return version

    def promote(self, version: str) -> Path:
        target = self.root / version
        if not target.is_dir():
            raise FileNotFoundError(f"model version not found: {version}")
        alias = self.root / "production"
        if alias.exists() or alias.is_symlink():
            if alias.is_dir() and not alias.is_symlink():
                shutil.rmtree(alias)
            else:
                alias.unlink()
        alias.symlink_to(target.resolve(), target_is_directory=True)
        return alias

    def latest_version(self) -> str | None:
        versions = sorted(path.name for path in self.root.iterdir() if path.is_dir() and path.name != "production")
        return versions[-1] if versions else None


class ABTestRouter:
    """Stable user-to-variant assignment using a deterministic hash."""

    def __init__(self, variants: dict[str, int]):
        if not variants or any(weight <= 0 for weight in variants.values()):
            raise ValueError("variants must contain positive weights")
        self.variants = variants
        self.total_weight = sum(variants.values())

    def assign(self, user_id: str) -> str:
        digest = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        bucket = digest % self.total_weight
        cumulative = 0
        for variant, weight in self.variants.items():
            cumulative += weight
            if bucket < cumulative:
                return variant
        return next(reversed(self.variants))


class RetrainingPipeline:
    """Validate data, train a candidate, evaluate it, and register it."""

    def __init__(self, registry: ModelRegistry, minimum_metric: float = 0.8):
        self.registry = registry
        self.minimum_metric = minimum_metric

    def run(
        self,
        features: np.ndarray,
        labels: Iterable[int],
        train_fn: Callable[[np.ndarray, np.ndarray], str | Path],
        evaluate_fn: Callable[[str | Path], dict[str, float]],
    ) -> dict[str, object]:
        target_array = np.asarray(list(labels))
        validation = validate_dataset(features, target_array)
        model_directory = train_fn(np.asarray(features), target_array)
        metrics = evaluate_fn(model_directory)
        score = float(metrics.get("accuracy", 0.0))
        if score < self.minimum_metric:
            return {"status": "rejected", "validation": validation, "metrics": metrics}
        version = self.registry.register(model_directory, metrics)
        return {"status": "registered", "version": version, "validation": validation, "metrics": metrics}
