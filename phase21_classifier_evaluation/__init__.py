"""Evaluation utilities for the project's numeric TensorFlow classifier."""

from .metrics import classification_metrics
from .robustness import evaluate_adversarial, evaluate_ood, evaluate_noise
from .performance import benchmark_inference, benchmark_batch_sizes, profile_model

__all__ = [
    "classification_metrics",
    "evaluate_adversarial",
    "evaluate_ood",
    "evaluate_noise",
    "benchmark_inference",
    "benchmark_batch_sizes",
    "profile_model",
]
