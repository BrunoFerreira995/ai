"""Classification, probability, and calibration metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """Calculate multiclass confidence calibration error (ECE)."""
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
    return float(error)


def _safe_metric(function, *args, **kwargs):
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return None


def classification_metrics(
    y_true: Iterable[int],
    probabilities: np.ndarray,
    labels: Iterable[int] | None = None,
    top_k: int = 5,
) -> dict[str, object]:
    """Return the complete Phase 21 report for a multiclass classifier."""
    true = np.asarray(list(y_true), dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or len(true) != len(probabilities):
        raise ValueError("y_true and probabilities must describe the same 2D prediction matrix")
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    label_list = list(labels) if labels is not None else list(range(probabilities.shape[1]))
    top_k = min(top_k, probabilities.shape[1])
    top_indices = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    top5 = float(np.mean([label in row for label, row in zip(true, top_indices)]))
    one_hot = np.eye(probabilities.shape[1])[true]
    per_class = {}
    matrix = confusion_matrix(true, predicted, labels=label_list)
    for index, label in enumerate(label_list):
        selected = true == label
        per_class[str(label)] = float(np.mean(predicted[selected] == label)) if selected.any() else None
    return {
        "accuracy": float(np.mean(true == predicted)),
        "top_1_accuracy": float(np.mean(true == predicted)),
        "top_5_accuracy": top5,
        "precision_weighted": float(precision_score(true, predicted, labels=label_list, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(true, predicted, labels=label_list, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(true, predicted, labels=label_list, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(true, predicted, labels=label_list, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(true, predicted, labels=label_list, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(true, predicted, labels=label_list, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(true, predicted, labels=label_list, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(true, predicted, labels=label_list, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(true, predicted, labels=label_list, average="micro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "cohen_kappa": float(cohen_kappa_score(true, predicted, labels=label_list)),
        "matthews_corrcoef": float(matthews_corrcoef(true, predicted)),
        "roc_auc_ovr": _safe_metric(roc_auc_score, one_hot, probabilities, multi_class="ovr", labels=label_list),
        "pr_auc_macro": _safe_metric(average_precision_score, one_hot, probabilities, average="macro"),
        "log_loss": _safe_metric(log_loss, true, probabilities, labels=label_list),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(true, probabilities),
        "confusion_matrix": matrix.tolist(),
        "per_class_accuracy": per_class,
        "labels": label_list,
    }
