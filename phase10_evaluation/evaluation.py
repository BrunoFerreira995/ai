"""Metrics and diagnostics for supervised classification models."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true: Iterable[object],
    y_pred: Iterable[object],
    y_probabilities: np.ndarray | None = None,
    labels: Iterable[object] | None = None,
) -> dict[str, object]:
    """Calculate standard classification metrics and a confusion matrix."""
    true = np.asarray(list(y_true))
    predicted = np.asarray(list(y_pred))
    if true.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    label_list = list(labels) if labels is not None else sorted(set(true) | set(predicted), key=str)
    results: dict[str, object] = {
        "accuracy": float(accuracy_score(true, predicted)),
        "precision": float(precision_score(true, predicted, labels=label_list, average="weighted", zero_division=0)),
        "recall": float(recall_score(true, predicted, labels=label_list, average="weighted", zero_division=0)),
        "f1": float(f1_score(true, predicted, labels=label_list, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, predicted, labels=label_list),
        "labels": label_list,
    }
    if y_probabilities is not None:
        probabilities = np.asarray(y_probabilities)
        try:
            if probabilities.ndim == 1 or probabilities.shape[1] == 2:
                positive = probabilities if probabilities.ndim == 1 else probabilities[:, 1]
                results["roc_auc"] = float(roc_auc_score(true, positive))
            else:
                results["roc_auc"] = float(roc_auc_score(true, probabilities, multi_class="ovr", labels=label_list))
        except ValueError:
            results["roc_auc"] = None
    else:
        results["roc_auc"] = None
    return results


def error_analysis(y_true: Iterable[object], y_pred: Iterable[object]) -> pd.DataFrame:
    """Return one row per prediction with binary FP/FN and error flags."""
    true = np.asarray(list(y_true))
    predicted = np.asarray(list(y_pred))
    if true.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    binary = set(true) | set(predicted) <= {0, 1}
    return pd.DataFrame(
        {
            "index": np.arange(len(true)),
            "true": true,
            "predicted": predicted,
            "is_error": true != predicted,
            "false_positive": (predicted == 1) & (true == 0) if binary else false_mask(true, predicted),
            "false_negative": (predicted == 0) & (true == 1) if binary else false_mask(true, predicted),
        }
    )


def false_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mark incorrect predictions for multiclass error analysis."""
    return y_true != y_pred


def class_distribution(labels: Iterable[object]) -> pd.Series:
    """Return class counts sorted by class label."""
    return pd.Series(labels).value_counts().sort_index()


def has_class_imbalance(labels: Iterable[object], threshold: float = 2.0) -> bool:
    """Detect imbalance when the largest class is at least ``threshold`` times the smallest."""
    if threshold < 1:
        raise ValueError("threshold must be at least 1")
    counts = class_distribution(labels)
    if counts.empty or counts.min() == 0:
        return False
    return bool(counts.max() / counts.min() >= threshold)


def group_metrics(
    y_true: Iterable[object], y_pred: Iterable[object], groups: Iterable[object]
) -> pd.DataFrame:
    """Calculate accuracy, precision, recall, and sample count per subgroup."""
    frame = pd.DataFrame({"true": list(y_true), "predicted": list(y_pred), "group": list(groups)})
    if not (len(frame["true"]) == len(frame["predicted"]) == len(frame["group"])):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    rows = []
    for group, subset in frame.groupby("group", sort=True):
        rows.append(
            {
                "group": group,
                "samples": len(subset),
                "accuracy": accuracy_score(subset["true"], subset["predicted"]),
                "precision": precision_score(subset["true"], subset["predicted"], average="weighted", zero_division=0),
                "recall": recall_score(subset["true"], subset["predicted"], average="weighted", zero_division=0),
            }
        )
    return pd.DataFrame(rows)
