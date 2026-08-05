"""Classification evaluation and error-analysis utilities."""

from .evaluation import (
    class_distribution,
    evaluate_classifier,
    error_analysis,
    group_metrics,
    has_class_imbalance,
)

__all__ = ["class_distribution", "evaluate_classifier", "error_analysis", "group_metrics", "has_class_imbalance"]
