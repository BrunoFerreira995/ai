"""Practical data-engineering primitives for tabular ML datasets."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_dataframe(
    frame: pd.DataFrame,
    *,
    missing: str = "drop",
    fill_values: dict[str, object] | None = None,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """Clean missing values and duplicate rows without mutating the input.

    ``missing`` accepts ``"drop"`` or ``"fill"``. With ``"fill"``,
    ``fill_values`` may provide a per-column replacement; numeric columns use
    their median and non-numeric columns use their mode by default.
    """
    result = frame.copy()
    if remove_duplicates:
        result = result.drop_duplicates().reset_index(drop=True)
    if missing == "drop":
        result = result.dropna().reset_index(drop=True)
    elif missing == "fill":
        replacements = fill_values or {}
        for column in result.columns:
            if column in replacements:
                value = replacements[column]
            elif pd.api.types.is_numeric_dtype(result[column]):
                value = result[column].median()
            else:
                modes = result[column].mode(dropna=True)
                value = modes.iloc[0] if not modes.empty else "unknown"
            result[column] = result[column].fillna(value)
    else:
        raise ValueError("missing must be 'drop' or 'fill'")
    return result


def detect_outliers_iqr(frame: pd.DataFrame, columns: Iterable[str] | None = None, multiplier: float = 1.5) -> pd.Series:
    """Return a boolean mask for rows containing an IQR outlier."""
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    selected = list(columns) if columns is not None else list(frame.select_dtypes(include=np.number).columns)
    if not selected:
        raise ValueError("at least one numeric column is required")
    mask = pd.Series(False, index=frame.index)
    for column in selected:
        values = frame[column]
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask |= (values < q1 - multiplier * iqr) | (values > q3 + multiplier * iqr)
    return mask


def remove_outliers_iqr(frame: pd.DataFrame, columns: Iterable[str] | None = None, multiplier: float = 1.5) -> pd.DataFrame:
    """Remove rows detected as IQR outliers."""
    return frame.loc[~detect_outliers_iqr(frame, columns, multiplier)].reset_index(drop=True)


def validate_labels(labels: Iterable[object], allowed_labels: Iterable[object]) -> None:
    """Raise ``ValueError`` when a label is missing or not in the allowed set."""
    allowed = set(allowed_labels)
    values = pd.Series(labels)
    if values.isna().any():
        raise ValueError("labels contain missing values")
    invalid = set(values) - allowed
    if invalid:
        raise ValueError(f"unknown labels: {sorted(invalid, key=str)}")


def normalize(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Apply min-max normalization to selected numeric columns."""
    result = frame.copy()
    selected = list(columns) if columns is not None else list(result.select_dtypes(include=np.number).columns)
    for column in selected:
        minimum, maximum = result[column].min(), result[column].max()
        if maximum == minimum:
            result[column] = 0.0
        else:
            result[column] = (result[column] - minimum) / (maximum - minimum)
    return result


def standardize(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Apply z-score standardization to selected numeric columns."""
    result = frame.copy()
    selected = list(columns) if columns is not None else list(result.select_dtypes(include=np.number).columns)
    for column in selected:
        mean, standard_deviation = result[column].mean(), result[column].std(ddof=0)
        result[column] = 0.0 if standard_deviation == 0 else (result[column] - mean) / standard_deviation
    return result


def encode_categorical(frame: pd.DataFrame, columns: Iterable[str], drop_first: bool = False) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    return pd.get_dummies(frame, columns=list(columns), drop_first=drop_first, dtype=float)


def select_features_by_variance(frame: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Keep numeric features whose variance is above ``threshold``."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    numeric = frame.select_dtypes(include=np.number)
    keep = numeric.columns[numeric.var(ddof=0) > threshold]
    return frame.loc[:, keep].copy()


def split_dataset(
    features: pd.DataFrame,
    labels: Iterable[object],
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Split features and labels into train, validation, and test sets."""
    if not 0 < validation_size < 1 or not 0 < test_size < 1 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1")
    feature_index = getattr(features, "index", None)
    labels_series = pd.Series(labels, index=feature_index, name="label")
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels_series,
        test_size=validation_size + test_size,
        random_state=random_state,
        stratify=labels_series if stratify else None,
    )
    relative_test_size = test_size / (validation_size + test_size)
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp if stratify else None,
    )
    return {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
    }
