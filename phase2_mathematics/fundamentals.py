"""Practical implementations of the Phase 2 mathematics topics.

The functions intentionally use NumPy so they are easy to inspect and reuse
before introducing TensorFlow's automatic differentiation APIs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


Array = np.ndarray


def matrix_multiply(left: Array, right: Array) -> Array:
    """Multiply two compatible matrices or tensors using the last two axes."""
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.ndim < 2 or right_array.ndim < 2:
        raise ValueError("matrix multiplication requires arrays with at least 2 dimensions")
    return np.matmul(left_array, right_array)


def eigendecomposition(matrix: Array) -> tuple[Array, Array]:
    """Return eigenvalues and normalized eigenvectors of a square matrix."""
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2 or matrix_array.shape[0] != matrix_array.shape[1]:
        raise ValueError("eigendecomposition requires a square matrix")
    eigenvalues, eigenvectors = np.linalg.eig(matrix_array)
    return eigenvalues, eigenvectors


def finite_difference_derivative(function: Callable[[float], float], x: float, step: float = 1e-5) -> float:
    """Approximate a one-dimensional derivative with a centered difference."""
    if step <= 0:
        raise ValueError("step must be positive")
    return float((function(x + step) - function(x - step)) / (2 * step))


def partial_derivative(
    function: Callable[[Array], float], point: Array, coordinate: int, step: float = 1e-5
) -> float:
    """Approximate one partial derivative at a point."""
    point_array = np.asarray(point, dtype=float)
    if coordinate < 0 or coordinate >= point_array.size:
        raise IndexError("coordinate is outside the point")
    offset = np.zeros_like(point_array)
    offset[coordinate] = step
    return float((function(point_array + offset) - function(point_array - offset)) / (2 * step))


def gradient(function: Callable[[Array], float], point: Array, step: float = 1e-5) -> Array:
    """Approximate the gradient vector of a scalar function."""
    point_array = np.asarray(point, dtype=float)
    return np.array(
        [partial_derivative(function, point_array, index, step) for index in range(point_array.size)]
    )


def chain_rule(
    outer_derivative: float | Callable[[float], float], inner_derivative: float | Callable[[float], float], value: float
) -> float:
    """Apply ``d(f(g(x)))/dx = f'(g(x)) * g'(x)``.

    Derivative values can be supplied directly, or as functions evaluated at
    ``value``. Supplying callables is useful for symbolic-looking examples.
    """
    outer_value = outer_derivative(value) if callable(outer_derivative) else outer_derivative
    inner_value = inner_derivative(value) if callable(inner_derivative) else inner_derivative
    return float(outer_value * inner_value)


def jacobian(function: Callable[[Array], Array], point: Array, step: float = 1e-5) -> Array:
    """Approximate the Jacobian of a vector-valued function."""
    point_array = np.asarray(point, dtype=float)
    base = np.asarray(function(point_array), dtype=float).reshape(-1)
    result = np.empty((base.size, point_array.size), dtype=float)
    for coordinate in range(point_array.size):
        offset = np.zeros_like(point_array)
        offset[coordinate] = step
        result[:, coordinate] = (
            np.asarray(function(point_array + offset), dtype=float).reshape(-1)
            - np.asarray(function(point_array - offset), dtype=float).reshape(-1)
        ) / (2 * step)
    return result


def hessian(function: Callable[[Array], float], point: Array, step: float = 1e-4) -> Array:
    """Approximate the Hessian matrix of a scalar function."""
    point_array = np.asarray(point, dtype=float)
    size = point_array.size
    result = np.empty((size, size), dtype=float)
    for row in range(size):
        for column in range(size):
            row_offset = np.zeros_like(point_array)
            column_offset = np.zeros_like(point_array)
            row_offset[row] = step
            column_offset[column] = step
            result[row, column] = (
                function(point_array + row_offset + column_offset)
                - function(point_array + row_offset - column_offset)
                - function(point_array - row_offset + column_offset)
                + function(point_array - row_offset - column_offset)
            ) / (4 * step**2)
    return result


def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
    """Calculate P(hypothesis | evidence) from Bayes' theorem."""
    if not 0 <= prior <= 1 or not 0 <= likelihood <= 1 or not 0 < evidence <= 1:
        raise ValueError("probabilities must satisfy 0 <= p <= 1 and evidence must be > 0")
    posterior = prior * likelihood / evidence
    if posterior > 1 + 1e-12:
        raise ValueError("the supplied probabilities produce an invalid posterior")
    return float(posterior)


def gaussian_pdf(value: float | Array, mean: float = 0.0, standard_deviation: float = 1.0) -> float | Array:
    """Evaluate a univariate Gaussian probability density."""
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive")
    value_array = np.asarray(value, dtype=float)
    density = np.exp(-0.5 * ((value_array - mean) / standard_deviation) ** 2)
    density /= standard_deviation * np.sqrt(2 * np.pi)
    return float(density) if density.ndim == 0 else density


def bernoulli_pmf(outcome: int | Array, probability: float) -> float | Array:
    """Evaluate the Bernoulli probability mass function."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    outcome_array = np.asarray(outcome, dtype=int)
    if not np.all((outcome_array == 0) | (outcome_array == 1)):
        raise ValueError("outcome must contain only 0 or 1")
    mass = probability**outcome_array * (1 - probability) ** (1 - outcome_array)
    return float(mass) if mass.ndim == 0 else mass


def softmax(logits: Array, axis: int = -1) -> Array:
    """Convert logits to probabilities using a numerically stable softmax."""
    logits_array = np.asarray(logits, dtype=float)
    shifted = logits_array - np.max(logits_array, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def categorical_cross_entropy(
    targets: Array, predictions: Array, axis: int = -1, epsilon: float = 1e-7
) -> float | Array:
    """Calculate categorical cross-entropy for one or more observations."""
    target_array = np.asarray(targets, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)
    if target_array.shape != prediction_array.shape:
        raise ValueError("targets and predictions must have the same shape")
    clipped = np.clip(prediction_array, epsilon, 1 - epsilon)
    loss = -np.sum(target_array * np.log(clipped), axis=axis)
    return float(loss) if loss.ndim == 0 else loss
