"""Noise, adversarial, and out-of-distribution evaluation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def _accuracy(model: tf.keras.Model, inputs: np.ndarray, labels: np.ndarray) -> float:
    probabilities = model.predict(inputs, verbose=0)
    return float(np.mean(np.argmax(probabilities, axis=1) == labels))


def evaluate_noise(model: tf.keras.Model, inputs: np.ndarray, labels: np.ndarray, noise_std: float = 0.05, seed: int = 42) -> dict[str, float]:
    """Compare clean accuracy with Gaussian-noise accuracy."""
    clean = _accuracy(model, inputs, labels)
    rng = np.random.default_rng(seed)
    noisy = np.asarray(inputs) + rng.normal(0.0, noise_std, size=np.asarray(inputs).shape)
    return {"clean_accuracy": clean, "noisy_accuracy": _accuracy(model, noisy.astype(np.float32), labels), "noise_std": float(noise_std)}


def fgsm_attack(model: tf.keras.Model, inputs: np.ndarray, labels: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Create Fast Gradient Sign Method examples."""
    values = tf.Variable(inputs, dtype=tf.float32)
    targets = tf.convert_to_tensor(labels, dtype=tf.int32)
    with tf.GradientTape() as tape:
        tape.watch(values)
        logits = model(values, training=True)
        selected = tf.gather(logits, targets, axis=1, batch_dims=1)
        loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(selected, 1e-7, 1.0)))
    gradient = tape.gradient(tf.reduce_mean(loss), values)
    if gradient is None:
        raise RuntimeError("Não foi possível calcular o gradiente FGSM para este modelo")
    return (values + epsilon * tf.sign(gradient)).numpy()


def pgd_attack(model: tf.keras.Model, inputs: np.ndarray, labels: np.ndarray, epsilon: float = 0.03, step_size: float = 0.005, steps: int = 5) -> np.ndarray:
    """Create projected-gradient-descent examples."""
    original = tf.convert_to_tensor(inputs, dtype=tf.float32)
    adversarial = tf.Variable(original)
    targets = tf.convert_to_tensor(labels, dtype=tf.int32)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adversarial)
            logits = model(adversarial, training=True)
            selected = tf.gather(logits, targets, axis=1, batch_dims=1)
            loss = -tf.reduce_mean(tf.math.log(tf.clip_by_value(selected, 1e-7, 1.0)))
        gradient = tape.gradient(tf.reduce_mean(loss), adversarial)
        if gradient is None:
            raise RuntimeError("Não foi possível calcular o gradiente PGD para este modelo")
        adversarial = tf.Variable(adversarial + step_size * tf.sign(gradient))
        adversarial = tf.clip_by_value(adversarial, original - epsilon, original + epsilon)
    return adversarial.numpy()


def evaluate_adversarial(model: tf.keras.Model, inputs: np.ndarray, labels: np.ndarray, epsilon: float = 0.01, pgd_steps: int = 5) -> dict[str, float]:
    """Evaluate clean, FGSM, and PGD accuracy."""
    result: dict[str, float | str | None] = {"clean_accuracy": _accuracy(model, inputs, labels)}
    try:
        fgsm = fgsm_attack(model, inputs, labels, epsilon)
        pgd = pgd_attack(model, inputs, labels, epsilon=max(epsilon, 0.03), steps=pgd_steps)
        result.update(fgsm_accuracy=_accuracy(model, fgsm, labels), pgd_accuracy=_accuracy(model, pgd, labels))
    except RuntimeError as error:
        result.update(fgsm_accuracy=None, pgd_accuracy=None, status=f"indisponível: {error}")
    return result


def evaluate_ood(model: tf.keras.Model, in_distribution: np.ndarray, out_distribution: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Measure simple max-probability OOD detection."""
    in_probabilities = model.predict(in_distribution, verbose=0)
    out_probabilities = model.predict(out_distribution, verbose=0)
    in_confidence = in_probabilities.max(axis=1)
    out_confidence = out_probabilities.max(axis=1)
    in_accept = in_confidence >= threshold
    out_reject = out_confidence < threshold
    return {"threshold": float(threshold), "in_distribution_accept_rate": float(in_accept.mean()), "ood_rejection_rate": float(out_reject.mean()), "ood_detection_rate": float((in_accept.sum() + out_reject.sum()) / (len(in_accept) + len(out_reject)))}
