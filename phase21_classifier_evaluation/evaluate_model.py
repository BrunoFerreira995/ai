#!/usr/bin/env python3
"""Generate a local evaluation report for the trained classifier."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from .metrics import classification_metrics
from .performance import available_devices, benchmark_batch_sizes, benchmark_inference, profile_model
from .robustness import evaluate_adversarial, evaluate_noise, evaluate_ood


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TensorFlow classifier")
    parser.add_argument("--model", default="artifacts/saved_model")
    parser.add_argument("--data", help="NPZ rotulado com arrays x e y")
    parser.add_argument("--output-dir", default="benchmark_results/classifier")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    load_started = time.perf_counter()
    model = tf.keras.models.load_model(model_path)
    load_time_ms = (time.perf_counter() - load_started) * 1000
    rng = np.random.default_rng(args.seed)
    if args.data:
        dataset = np.load(args.data)
        if "x" not in dataset or "y" not in dataset:
            raise ValueError("O NPZ deve conter arrays x e y")
        inputs = np.asarray(dataset["x"], dtype="float32")
        labels = np.asarray(dataset["y"], dtype=int)
        evaluation_note = "Métricas calculadas contra os rótulos fornecidos no NPZ."
    else:
        inputs = rng.normal(size=(args.samples, args.features)).astype("float32")
        labels = None
    probabilities = model.predict(inputs, verbose=0)
    if labels is None:
        labels = np.argmax(probabilities, axis=1)
        evaluation_note = "Sem --data, foram usados pseudo-rótulos previstos pelo próprio modelo; as métricas não são uma avaliação independente."
    in_distribution = rng.normal(size=inputs.shape).astype("float32")
    out_distribution = (rng.normal(size=inputs.shape) * 8).astype("float32")
    noisy_inputs = (inputs + rng.normal(0.0, 0.05, size=inputs.shape)).astype("float32")
    noisy_probabilities = model.predict(noisy_inputs, verbose=0)
    report = {
        "model": str(model_path),
        "evaluation_note": evaluation_note,
        "model_input_shape": [int(item) if item is not None else None for item in model.input_shape],
        "devices": available_devices(),
        "classification": classification_metrics(labels, probabilities, labels=list(range(probabilities.shape[1]))),
        "noise_robustness": evaluate_noise(model, inputs, labels),
        "noise_calibration": {
            "clean_ece": classification_metrics(labels, probabilities)["ece"],
            "noisy_ece": classification_metrics(labels, noisy_probabilities)["ece"],
        },
        "adversarial_robustness": evaluate_adversarial(model, inputs[: min(64, len(inputs))], labels[: min(64, len(labels))]),
        "ood": evaluate_ood(model, in_distribution[:64], out_distribution[:64]),
        "performance": benchmark_inference(model, inputs[:32]),
        "model_load_time_ms": load_time_ms,
        "batch_sizes": benchmark_batch_sizes(model, inputs),
    }
    if args.profile:
        report["profile"] = profile_model(model, inputs[:32], Path(args.output_dir) / "profile")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "classifier_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Classifier report: {report_path}")
    print(f"Accuracy: {report['classification']['accuracy']:.4f}")
    print(f"Latency: {report['performance']['latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
