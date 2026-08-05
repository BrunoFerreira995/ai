#!/usr/bin/env python3
"""Train, evaluate, and export a TensorFlow classifier.

Examples:
    .venv/bin/python train.py
    .venv/bin/python train.py --data data/dataset.npz --epochs 10

An input NPZ file must contain arrays named ``x`` and ``y``. ``x`` contains
the samples and ``y`` contains integer class labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from phase4_data_engineering.pipeline import split_dataset
from phase7_model_design import ModelConfig, build_classifier
from phase8_training import TrainingConfig, train_model
from phase10_evaluation import evaluate_classifier
from phase13_export import export_onnx, export_saved_model, export_tflite


def make_demo_dataset(samples: int, features: int, classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a small linearly separable demo dataset."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2, size=(classes, features)).astype(np.float32)
    labels = rng.integers(0, classes, size=samples)
    values = centers[labels] + rng.normal(0, 0.6, size=(samples, features)).astype(np.float32)
    return values, labels.astype(np.int32)


def load_dataset(path: str | None, samples: int, features: int, classes: int, seed: int):
    if path is None:
        return make_demo_dataset(samples, features, classes, seed)
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {dataset_path}. "
            "Remova --data para usar o dataset de demonstração ou crie um .npz com x e y."
        )
    data = np.load(dataset_path)
    if "x" not in data or "y" not in data:
        raise ValueError("NPZ dataset must contain arrays named 'x' and 'y'")
    x, y = np.asarray(data["x"], dtype=np.float32), np.asarray(data["y"], dtype=np.int32)
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of samples")
    return x, y


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export a TensorFlow classifier")
    parser.add_argument("--data", help="optional NPZ file containing x and y arrays")
    parser.add_argument("--output-dir", default="artifacts", help="directory for models and metrics")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=600, help="demo sample count when --data is omitted")
    parser.add_argument("--features", type=int, default=8, help="demo feature count")
    parser.add_argument("--classes", type=int, default=3, help="demo class count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    x, y = load_dataset(args.data, args.samples, args.features, args.classes, args.seed)
    classes = int(np.max(y)) + 1
    splits = split_dataset(x if isinstance(x, np.ndarray) else np.asarray(x), y, validation_size=0.2, test_size=0.2)
    x_train, y_train = splits["train"]
    x_validation, y_validation = splits["validation"]
    x_test, y_test = splits["test"]

    config = ModelConfig(input_shape=tuple(x.shape[1:]), num_classes=classes, backbone="dense")
    model = build_classifier(config)
    output_dir = Path(args.output_dir)
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        log_dir=str(output_dir / "logs"),
        checkpoint_path=str(output_dir / "checkpoints" / "best.keras"),
    )
    train_model(model, x_train, y_train, x_validation, y_validation, training_config)

    probabilities = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=-1)
    metrics = evaluate_classifier(y_test, predictions, probabilities, labels=list(range(classes)))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, default=json_safe, indent=2), encoding="utf-8")
    export_saved_model(model, output_dir / "saved_model")
    export_tflite(model, output_dir / "model.tflite")
    export_onnx(model, output_dir / "model.onnx")

    print(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
