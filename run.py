#!/usr/bin/env python3
"""Run inference with an exported TensorFlow SavedModel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if sys.prefix == sys.base_prefix:
    venv_python = Path(__file__).resolve().parent / ".venv/bin/python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])

import numpy as np
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the trained TensorFlow model")
    parser.add_argument("--model", default="artifacts/saved_model", help="SavedModel directory")
    parser.add_argument("--input", help="optional .npy file containing one or more input samples")
    parser.add_argument("--features", type=int, default=8, help="feature count for generated demo input")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}. Execute ./train.sh primeiro.")

    model = tf.keras.models.load_model(model_path)
    classes_file = model_path.parent / "classes.json"
    class_names = json.loads(classes_file.read_text(encoding="utf-8")) if classes_file.exists() else None
    rules_file = model_path.parent / "business_rules.json"
    business_rules = json.loads(rules_file.read_text(encoding="utf-8")) if rules_file.exists() else {}
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(f"Entrada não encontrada: {input_path}")
        values = np.load(input_path).astype("float32")
    else:
        rng = np.random.default_rng(args.seed)
        values = rng.normal(size=(1, args.features)).astype("float32")

    if values.ndim == len(model.input_shape) - 1:
        values = values[np.newaxis, ...]
    probabilities = model.predict(values, verbose=0)
    predictions = np.argmax(probabilities, axis=-1)
    result = {"classes": predictions.tolist(), "probabilities": probabilities.tolist()}
    if class_names:
        result["class_names"] = [class_names[int(index)] for index in predictions]
        result["responses"] = [business_rules.get(name, f"A classificação recebida foi {name}. Como posso ajudar?") for name in result["class_names"]]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
