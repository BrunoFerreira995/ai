"""Export Keras models to SavedModel, TFLite, ONNX, TensorRT, and Core ML."""

from __future__ import annotations

import platform
from pathlib import Path

import tensorflow as tf
from tensorflow import keras


def export_saved_model(model: keras.Model, destination: str | Path) -> Path:
    """Export a TensorFlow SavedModel directory."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(target), save_format="tf")
    return target


def export_tflite(
    model: keras.Model,
    destination: str | Path,
    *,
    quantize: bool = False,
    representative_dataset=None,
) -> Path:
    """Export a Keras model to TFLite, optionally with full INT8 quantization."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        if representative_dataset is None:
            raise ValueError("representative_dataset is required for quantized export")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    target.write_bytes(converter.convert())
    return target


def export_onnx(model: keras.Model, destination: str | Path, *, opset: int = 15) -> Path:
    """Export a Keras model to ONNX through tf2onnx."""
    try:
        import tf2onnx
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("tf2onnx is required for ONNX export") from error
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(model, opset=opset, output_path=str(target))
    return target


def export_tensorrt(
    saved_model_directory: str | Path,
    destination: str | Path,
    *,
    precision: str = "FP32",
) -> Path:
    """Convert a SavedModel to a TensorRT-optimized SavedModel.

    TensorRT requires an NVIDIA CUDA/TensorRT runtime and is not available on
    macOS or CPU-only installations.
    """
    if platform.system() != "Linux":
        raise RuntimeError("TensorRT export requires Linux with an NVIDIA TensorRT runtime")
    if precision.upper() not in {"FP32", "FP16", "INT8"}:
        raise ValueError("precision must be FP32, FP16, or INT8")
    try:
        converter = tf.experimental.tensorrt.Converter(
            input_saved_model_dir=str(saved_model_directory),
            precision_mode=precision.upper(),
        )
        converter.convert()
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        converter.save(str(target))
        return target
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("TensorRT runtime is not installed or unavailable") from error


def export_coreml(model: keras.Model, destination: str | Path) -> Path:
    """Export a Keras model to Core ML ``.mlpackage`` format."""
    try:
        import coremltools as ct
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("coremltools is required for Core ML export") from error
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    converted = ct.convert(model, source="tensorflow")
    converted.save(str(target))
    return target
