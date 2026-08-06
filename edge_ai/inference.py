"""TFLite inference with optional hardware delegates."""

from __future__ import annotations

import platform
from pathlib import Path

import numpy as np


def detect_platform() -> str:
    """Return a normalized platform name for deployment diagnostics."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if "android" in system:
        return "android"
    if system == "darwin":
        return "ios" if "iphone" in machine else "macos"
    if system == "linux" and ("jetson" in platform.platform().lower() or "aarch64" in machine):
        return "jetson_or_arm_linux"
    return f"{system}_{machine}"


class EdgeInterpreter:
    """Load and execute a TensorFlow Lite model on CPU or with a delegate."""

    def __init__(self, model_path: str | Path, delegate_path: str | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")
        interpreter_class = None
        load_delegate = None
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate as runtime_load_delegate

            interpreter_class, load_delegate = Interpreter, runtime_load_delegate
        except ImportError:
            import tensorflow as tf

            interpreter_class = tf.lite.Interpreter
        delegates = [load_delegate(delegate_path)] if delegate_path and load_delegate else None
        if delegate_path and not load_delegate:
            raise RuntimeError("A delegate path requires tflite-runtime or TensorFlow delegate support")
        self.interpreter = interpreter_class(model_path=str(self.model_path), experimental_delegates=delegates)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.input_details[0]["shape"])

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Run one batch and return the first output tensor."""
        input_detail = self.input_details[0]
        array = np.asarray(values, dtype=input_detail["dtype"])
        if tuple(array.shape) != self.input_shape:
            raise ValueError(f"expected input shape {self.input_shape}, got {array.shape}")
        self.interpreter.set_tensor(input_detail["index"], array)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])
