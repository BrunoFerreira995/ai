"""Portable TensorFlow Lite inference for edge devices."""

from .inference import EdgeInterpreter, detect_platform

__all__ = ["EdgeInterpreter", "detect_platform"]
