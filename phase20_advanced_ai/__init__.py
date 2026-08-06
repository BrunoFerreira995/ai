"""Advanced AI baselines and model adapters."""

from .vision import (
    FaceRecognizer,
    ImageClassifier,
    VideoFrameSampler,
    detect_objects,
    extract_text,
    estimate_pose,
    segment_instances,
)
from .nlp import DictionaryTranslator, PortugueseTextClassifier

__all__ = [
    "DictionaryTranslator",
    "FaceRecognizer",
    "ImageClassifier",
    "PortugueseTextClassifier",
    "VideoFrameSampler",
    "detect_objects",
    "estimate_pose",
    "extract_text",
    "segment_instances",
]
