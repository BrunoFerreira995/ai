"""Computer-vision baselines and adapters for pretrained models."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


class ImageClassifier:
    """Run image classification with a Keras model."""

    def __init__(self, model, class_names: list[str] | None = None):
        self.model = model
        self.class_names = class_names

    def predict(self, image: np.ndarray) -> dict[str, object]:
        batch = image[np.newaxis, ...] if image.ndim == 3 else image
        probabilities = np.asarray(self.model.predict(batch, verbose=0))
        index = int(np.argmax(probabilities[0]))
        return {
            "class_index": index,
            "class_name": self.class_names[index] if self.class_names else None,
            "confidence": float(probabilities[0, index]),
            "probabilities": probabilities[0].tolist(),
        }


def detect_objects(
    image: np.ndarray,
    model_path: str | Path,
    config_path: str | Path,
    labels: list[str] | None = None,
    confidence_threshold: float = 0.5,
) -> list[dict[str, object]]:
    """Run an OpenCV DNN object detector from model and config files."""
    network = cv2.dnn.readNet(str(model_path), str(config_path))
    height, width = image.shape[:2]
    network.setInput(cv2.dnn.blobFromImage(image, 1 / 255.0, (300, 300), swapRB=True))
    detections = network.forward()
    results = []
    for detection in np.asarray(detections).reshape(-1, detections.shape[-1]):
        confidence = float(detection[2])
        if confidence < confidence_threshold:
            continue
        box = (detection[3:7] * np.array([width, height, width, height])).astype(int).tolist()
        class_index = int(detection[1])
        results.append({"class_index": class_index, "class_name": labels[class_index] if labels else None, "confidence": confidence, "box": box})
    return results


def segment_instances(image: np.ndarray, segmentation_model: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a supplied instance-segmentation model and return its masks."""
    masks = np.asarray(segmentation_model(image[np.newaxis, ...]))
    if masks.ndim < 3:
        raise ValueError("segmentation model must return at least 3 dimensions")
    return masks


def extract_text(image: np.ndarray, language: str = "por") -> str:
    """Extract OCR text through optional Tesseract integration."""
    try:
        import pytesseract
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("install pytesseract and the Tesseract binary for OCR") from error
    return str(pytesseract.image_to_string(image, lang=language))


def estimate_pose(image: np.ndarray, pose_model: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Run a supplied pose-estimation model and return keypoints."""
    keypoints = np.asarray(pose_model(image[np.newaxis, ...]))
    if keypoints.shape[-1] < 2:
        raise ValueError("pose model must return x/y keypoints")
    return keypoints


class FaceRecognizer:
    """Compare face embeddings produced by an external face encoder."""

    def __init__(self, encoder: Callable[[np.ndarray], np.ndarray], threshold: float = 0.75):
        self.encoder = encoder
        self.threshold = threshold

    def similarity(self, first: np.ndarray, second: np.ndarray) -> float:
        first_embedding = np.asarray(self.encoder(first), dtype=float).reshape(-1)
        second_embedding = np.asarray(self.encoder(second), dtype=float).reshape(-1)
        denominator = np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding)
        return float(np.dot(first_embedding, second_embedding) / denominator) if denominator else 0.0

    def is_same_person(self, first: np.ndarray, second: np.ndarray) -> bool:
        return self.similarity(first, second) >= self.threshold


class VideoFrameSampler:
    """Sample frames for video-understanding models."""

    def __init__(self, video_path: str | Path):
        self.video_path = str(video_path)

    def sample(self, count: int = 8) -> list[np.ndarray]:
        if count <= 0:
            raise ValueError("count must be positive")
        capture = cv2.VideoCapture(self.video_path)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(0, total - 1), count, dtype=int)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if success:
                frames.append(frame)
        capture.release()
        return frames
