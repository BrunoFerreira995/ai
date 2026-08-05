"""Storage helpers for datasets collected during Phase 3."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def write_csv(rows: Iterable[dict[str, object]], destination: str | Path, fieldnames: list[str] | None = None) -> Path:
    """Write structured records to a UTF-8 CSV file."""
    rows = list(rows)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0].keys()) if rows else (fieldnames or [])
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_tfrecord(records: Iterable[tuple[bytes, int]], destination: str | Path) -> Path:
    """Write ``(raw_bytes, integer_label)`` pairs to a TFRecord file."""
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("TensorFlow is required to write TFRecords") from error

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(target)) as writer:
        for content, label in records:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "content": tf.train.Feature(bytes_list=tf.train.BytesList(value=[content])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return target


def create_media_manifest(
    directory: str | Path,
    destination: str | Path,
    extensions: dict[str, set[str]] | None = None,
) -> Path:
    """Create a CSV manifest for images, videos, and audio files."""
    default_extensions = extensions or {
        "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
        "video": {".mp4", ".mov", ".avi", ".mkv"},
        "audio": {".wav", ".mp3", ".flac", ".m4a", ".ogg"},
    }
    root = Path(directory)
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        suffix = path.suffix.lower()
        media_type = next((kind for kind, allowed in default_extensions.items() if suffix in allowed), None)
        if media_type:
            rows.append({"path": str(path.relative_to(root)), "media_type": media_type, "extension": suffix})
    return write_csv(rows, destination, ["path", "media_type", "extension"])
