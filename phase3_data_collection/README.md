# Phase 3 — Data Collection

This module provides a small, reusable foundation for collecting and storing
dataset inputs.

## Capabilities

- Download public dataset or media files with optional SHA-256 verification.
- Scrape absolute links from an HTML page, optionally filtering by extension.
- Fetch JSON from APIs with query parameters and custom headers.
- Create and append manual annotation records in CSV format.
- Write structured CSV data and TensorFlow `TFRecord` examples.
- Build a manifest for image, video, and audio files.

Run the tests from the project root:

```bash
.venv/bin/python -m unittest discover -s phase3_data_collection -p 'test_*.py'
```

Example:

```python
from phase3_data_collection import AnnotationRecord, AnnotationWriter, create_media_manifest

labels = AnnotationWriter("data/annotations.csv")
labels.append(AnnotationRecord("data/images/cat.jpg", "cat", "ana"))
create_media_manifest("data", "data/manifest.csv")
```
