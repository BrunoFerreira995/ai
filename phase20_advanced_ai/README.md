# Phase 20 — Advanced AI

Implemented baselines and adapters for:

- Image classification
- Object detection through OpenCV DNN
- Instance segmentation through supplied model callbacks
- OCR through optional Tesseract
- Pose estimation through supplied model callbacks
- Face recognition through embedding similarity
- Video frame sampling for video understanding
- Portuguese text classification
- Dictionary translation baseline

Detection, segmentation, pose, OCR, and face recognition require compatible
weights or external runtimes for production quality. The adapters keep
preprocessing and inference interfaces consistent.

Run the tests:

```bash
.venv/bin/python -m unittest discover -s phase20_advanced_ai -p 'test_*.py'
```
