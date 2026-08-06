"""FastAPI service for the exported TensorFlow SavedModel."""

from __future__ import annotations

import json
import hmac
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from security.controls import RateLimiter, validate_input

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/saved_model"))
CLASSES_PATH = MODEL_PATH.parent / "classes.json"


class PredictionRequest(BaseModel):
    instances: list[list[float]] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]]
    class_names: list[str] | None = None


app = FastAPI(title="AI Model API", version="1.0.0")
_model = None
_class_names: list[str] | None = None
_rate_limiter = RateLimiter(limit=int(os.getenv("RATE_LIMIT", "60")), window_seconds=60)


def authorize_request(api_key: str | None = Header(default=None), x_client_id: str = Header(default="anonymous")) -> None:
    """Optionally enforce API key and always apply a request rate limit."""
    if not _rate_limiter.allow(x_client_id):
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    expected_key = os.getenv("API_KEY")
    if expected_key and not api_key or expected_key and not hmac.compare_digest(api_key or "", expected_key):
        raise HTTPException(status_code=401, detail="invalid API key")


def get_model():
    global _model, _class_names
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
        if CLASSES_PATH.exists():
            _class_names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_model()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, _: None = Depends(authorize_request)) -> PredictionResponse:
    try:
        model = get_model()
        values = validate_input(np.asarray(request.instances, dtype=np.float32))
        expected_features = model.input_shape[-1]
        if values.ndim != 2 or values.shape[1] != expected_features:
            raise ValueError(f"expected instances shaped (n, {expected_features})")
        probabilities = model.predict(values, verbose=0)
        predictions = np.argmax(probabilities, axis=-1).astype(int).tolist()
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities.tolist(),
            class_names=[_class_names[index] for index in predictions] if _class_names else None,
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
