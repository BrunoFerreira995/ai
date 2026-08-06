# Phase 18 — Security

Implemented controls:

- Hashed API keys with role-based authorization
- Sliding-window rate limiting
- Fernet authenticated encryption for model/artifact files
- Input validation against NaN, infinity, and unsafe ranges
- FGSM adversarial-example generation for robustness tests
- Optional API key enforcement and rate limiting in FastAPI

Enable API protection when starting the backend:

```bash
API_KEY=change-me MODEL_PATH=artifacts/saved_model .venv/bin/uvicorn backend.app:app
```

Generate a key for file encryption with:

```bash
.venv/bin/python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```
