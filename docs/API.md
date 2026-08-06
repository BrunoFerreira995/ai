# API

A API FastAPI está em `backend/app.py`.

## Iniciar

```bash
MODEL_PATH=artifacts/saved_model .venv/bin/uvicorn backend.app:app --reload
```

Documentação interativa: <http://127.0.0.1:8000/docs>

## Health check

```bash
curl http://127.0.0.1:8000/health
```

## Previsão

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[0, 0, 0, 0, 0, 0, 0, 0]]}'
```

Quando `API_KEY` estiver configurada, envie também:

```bash
-H 'X-API-Key: change-me'
```
