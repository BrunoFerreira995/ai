# Backend

## FastAPI local

```bash
MODEL_PATH=artifacts/saved_model .venv/bin/uvicorn backend.app:app --reload
```

Open <http://127.0.0.1:8000/docs>.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"instances": [[0, 0, 0, 0, 0, 0, 0, 0]]}'
```

## Docker and Kubernetes

```bash
docker compose -f backend/docker-compose.yml up --build
kubectl apply -f backend/kubernetes.yaml
```
