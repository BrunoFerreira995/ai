# Deployment

## Local

```bash
MODEL_PATH=artifacts/saved_model .venv/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker compose -f backend/docker-compose.yml up --build
```

## Kubernetes

```bash
kubectl apply -f backend/kubernetes.yaml
```

## Cloud

Templates para AWS, Google Cloud e Azure estão em `cloud/`.
Antes do deploy, substitua imagem, registry, IDs de conta, secrets e recursos
de rede pelos valores do ambiente.
