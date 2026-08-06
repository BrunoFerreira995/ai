# Cloud deployment

As configurações abaixo usam a imagem Docker do backend:

```text
ghcr.io/SEU_USUARIO/ai-model-api:latest
```

Substitua o nome da imagem antes do deploy e publique-a no registry do
provedor escolhido.

## AWS ECS/Fargate

Requisitos: ECR, ECS cluster e uma VPC com subnets públicas/privadas.

```bash
aws ecs register-task-definition --cli-input-json file://cloud/aws/ecs-task-definition.json
```

Depois crie o service ECS apontando para a task definition e configure o
load balancer/target group na porta 8000.

## Google Cloud Run

```bash
gcloud run services replace cloud/gcp/cloudrun.yaml --region us-central1
```

## Azure Container Apps

```bash
az containerapp create \
  --resource-group SEU_RESOURCE_GROUP \
  --name ai-model-api \
  --environment SEU_CONTAINER_APP_ENV \
  --yaml cloud/azure/container-app.yaml
```

Essas configurações são templates de infraestrutura; autenticação, domínio,
secrets, registry e monitoramento devem ser configurados para cada ambiente.
