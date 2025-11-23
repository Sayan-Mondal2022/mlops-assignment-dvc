#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG ==========
RESOURCE_GROUP="mlops-assignment-resources"
CONTAINER_APP_NAME="cpu-predictor-new"
ACR_NAME="sayanacrmlops"
IMAGE_NAME="cpu-predictor"
IMAGE_TAG="latest"
DOCKERFILE_PATH="api/Dockerfile"

# ========= BUILD ==========
echo "🚀 Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f ${DOCKERFILE_PATH} .

# ========= TAG ============
echo "🏷  Tagging image for ACR..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}

# ========= LOGIN ===========
echo "🔐 Logging into ACR..."
az acr login --name ${ACR_NAME}

# ========= PUSH ============
echo "📤 Pushing image to ACR..."
docker push ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}

# ========= UPDATE APP =======
echo "🔄 Updating Container App to use new image..."
az containerapp update \
  --name ${CONTAINER_APP_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --image "${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

# ========= RESTART REVISION =======
echo "♻ Restarting active revision..."
az containerapp revision restart \
  --name ${CONTAINER_APP_NAME} \
  --resource-group ${RESOURCE_GROUP}

# ========= SHOW INFO =============
echo "🌍 Fetching Public URL..."
az containerapp show \
  --name ${CONTAINER_APP_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --query properties.configuration.ingress.fqdn \
  -o tsv
