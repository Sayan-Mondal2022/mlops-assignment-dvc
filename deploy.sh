#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG ==========

SUBSCRIPTION_ID="YOUR_SUBSCRIPTION_ID_HERE"   # <--- ADD THIS
RESOURCE_GROUP="mlops-assignment-resources"
CONTAINER_APP_NAME="cpu-predictor-new"
ACR_NAME="sayanacrmlops"
IMAGE_NAME="cpu-predictor"
IMAGE_TAG="latest"
DOCKERFILE_PATH="api/Dockerfile"

# ========= AZ LOGIN ==========
echo "🔐 Logging into Azure..."
az login --use-device-code

echo "📌 Setting subscription..."
az account set --subscription "$SUBSCRIPTION_ID"

# ========= ACR PERMISSIONS ==========
echo "🔧 Ensuring Container App can pull from ACR..."
az acr update -n $ACR_NAME --admin-enabled true

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

# ========= GET LATEST REVISION ==========
echo "🔍 Fetching latest revision name..."
REVISION_NAME=$(az containerapp revision list \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query "[?active==\`true\`].name" -o tsv)

echo "📌 Latest Active Revision: $REVISION_NAME"

# ========= RESTART REVISION ==============
echo "♻ Restarting the latest revision..."
az containerapp revision restart \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --revision $REVISION_NAME

# ========= GET URL ==========
echo "🌍 App URL:"
az containerapp show \
  --name ${CONTAINER_APP_NAME} \
  --resource-group ${RESOURCE_GROUP} \
  --query properties.configuration.ingress.fqdn \
  -o tsv

echo "✅ Deployment Completed Successfully!"
