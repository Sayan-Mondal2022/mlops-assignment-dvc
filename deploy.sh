#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="mlops-assignment-resources"
LOCATION="centralindia"
ACR_NAME="sayanacrmlops"
IMAGE_NAME="cpu-predictor"
IMAGE_TAG="latest"
CONTAINER_APP_NAME="cpu-predictor-app"
ENV_NAME="cpu-env"
TARGET_PORT=8501
DOCKERFILE_PATH="app/Dockerfile"
BUILD_CONTEXT="."

echo "== Starting Deployment =="

if ! az account show >/dev/null 2>&1; then
  az login --use-device-code
fi

az group show -n "$RESOURCE_GROUP" >/dev/null

az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" >/dev/null
ACR_SERVER=$(az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" --query loginServer -o tsv)

az acr update -n "$ACR_NAME" --admin-enabled true >/dev/null
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query "username" -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)

echo "Building Docker image..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" -f "$DOCKERFILE_PATH" "$BUILD_CONTEXT"
FULL_IMAGE="$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$FULL_IMAGE"

echo "Pushing image to ACR..."
echo "$ACR_PASS" | docker login "$ACR_SERVER" -u "$ACR_USER" --password-stdin
docker push "$FULL_IMAGE"

if ! az containerapp env show -g "$RESOURCE_GROUP" -n "$ENV_NAME" >/dev/null 2>&1; then
  az containerapp env create \
    --name "$ENV_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION"
fi

if az containerapp show --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp secret set \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --secrets acr-pwd="$ACR_PASS" >/dev/null

  az containerapp update \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --image "$FULL_IMAGE" \
    --ingress external \
    --target-port "$TARGET_PORT" \
    --registry-server "$ACR_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS"
else
  az containerapp create \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ENV_NAME" \
    --image "$FULL_IMAGE" \
    --ingress external \
    --target-port "$TARGET_PORT" \
    --registry-server "$ACR_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS"
fi

echo "Restarting revision..."
az containerapp revision restart \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" >/dev/null || true

URL=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  -o tsv)

echo "== Deployment Complete =="
echo "URL: https://$URL"