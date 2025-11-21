#!/usr/bin/env bash
set -euo pipefail

#############################################
# CONFIG
#############################################
RESOURCE_GROUP="mlops-assignment-resources"
LOCATION="centralindia"
ACR_NAME="sayanacrmlops"
IMAGE_NAME="cpu-predictor"
IMAGE_TAG="latest"
CONTAINER_APP_NAME="cpu-predictor-app"
ENV_NAME="cpu-env"
TARGET_PORT=5001               # IMPORTANT: matches Flask app.py
DOCKERFILE_PATH="app/Dockerfile"   # Your Dockerfile is in repo root
BUILD_CONTEXT="."

#############################################
echo "============================================"
echo "🚀 Azure Deployment Started"
echo " Resource Group: $RESOURCE_GROUP"
echo " ACR: $ACR_NAME"
echo " App: $CONTAINER_APP_NAME"
echo " Port: $TARGET_PORT"
echo "============================================"

#############################################
# LOGIN CHECK
#############################################
if ! az account show >/dev/null 2>&1; then
  echo "🔐 Logging into Azure…"
  az login --use-device-code
fi

#############################################
# VERIFY RESOURCE GROUP
#############################################
if ! az group show -n "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "❌ ERROR: Resource group '$RESOURCE_GROUP' does not exist."
  exit 1
fi

#############################################
# VERIFY ACR EXISTS
#############################################
if ! az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "❌ ERROR: ACR '$ACR_NAME' not found."
  exit 1
fi

ACR_SERVER=$(az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" --query loginServer -o tsv)

#############################################
# ENABLE ACR ADMIN LOGIN
#############################################
echo "🔧 Enabling ACR admin user…"
az acr update -n "$ACR_NAME" --admin-enabled true >/dev/null

ACR_USER=$(az acr credential show -n "$ACR_NAME" --query "username" -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)

#############################################
# BUILD DOCKER IMAGE
#############################################
echo "🐳 Building Docker image..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" -f "$DOCKERFILE_PATH" "$BUILD_CONTEXT"

FULL_IMAGE="$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$FULL_IMAGE"

#############################################
# PUSH TO ACR
#############################################
echo "📤 Pushing image to ACR..."
echo "$ACR_PASS" | docker login "$ACR_SERVER" -u "$ACR_USER" --password-stdin
docker push "$FULL_IMAGE"

#############################################
# ENSURE CONTAINER APPS ENVIRONMENT EXISTS
#############################################
if ! az containerapp env show -g "$RESOURCE_GROUP" -n "$ENV_NAME" >/dev/null 2>&1; then
  echo "🌍 Creating Container App environment..."
  az containerapp env create \
    --name "$ENV_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION"
else
  echo "🌍 Using existing Container App environment: $ENV_NAME"
fi

#############################################
# CREATE OR UPDATE CONTAINER APP
#############################################
if az containerapp show --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "♻ Updating existing Container App..."

  # Update ACR secret
  az containerapp secret set \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --secrets acr-pwd="$ACR_PASS" >/dev/null

  # Update without full recreate
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
  echo "✨ Creating new Container App…"

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

#############################################
# OPTIONAL: Restart container app (Azure caching fix)
#############################################
echo "🔄 Restarting Container App to apply changes..."
az containerapp revision restart \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" >/dev/null || true


#############################################
# GET PUBLIC URL
#############################################
URL=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  -o tsv)

echo "============================================"
echo "🎉 Deployment Completed!"
echo "🌐 App URL: https://$URL"
echo "============================================"
