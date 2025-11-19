#!/bin/bash

# Deployment script for cyberbullying model to Google Cloud

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION=${2:-us-central1}
SERVICE_NAME="cyberbullying-predict"
REPO_NAME="cyberbullying-repo"

echo "🚀 Deploying Cyberbullying Model to Google Cloud"
echo "📍 Project: $PROJECT_ID"
echo "🌍 Region: $REGION"

# Enable services
echo "✅ Enabling required services..."
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  --project $PROJECT_ID

# Create Artifact Registry
echo "📦 Creating Artifact Registry..."
gcloud artifacts repositories create $REPO_NAME \
  --repository-format=docker \
  --location=$REGION \
  --project=$PROJECT_ID \
  2>/dev/null || echo "Repository already exists"

# Configure Docker
echo "🐳 Configuring Docker..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push
echo "🔨 Building Docker image..."
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest .

echo "📤 Pushing to Artifact Registry..."
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --project $PROJECT_ID

# Get URL
echo "✨ Deployment complete!"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --format='value(status.url)' \
  --project $PROJECT_ID)

echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "Test the service:"
echo "curl -X GET $SERVICE_URL/health"