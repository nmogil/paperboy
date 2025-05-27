#!/bin/bash

# deploy_cloudrun.sh - Deploy Paperboy to Google Cloud Run

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
SERVICE_NAME="${SERVICE_NAME:-paperboy}"
REGION="${REGION:-us-central1}"
IMAGE_NAME="${IMAGE_NAME:-paperboy-lightweight}"
MEMORY="${MEMORY:-512Mi}"  # Reduced from 1Gi - sufficient for lightweight version
CPU="${CPU:-1}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-50}"  # Need 50 instances for 50 concurrent requests
CONCURRENCY="${CONCURRENCY:-1}"       # MUST be 1 due to in-memory state

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Deploying Paperboy to Cloud Run${NC}"

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: GCP_PROJECT_ID environment variable is not set${NC}"
    echo "Please set it with: export GCP_PROJECT_ID=your-project-id"
    exit 1
fi

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI is not installed${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${RED}❌ Error: Not authenticated with gcloud${NC}"
    echo "Please run: gcloud auth login"
    exit 1
fi

# Set the project
echo -e "${YELLOW}📋 Setting project to: $PROJECT_ID${NC}"
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com

# Check if .env file exists
if [ ! -f "config/.env" ]; then
    echo -e "${RED}❌ Error: config/.env file not found${NC}"
    echo "Please create config/.env with required environment variables"
    exit 1
fi

# Create secrets in Secret Manager (if not exists)
echo -e "${YELLOW}🔐 Creating secrets in Secret Manager...${NC}"

# Function to create or update secret
create_or_update_secret() {
    local secret_name=$1
    local secret_value=$2
    
    if gcloud secrets describe "$secret_name" --project="$PROJECT_ID" &> /dev/null; then
        echo "Secret $secret_name already exists, updating..."
        echo -n "$secret_value" | gcloud secrets versions add "$secret_name" --data-file=-
    else
        echo "Creating secret $secret_name..."
        echo -n "$secret_value" | gcloud secrets create "$secret_name" --data-file=- --replication-policy="automatic"
    fi
}

# Read .env file and create secrets
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    # Remove quotes from value
    value="${value%\"}"
    value="${value#\"}"
    create_or_update_secret "$key" "$value"
done < config/.env

# Build the container image
echo -e "${YELLOW}🏗️  Building container image...${NC}"
gcloud builds submit \
    --tag "gcr.io/$PROJECT_ID/$IMAGE_NAME" \
    --timeout=20m

# Deploy to Cloud Run
echo -e "${YELLOW}🚀 Deploying to Cloud Run...${NC}"

# Prepare environment variables from secrets
ENV_VARS=""
while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    ENV_VARS="${ENV_VARS}${key}=projects/$PROJECT_ID/secrets/$key:latest,"
done < config/.env

# Add lightweight mode flag
ENV_VARS="${ENV_VARS}USE_LIGHTWEIGHT=true,"
ENV_VARS="${ENV_VARS}PORT=8080"

# Deploy the service
gcloud run deploy "$SERVICE_NAME" \
    --image "gcr.io/$PROJECT_ID/$IMAGE_NAME" \
    --platform managed \
    --region "$REGION" \
    --memory "$MEMORY" \
    --cpu "$CPU" \
    --min-instances "$MIN_INSTANCES" \
    --max-instances "$MAX_INSTANCES" \
    --concurrency "$CONCURRENCY" \
    --timeout 3600 \
    --no-allow-unauthenticated \
    --set-secrets="$ENV_VARS" \
    --service-account "$SERVICE_NAME@$PROJECT_ID.iam.gserviceaccount.com" 2>/dev/null || {
        echo -e "${YELLOW}Service account doesn't exist, deploying without it...${NC}"
        gcloud run deploy "$SERVICE_NAME" \
            --image "gcr.io/$PROJECT_ID/$IMAGE_NAME" \
            --platform managed \
            --region "$REGION" \
            --memory "$MEMORY" \
            --cpu "$CPU" \
            --min-instances "$MIN_INSTANCES" \
            --max-instances "$MAX_INSTANCES" \
            --concurrency "$CONCURRENCY" \
            --timeout 3600 \
            --no-allow-unauthenticated \
            --set-secrets="$ENV_VARS"
    }

# Get the service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --platform managed \
    --region "$REGION" \
    --format 'value(status.url)')

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}Service URL: $SERVICE_URL${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "1. To allow public access: gcloud run services add-iam-policy-binding $SERVICE_NAME --member=\"allUsers\" --role=\"roles/run.invoker\" --region=$REGION"
echo "2. To test the API: curl -H \"X-API-Key: YOUR_API_KEY\" $SERVICE_URL/docs"
echo "3. To view logs: gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME\" --limit 50"