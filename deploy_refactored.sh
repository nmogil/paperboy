#!/bin/bash
# Deploy script for the refactored Paperboy API

echo "🚀 Deploying Refactored Paperboy API..."
echo "=================================="

# Check if config/.env exists
if [ ! -f config/.env ]; then
    echo "❌ Error: config/.env file not found!"
    echo "Please ensure your environment variables are set in config/.env"
    exit 1
fi

# Create data directory for refactored version
echo "📁 Creating data directory..."
mkdir -p data-refactored

# Stop any existing refactored container
echo "🛑 Stopping existing refactored container (if any)..."
docker-compose -f docker-compose.refactored.yaml down

# Build the image
echo "🔨 Building Docker image..."
docker-compose -f docker-compose.refactored.yaml build

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

# Start the container
echo "🏃 Starting container..."
docker-compose -f docker-compose.refactored.yaml up -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container!"
    exit 1
fi

# Wait for container to be healthy
echo "⏳ Waiting for API to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -f http://localhost:8001/digest-status/health >/dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    attempt=$((attempt + 1))
    echo "Waiting... (attempt $attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ API failed to start within 60 seconds"
    echo "Check logs with: docker logs paperboy-api-refactored"
    exit 1
fi

# Display status
echo ""
echo "✅ Deployment Complete!"
echo "=================================="
echo "📍 API URL: http://localhost:8001"
echo "📋 Container: paperboy-api-refactored"
echo "📊 View logs: docker logs paperboy-api-refactored -f"
echo "📈 Monitor stats: docker stats paperboy-api-refactored"
echo ""
echo "🧪 Test with your payload:"
echo "curl -X POST http://localhost:8001/generate-digest \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'X-API-Key: YOUR_API_KEY' \\"
echo "  -d @test_payload.json"
echo ""
echo "Or use: python test_api_local.py new"