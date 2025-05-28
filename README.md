# Paperboy

An AI-powered academic paper recommendation system that ranks and analyzes arXiv papers based on user research profiles. It uses OpenAI models to provide personalized digests of relevant research papers with detailed analysis and reasoning.

## Project Structure

```
paperboy/
├── src/                          # Source code
│   ├── agent.py                 # Core AI logic using Pydantic AI
│   ├── agent_prompts.py         # Structured LLM prompts for JSON output
│   ├── agent_tools_lightweight.py # Article content extraction (httpx-based)
│   ├── api_models.py            # FastAPI request/response models
│   ├── config.py                # Centralized configuration via Pydantic BaseSettings
│   ├── fetcher_lightweight.py   # arXiv web scraping using httpx
│   ├── main.py                  # FastAPI application with background tasks
│   ├── models.py                # Pydantic models for type safety
│   ├── security.py              # API key authentication middleware
│   └── state.py                 # JSON-based state persistence
├── config/                      # Configuration files
│   ├── settings.py             # Additional configuration settings
│   └── .env                    # Environment variables (create from .env.example)
├── data/                       # Data persistence
│   ├── agent_state.json        # Agent state storage
│   ├── test_state.json         # Test data
│   └── arxiv_cs_submissions_2025-04-01.json # Sample arXiv data
├── archived_full_implementation/ # Archived Playwright-based implementation
├── Dockerfile                  # Production Docker configuration
├── docker-compose.lightweight.yaml # Lightweight Docker Compose
├── requirements.lightweight.txt # Lightweight dependencies (httpx)
├── deploy_cloudrun.sh          # Google Cloud Run deployment script
├── cloudbuild.yaml             # Google Cloud Build configuration
└── CLAUDE.md                   # AI assistant instructions
```

## Features

### 🧠 AI-Powered Analysis
- **Intelligent Ranking**: Uses OpenAI models (GPT-4 by default) to rank papers by relevance to user research profile
- **Detailed Analysis**: Provides in-depth analysis of top papers with reasoning and key insights
- **Personalized Recommendations**: Tailors suggestions based on user's research goals and academic background
- **HTML Digest Generation**: Creates formatted research digests with structured analysis

### 🚀 Production-Ready API
- **Asynchronous Processing**: Background task execution for long-running digest generation
- **RESTful Endpoints**: Clean API design with status tracking and health checks
- **API Authentication**: Secure access via API key middleware
- **Webhook Support**: Optional callback URLs for task completion notifications
- **Auto-Documentation**: Swagger UI and ReDoc available at `/docs` and `/redoc`

### 🐳 Cloud-Native Deployment
- **Docker Support**: Lightweight containers with security best practices
- **Google Cloud Run**: One-click deployment with included configuration
- **Auto-Scaling**: Handles variable workloads efficiently
- **Health Monitoring**: Built-in health checks and optional Logfire integration
- **Resource Limits**: Optimized for cloud environments with proper resource constraints

### 📊 Data Pipeline
- **arXiv Integration**: Automated fetching and parsing of academic papers
- **Web Scraping**: Robust content extraction using httpx (lightweight) or Playwright (full)
- **Content Processing**: Intelligent truncation and formatting for LLM analysis
- **State Persistence**: JSON-based storage for task tracking and results

### 🛠 Developer Experience
- **Type Safety**: Comprehensive Pydantic models throughout the codebase
- **Modern Python**: Async/await patterns, dependency injection, and clean architecture
- **Easy Configuration**: Environment-based settings with sensible defaults
- **Testing**: Mocked LLM calls for reliable testing
- **Documentation**: Detailed setup instructions and API documentation

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key
- (Optional) Google Cloud account for deployment

### 1. Clone and Setup

```bash
git clone https://github.com/your-username/paperboy.git
cd paperboy
```

### 2. Configure Environment

Create your environment file:
```bash
# Copy the example configuration
cp config/.env.example config/.env

# Edit with your settings (required: OPENAI_API_KEY, API_KEY)
nano config/.env
```

Required environment variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
API_KEY=your_secure_api_key_for_authentication
```

### 3. Run with Docker

```bash
# Start the service
docker-compose up --build

# Or run in background
docker-compose -f docker-compose.lightweight.yaml up -d --build
```

### 4. Access the API

- **API Endpoints**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs  
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/digest-status/health

### Local Development (Alternative)

If you prefer running without Docker:

```bash
# Install dependencies
pip install -r requirements.lightweight.txt

# Run the server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

All configuration is managed via environment variables in `config/.env`. The system uses Pydantic BaseSettings for type-safe configuration loading.

### Core Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API access token | - | ✅ |
| `API_KEY` | Authentication key for API endpoints | - | ✅ |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4o` | |
| `TOP_N_ARTICLES` | Number of articles to analyze | `5` | |
| `LOG_LEVEL` | Logging verbosity | `INFO` | |

### Performance & Limits

| Variable | Description | Default |
|----------|-------------|---------|
| `CRAWLER_TIMEOUT` | Web scraping timeout (ms) | `25000` |
| `HTTP_TIMEOUT` | HTTP request timeout (seconds) | `30` |
| `TASK_TIMEOUT` | Max digest generation time (seconds) | `300` |
| `AGENT_RETRIES` | LLM retry attempts | `2` |
| `ANALYSIS_CONTENT_MAX_CHARS` | Max content per article for analysis | `8000` |
| `RANKING_INPUT_MAX_ARTICLES` | Max articles sent to ranking LLM | `20` |

### Optional Features

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_LIGHTWEIGHT` | Use httpx instead of Playwright | `true` |
| `LOGFIRE_TOKEN` | Monitoring service token | - |

## API Usage

### Authentication

All API endpoints (except health check) require authentication via the `X-API-Key` header:

```bash
curl -H "X-API-Key: your_api_key_from_env" http://localhost:8000/endpoint
```

### Core Workflow

1. **Submit a digest request** → Get task ID immediately
2. **Poll for status** → Track progress and get results  
3. **Optional webhooks** → Receive notifications when complete

### Generate Research Digest

**POST** `/generate-digest`

Creates a personalized research digest based on user profile and preferences.

```bash
curl -X POST http://localhost:8000/generate-digest \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_info": {
      "name": "Dr. Jane Smith",
      "title": "Machine Learning Researcher", 
      "goals": "Exploring latest developments in transformer architectures and efficient training methods"
    },
    "target_date": "2025-05-01",
    "top_n_articles": 5,
    "callback_url": "https://your-app.com/webhooks/digest-complete"
  }'
```

**Response:**
```json
{
  "task_id": "abc123",
  "status": "pending",
  "message": "Digest generation started"
}
```

### Check Status

**GET** `/digest-status/{task_id}`

```bash
curl http://localhost:8000/digest-status/abc123 \
  -H "X-API-Key: your_api_key"
```

**Response (Completed):**
```json
{
  "task_id": "abc123",
  "status": "completed", 
  "result": {
    "digest_html": "<html>...</html>",
    "articles_analyzed": 5,
    "generation_time": "2024-03-20T10:30:00Z"
  }
}
```

### Health Check

**GET** `/digest-status/health` (no authentication required)

```bash
curl http://localhost:8000/digest-status/health
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Deployment

### Google Cloud Run (Recommended)

Deploy to Google Cloud Run with the included script:

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT=your-project-id

# Deploy (creates necessary resources)
./deploy_cloudrun.sh

# Or use Cloud Build for CI/CD
gcloud builds submit
```

The deployment includes:
- Automatic scaling (0-1000 instances)
- Health checks and monitoring
- Secure environment variable management
- Production-optimized container

### Docker Commands

```bash
# Development
docker-compose up --build              # Start with rebuild
docker-compose up -d                   # Run in background
docker-compose logs -f                 # View logs

# Production
docker build -t paperboy:latest .      # Build production image
docker run -p 8000:8000 --env-file config/.env paperboy:latest

# Maintenance
docker-compose down --volumes --remove-orphans  # Clean shutdown
docker system prune                             # Clean up resources
```

### Environment Variables for Production

For production deployments, ensure these variables are properly set:

```bash
# Required
OPENAI_API_KEY=sk-...
API_KEY=your-secure-random-key

# Recommended production settings
OPENAI_MODEL=gpt-4o
LOG_LEVEL=INFO
TASK_TIMEOUT=300
USE_LIGHTWEIGHT=true

# Optional monitoring
LOGFIRE_TOKEN=your-logfire-token
```

## Development

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_agent.py::test_rank_articles

# Integration test
pytest tests/integration_test.py
```

### Architecture

The system follows a modular design:

- **`main.py`**: FastAPI app with background task handling
- **`agent.py`**: Core AI logic using Pydantic AI for ranking/analysis  
- **`fetcher_lightweight.py`**: arXiv paper fetching with httpx
- **`agent_tools_lightweight.py`**: Content extraction and processing
- **`models.py`**: Type-safe Pydantic models throughout
- **`config.py`**: Centralized environment-based configuration
- **`state.py`**: JSON-based persistence for task tracking

### Key Patterns

- **Async/Await**: All I/O operations are async for performance
- **Type Safety**: Comprehensive Pydantic models for validation
- **Dependency Injection**: Configuration via BaseSettings
- **Background Tasks**: Long operations handled asynchronously
- **Security**: API key middleware on all protected endpoints

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following the existing patterns
4. Add tests for new functionality
5. Update configuration in `src/config.py` if needed
6. Submit a pull request

### Code Style

- **Type Hints**: Required for all functions and class methods
- **Async/Await**: Use async patterns for I/O operations
- **Pydantic Models**: Create models for all data structures
- **Error Handling**: Comprehensive exception handling with logging
- **Documentation**: Clear docstrings for public APIs

## Troubleshooting

### Common Issues

#### Startup Problems
```bash
# ValidationError on startup
# ✅ Ensure config/.env exists with required variables
cp config/.env.example config/.env
nano config/.env  # Add OPENAI_API_KEY and API_KEY

# ❌ Missing OpenAI API key
export OPENAI_API_KEY=sk-your-key-here
```

#### Docker Issues  
```bash
# Container won't start
docker-compose logs -f                    # Check logs

# Port conflicts
# Edit docker-compose.yaml: "8001:8000"   # Use different port

# Permission issues (Linux)
sudo chown -R $USER:$USER .

# Clean up
docker-compose down --volumes --remove-orphans
docker system prune
```

#### API Authentication
```bash
# 401 Unauthorized
# ✅ Verify X-API-Key header matches config/.env API_KEY

curl -H "X-API-Key: your_api_key" http://localhost:8000/digest-status/health
```

#### Performance Issues
```bash
# Out of memory or context limits
# ✅ Adjust these in config/.env:
RANKING_INPUT_MAX_ARTICLES=10        # Reduce articles sent to LLM
ANALYSIS_CONTENT_MAX_CHARS=4000      # Reduce content per article
OPENAI_MODEL=gpt-3.5-turbo          # Use cheaper model

# Timeout issues
TASK_TIMEOUT=600                     # Increase timeout
HTTP_TIMEOUT=60                      # Increase HTTP timeout
```

#### Production Deployment
```bash
# Cloud Run deployment fails
gcloud auth configure-docker
gcloud config set project your-project-id

# Environment variables not loading
# ✅ Use Google Cloud Console to set env vars
# Or update deploy_cloudrun.sh with proper --set-env-vars
```

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-username/paperboy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/paperboy/discussions)
- **Documentation**: See `/docs` endpoints or `CLAUDE.md`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI](https://openai.com/) for GPT API access
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Pydantic](https://pydantic.dev/) for data validation and settings
- [arXiv](https://arxiv.org/) for providing open access to research papers
