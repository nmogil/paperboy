# Paperboy

An AI-powered academic paper recommendation system that ranks and analyzes arXiv papers based on user research profiles. It uses OpenAI models to provide personalized digests of relevant research papers with detailed analysis and reasoning.

## Project Structure

```
paperboy/
├── src/                          # Source code
│   ├── api_models.py            # FastAPI request/response models
│   ├── cache.py                 # In-memory cache with TTL (fallback)
│   ├── cache_supabase.py        # Hybrid cache with Supabase persistence
│   ├── circuit_breaker.py       # Circuit breaker pattern for external services
│   ├── config.py                # Centralized configuration via Pydantic BaseSettings
│   ├── content_extractor.py     # Tavily API for full article content extraction
│   ├── digest_service_enhanced.py # Orchestrates the full digest generation workflow
│   ├── fetch_service.py         # Daily content fetching and storage with background processing
│   ├── fetcher_lightweight.py   # arXiv web scraping using httpx
│   ├── graceful_shutdown.py     # Proper handling of shutdown signals
│   ├── llm_client.py            # Direct OpenAI integration for ranking and analysis
│   ├── main.py                  # FastAPI application with background tasks
│   ├── metrics.py               # Performance monitoring and API call tracking
│   ├── models.py                # Pydantic models for type safety
│   ├── news_fetcher.py          # NewsAPI integration with intelligent query generation
│   ├── query_generator.py       # Smart news query generation based on user profiles
│   ├── security.py              # API key authentication middleware
│   └── state_supabase.py        # Supabase-based distributed state management
├── config/                      # Configuration files
│   └── .env                    # Environment variables (create from .env.example)
├── data/                       # Data persistence
│   └── (various state files)   # Local JSON state storage (when Supabase not used)
├── archived/                   # Archived implementations
│   └── agent_implementation/   # Original Pydantic AI agent code
├── pipedream/                  # Webhook integrations
│   └── trigger-generate-digest.js # Batch user processing webhook
├── templates/                  # (Reserved for future HTML templates)
├── Dockerfile                  # Production Docker configuration
├── docker-compose.yaml         # Docker Compose configuration
├── requirements.lightweight.txt # Lightweight dependencies
├── deploy_cloudrun.sh          # Google Cloud Run deployment script
├── cloudbuild.yaml             # Google Cloud Build configuration
├── CLAUDE.md                   # AI assistant instructions
└── PROJECT_ARCH.md             # Detailed architecture documentation
```

## Features

### 🧠 AI-Powered Analysis

- **Intelligent Ranking**: Uses OpenAI models to rank papers and news by relevance to user research profile
- **Mixed Content Sources**: Combines arXiv papers with industry news for comprehensive coverage
- **Detailed Analysis**: Provides in-depth analysis of top content with reasoning and key insights
- **Personalized Recommendations**: Tailors suggestions based on user's research goals and role
- **HTML Digest Generation**: Creates beautifully formatted digests with separate sections for papers and news

### 🚀 Production-Ready API

- **Asynchronous Processing**: Background task execution for long-running digest generation
- **Two-Phase Operation**: Separate source fetching and digest generation for flexibility
- **RESTful Endpoints**: Clean API design with status tracking and health checks
- **API Authentication**: Secure access via API key middleware
- **Webhook Support**: Optional callback URLs for task completion notifications
- **Auto-Documentation**: Swagger UI and ReDoc available at `/docs` and `/redoc`

### 🐳 Cloud-Native Deployment

- **Docker Support**: Lightweight containers with security best practices
- **Google Cloud Run**: One-click deployment with automatic scaling
- **Supabase Integration**: Distributed state management for higher concurrency
- **Circuit Breakers**: Graceful degradation when external services fail
- **Graceful Shutdown**: Proper SIGTERM handling for Cloud Run
- **Health Monitoring**: Built-in health checks and optional Logfire integration

### 📊 Data Pipeline

- **Daily Source Fetching**: Pre-fetch content for faster digest generation
- **arXiv Integration**: Automated fetching and parsing of academic papers
- **News Integration**: NewsAPI with intelligent query generation
- **Content Extraction**: Tavily API for full article content with prioritization
- **State Persistence**: Supabase for distributed state with JSON fallback
- **Caching**: Hybrid two-tier cache system for performance

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
SUPABASE_URL=your_supabase_project_url    # For distributed state
SUPABASE_KEY=your_supabase_anon_key      # For distributed state
```

### 3. Run with Docker

```bash
# Start the service
docker-compose up --build

# Or run in background
docker-compose up -d --build
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

| Variable         | Description                          | Default  | Required |
| ---------------- | ------------------------------------ | -------- | -------- |
| `OPENAI_API_KEY` | OpenAI API access token              | -        | ✅       |
| `API_KEY`        | Authentication key for API endpoints | -        | ✅       |
| `SUPABASE_URL`   | Supabase project URL                 | -        | ✅       |
| `SUPABASE_KEY`   | Supabase anon key                    | -        | ✅       |
| `OPENAI_MODEL`   | OpenAI model to use                  | `gpt-4o` |          |
| `TOP_N_ARTICLES` | Number of papers to analyze          | `5`      |          |
| `TOP_N_NEWS`     | Number of news articles to analyze   | `5`      |          |
| `LOG_LEVEL`      | Logging verbosity                    | `INFO`   |          |

### Performance & Limits

| Variable                     | Description                          | Default |
| ---------------------------- | ------------------------------------ | ------- |
| `HTTP_TIMEOUT`               | HTTP request timeout (seconds)       | `30`    |
| `TASK_TIMEOUT`               | Max digest generation time (seconds) | `300`   |
| `AGENT_RETRIES`              | LLM retry attempts                   | `2`     |
| `ANALYSIS_CONTENT_MAX_CHARS` | Max content per article for analysis | `20000` |
| `RANKING_INPUT_MAX_ARTICLES` | Max articles sent to ranking LLM     | `30`    |
| `SUMMARY_MAX_CONCURRENT`     | Max concurrent summary generation    | `3`     |
| `EXTRACT_MAX_CONCURRENT`     | Max concurrent content extraction    | `3`     |

### Optional Features

| Variable              | Description                       | Default |
| --------------------- | --------------------------------- | ------- |
| `NEWS_ENABLED`        | Enable news fetching              | `true`  |
| `NEWSAPI_KEY`         | NewsAPI key for news              | -       |
| `TAVILY_API_KEY`      | Tavily key for content extraction | -       |
| `NEWS_MAX_ARTICLES`   | Max news articles to fetch        | `50`    |
| `NEWS_MAX_EXTRACT`    | Max articles to extract content   | `10`    |
| `NEWS_CACHE_TTL`      | News cache duration (seconds)     | `3600`  |
| `USE_LIGHTWEIGHT`     | Use httpx instead of Playwright   | `true`  |
| `LOGFIRE_TOKEN`       | Monitoring service token          | -       |

## API Usage

### Authentication

All API endpoints (except health check) require authentication via the `X-API-Key` header:

```bash
curl -H "X-API-Key: your_api_key_from_env" http://localhost:8000/endpoint
```

### Core Workflow

#### Option 1: Direct Digest Generation
1. **Submit a digest request** → Get task ID immediately
2. **Poll for status** → Track progress and get results
3. **Optional webhooks** → Receive notifications when complete

#### Option 2: Two-Phase Operation (Recommended for batch processing)
1. **Fetch sources daily** → Pre-fetch content for a specific date
2. **Generate digests** → Use pre-fetched sources for faster generation
3. **Poll for status** → Track progress and get results

### Fetch Daily Sources (Optional)

**POST** `/fetch-sources`

Pre-fetch sources for a specific date to speed up digest generation.

```bash
curl -X POST http://localhost:8000/fetch-sources \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "source_date": "2025-01-07"
  }'
```

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
      "goals": "Exploring latest developments in transformer architectures and efficient training methods",
      "news_interest": "AI industry trends and breakthrough technologies"
    },
    "source_date": "2025-01-07",  # Use pre-fetched sources
    "top_n_articles": 5,
    "top_n_news": 5,
    "digest_sources": {
      "arxiv": true,
      "news_api": true
    },
    "callback_url": "https://your-app.com/webhooks/digest-complete"
  }'
```

#### Request Parameters

| Parameter        | Type    | Required | Description                                           |
| ---------------- | ------- | -------- | ----------------------------------------------------- |
| `user_info`      | Object  | Yes      | User profile with name, title, goals, news_interest  |
| `source_date`    | String  | No       | Use pre-fetched sources from this date (YYYY-MM-DD)  |
| `target_date`    | String  | No       | Target date for content if fetching (YYYY-MM-DD)     |
| `top_n_articles` | Integer | No       | Number of papers to include (default: 5)             |
| `top_n_news`     | Integer | No       | Number of news articles to include (default: 5)      |
| `digest_sources` | Object  | No       | Control which sources to include in digest           |
| `callback_url`   | String  | No       | Webhook URL for completion notification              |

#### Digest Sources Control

The `digest_sources` parameter allows you to control which content sources are included:

```json
{
  "digest_sources": {
    "arxiv": true, // Include ArXiv research papers
    "news_api": false // Exclude news articles
  }
}
```

- **Default behavior**: If `digest_sources` is not provided, ArXiv is enabled and news follows the global `NEWS_ENABLED` setting
- **Source options**:
  - `arxiv`: Research papers from ArXiv
  - `news_api`: Industry news and articles
- **Examples**:
  - Papers only: `{"arxiv": true, "news_api": false}`
  - News only: `{"arxiv": false, "news_api": true}`
  - Both sources: `{"arxiv": true, "news_api": true}`

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
    "articles_analyzed": 10,
    "source_date": "2025-01-07",
    "digest_type": "mixed",
    "generation_time": "2024-03-20T10:30:00Z"
  }
}
```

### Check Fetch Status

**GET** `/fetch-status/{task_id}`

```bash
curl http://localhost:8000/fetch-status/xyz789 \
  -H "X-API-Key: your_api_key"
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
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key

# Recommended production settings
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
TASK_TIMEOUT=300
USE_LIGHTWEIGHT=true

# Optional news integration
NEWS_ENABLED=true
NEWSAPI_KEY=your-newsapi-key
TAVILY_API_KEY=your-tavily-key

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
- **`digest_service_enhanced.py`**: Orchestrates the full digest generation workflow
- **`llm_client.py`**: Direct OpenAI integration for ranking and analysis
- **`fetcher_lightweight.py`**: arXiv paper fetching with httpx
- **`news_fetcher.py`**: NewsAPI integration with intelligent query generation
- **`content_extractor.py`**: Tavily API for full article content extraction
- **`fetch_service.py`**: Daily content fetching and storage
- **`state_supabase.py`**: Distributed state management
- **`circuit_breaker.py`**: Graceful degradation when services fail
- **`models.py`**: Type-safe Pydantic models throughout
- **`config.py`**: Centralized environment-based configuration

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
ANALYSIS_CONTENT_MAX_CHARS=10000     # Reduce content per article
OPENAI_MODEL=gpt-4o-mini            # Use cheaper model
TOP_N_ARTICLES=3                     # Reduce papers analyzed
TOP_N_NEWS=3                         # Reduce news analyzed

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
