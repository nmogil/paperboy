# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paperboy is an AI-powered academic paper recommendation system that ranks and analyzes arXiv papers and news articles based on user research profiles. It uses OpenAI models to provide personalized digests of relevant research papers and industry news. The system includes enhanced reliability features including circuit breakers, Supabase state management, graceful shutdown, and comprehensive error handling.

**Key Features:**
- **Mixed Content Sources**: ArXiv papers + NewsAPI articles ranked together
- **Supabase Integration**: External state management and distributed caching
- **Circuit Breakers**: Graceful degradation when external services fail  
- **Graceful Shutdown**: Proper SIGTERM handling for Cloud Run
- **Enhanced Error Handling**: Comprehensive fallback strategies
- **Distributed State**: Enables higher concurrency in Cloud Run

## Key Commands

### Running the Service

```bash
# Development with Docker Compose (includes default Supabase credentials)
docker-compose up --build

# Production deployment
docker build -t paperboy:latest .
docker run -p 8000:8000 --env-file config/.env paperboy:latest

# Cloud Run deployment (with Supabase and Secret Manager support)
export SUPABASE_URL=your-supabase-url
export SUPABASE_KEY=your-supabase-key
./deploy_cloudrun.sh

# Cloud Build deployment (automatic with Artifact Registry)
gcloud builds submit
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest --cov=src

# Run integration tests (Note: uses old agent architecture from archived/)
pytest tests/integration_test.py

# Run with verbose output
pytest -v tests/test_agent.py::test_rank_articles
```

### Development Setup

```bash
# Install dependencies
pip install -r requirements.lightweight.txt

# Create virtual environment (recommended, see CONTRIBUTING.md)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run locally without Docker (default port 8000)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# View API documentation
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### Linting and Type Checking

```bash
# Currently no linting/type checking configured
# Consider adding ruff and mypy in the future
```

## Architecture Overview

For detailed architecture with Mermaid diagrams, see PROJECT_ARCH.md. For contribution guidelines, see CONTRIBUTING.md.

The codebase follows a modular architecture with clear separation of concerns:

1. **API Layer** (`main.py`): FastAPI application handling HTTP endpoints and background tasks
2. **Digest Service** (`digest_service_enhanced.py`): Orchestrates the full digest generation workflow
3. **LLM Client** (`llm_client.py`): Direct OpenAI integration for ranking and analysis
4. **Data Fetching** (`fetcher_lightweight.py`): Web scraping of arXiv using httpx
5. **News Integration** (`news_fetcher.py`): NewsAPI integration with intelligent query generation
6. **Content Extraction** (`content_extractor.py`): Tavily API for full article content extraction
7. **Query Generation** (`query_generator.py`): Smart news query generation based on user profiles
8. **Models** (`models.py`, `api_models.py`): Pydantic models for type safety across the system
9. **Configuration** (`config.py`): Centralized settings management via Pydantic BaseSettings
10. **State Management**: 
    - `state.py`: In-memory task state persistence (fallback)
    - `state_supabase.py`: Supabase-based distributed state management
11. **Security** (`security.py`): API key authentication middleware
12. **Caching**:
    - `cache.py`: In-memory cache with TTL (fallback)
    - `cache_supabase.py`: Hybrid cache with Supabase persistence
13. **Metrics** (`metrics.py`): Performance monitoring and API call tracking
14. **Reliability**:
    - `circuit_breaker.py`: Circuit breaker pattern for external services
    - `graceful_shutdown.py`: Proper handling of shutdown signals

### Key Design Patterns

- **Async/Await**: All I/O operations are async for scalability
- **Background Tasks**: Long-running operations handled via FastAPI BackgroundTasks
- **Type Safety**: Extensive use of Pydantic models for validation
- **Dependency Injection**: Configuration injected via Pydantic BaseSettings
- **Security**: API key authentication middleware on all endpoints
- **Error Handling**: Graceful degradation when external services fail
- **Rate Limiting**: Semaphore-based concurrency control for API calls
- **Caching**: Reduces redundant API calls with TTL-based cache

### Critical Dependencies

- `openai`: Direct GPT model integration (replaced pydantic-ai)
- `httpx`: Async HTTP client for web scraping (lightweight alternative to Playwright)
- `beautifulsoup4`: HTML parsing for content extraction
- `fastapi`: Modern async web framework
- `logfire`: Production monitoring and observability
- `tenacity`: Retry logic for external API calls
- `supabase`: External state management and caching

### Environment Configuration

Required environment variables in `config/.env`:

**Core Configuration:**
- `OPENAI_API_KEY`: OpenAI API access (required)
- `API_KEY`: Authentication for API endpoints (required)
- `OPENAI_MODEL`: Model selection (default: gpt-4.1-mini-2025-04-14 per .env.example)
- `TOP_N_ARTICLES`: Number of articles to analyze (default: 5)
- `LOG_LEVEL`: Logging level (default: INFO)

**Performance Settings:**
- `CRAWLER_TIMEOUT`: Web crawler timeout in ms (default: 25000)
- `HTTP_TIMEOUT`: HTTP request timeout in seconds (default: 30)
- `TASK_TIMEOUT`: Max digest generation time in seconds (default: 300)
- `AGENT_RETRIES`: LLM retry attempts (default: 2)
- `ANALYSIS_CONTENT_MAX_CHARS`: Max content for analysis (default: 20000 per .env.example)
- `RANKING_INPUT_MAX_ARTICLES`: Max articles for ranking (default: 30 per .env.example)

**News Integration:**
- `NEWSAPI_KEY`: NewsAPI key for news fetching (optional)
- `TAVILY_API_KEY`: Tavily API key for content extraction (optional)
- `NEWS_ENABLED`: Enable news fetching (default: true)
- `NEWS_MAX_ARTICLES`: Max news articles to fetch (default: 50)
- `NEWS_MAX_EXTRACT`: Max articles to extract full content (default: 10)
- `NEWS_CACHE_TTL`: Cache duration in seconds (default: 3600)

**Supabase Integration:**
- `SUPABASE_URL`: Supabase project URL (optional, for distributed state)
- `SUPABASE_KEY`: Supabase anon key (optional, for distributed state)
- `USE_SUPABASE`: Enable Supabase for state management (default: false)

**Deployment:**
- `USE_LIGHTWEIGHT`: Use httpx instead of Playwright (default: true)
- `LOGFIRE_TOKEN`: Monitoring service token (optional)

### Testing Approach

Tests use pytest with comprehensive mocking:

- Mock `openai.AsyncOpenAI` for deterministic testing
- Mock NewsAPI and Tavily responses for integration tests
- Test both ranking and analysis workflows
- Validate model attributes and JSON parsing
- Test error handling and graceful degradation
- `conftest.py` provides session-scoped fixtures and automatic env setup

### Docker Security

Production containers implement:

- Non-root user (UID 10001)
- Read-only filesystem with tmpfs mounts for `/tmp` and `/data`
- Dropped capabilities except NET_BIND_SERVICE
- Resource limits (CPU: 0.5, Memory: 512M for lightweight)
- Security options (no-new-privileges)
- Health checks at `/digest-status/health` (12-hour intervals in docker-compose)
- Strict file permissions (550 for code, 750 for data)

### API Workflow

1. **POST `/generate-digest`** → Returns task_id immediately
2. **Background Process**:
   - Fetch arXiv papers + news articles (parallel)
   - Rank mixed content by relevance
   - Extract full content for top items
   - Analyze with GPT models
   - Generate HTML digest with separate sections
3. **GET `/digest-status/{task_id}`** → Check progress/results
4. **Optional**: Webhook callbacks on completion

### Important Notes

- The system uses the lightweight implementation by default (httpx instead of Playwright)
- State persistence uses JSON files in the `data/` directory or Supabase if configured
- All LLM interactions now use direct OpenAI client (not pydantic-ai)
- Cloud Run deployment supports higher concurrency with Supabase state management
- News fetching gracefully degrades if APIs are unavailable
- Content extraction is prioritized for high-relevance articles
- The archived full implementation with Playwright/crawl4ai is in `archived/`
- **Missing Supabase Schema**: The deployment mentions `supabase_setup.sql` but this file doesn't exist - you'll need to create tables manually
- **Pipedream Integration**: There's a webhook integration in `pipedream/trigger-generate-digest.js` for batch user processing

### Common Development Tasks

```bash
# Check Docker logs
docker-compose logs -f

# Clean up Docker resources
docker-compose down --volumes --remove-orphans
docker system prune

# Check API health
curl http://localhost:8000/digest-status/health

# Test API with authentication
curl -X POST http://localhost:8000/generate-digest \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"user_info": {"name": "Test User", "title": "Researcher", "goals": "ML papers"}}'

# Deploy to Cloud Run with environment variables and Secret Manager
export GCP_PROJECT_ID=your-project-id
export NEWSAPI_KEY=your-newsapi-key
export TAVILY_API_KEY=your-tavily-key
export SUPABASE_URL=your-supabase-url
export SUPABASE_KEY=your-supabase-key
./deploy_cloudrun.sh

# Check Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=paperboy" --limit 50
```

### Performance Considerations

- The ranking stage is limited to `RANKING_INPUT_MAX_ARTICLES` to prevent context overflow
- Article content is truncated to `ANALYSIS_CONTENT_MAX_CHARS` for analysis
- Background tasks have a `TASK_TIMEOUT` to prevent hanging operations
- HTTP requests use `HTTP_TIMEOUT` for reliability
- News API calls are cached with `NEWS_CACHE_TTL` to reduce costs
- Content extraction is limited to `NEWS_MAX_EXTRACT` articles to conserve Tavily quota
- Consider using GPT-3.5-turbo for cost optimization during development

### News Feature Implementation

The system now includes comprehensive news integration:

1. **Query Generation**: Smart query generation based on user profile using LLM
2. **News Fetching**: NewsAPI integration with rate limiting and caching
3. **Content Extraction**: Tavily API for full article content with prioritization
4. **Mixed Ranking**: Papers and news ranked together with type-aware scoring
5. **Visual Separation**: HTML output has distinct sections for papers vs news
6. **Source Control**: Use `digest_sources` parameter to control which sources to include

Error handling ensures the digest continues even if news APIs fail.

### Deployment Scripts

**deploy_cloudrun.sh** features:
- Automatic Google Cloud API enablement
- Secret Manager integration for secure environment variables
- Service account support with automatic fallback
- Colored output for better user experience
- Automatic secret creation/update from .env file
- Comprehensive error handling and validation

**cloudbuild.yaml** configuration:
- Uses Artifact Registry (not Container Registry)
- High-performance build machine (N1_HIGHCPU_8)
- Automatic secret mapping from Secret Manager
- Cloud logging integration