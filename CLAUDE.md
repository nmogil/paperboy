# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paperboy is an AI-powered academic paper recommendation system that ranks and analyzes arXiv papers based on user research profiles. It uses OpenAI models to provide personalized digests of relevant research papers and now includes news integration via NewsAPI and Tavily for content extraction.

## Key Commands

### Running the Service

```bash
# Development with Docker Compose
docker-compose up --build

# Production deployment
docker build -t paperboy:latest .
docker run -p 8000:8000 --env-file config/.env paperboy:latest

# Cloud Run deployment
./deploy_cloudrun.sh

# Cloud Build deployment (automatic)
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

# Run integration tests
pytest tests/integration_test.py

# Run with verbose output
pytest -v tests/test_agent.py::test_rank_articles
```

### Development Setup

```bash
# Install dependencies
pip install -r requirements.lightweight.txt

# Run locally without Docker (default port 8000)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# View API documentation
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### Linting and Type Checking

```bash
# If available, run these commands before committing:
# Note: Check if these are configured in the project
# ruff check src/
# mypy src/
```

## Architecture Overview

The codebase follows a modular architecture with clear separation of concerns:

1. **API Layer** (`main.py`): FastAPI application handling HTTP endpoints and background tasks
2. **Digest Service** (`digest_service.py`): Orchestrates the full digest generation workflow
3. **LLM Client** (`llm_client.py`): Direct OpenAI integration for ranking and analysis
4. **Data Fetching** (`fetcher_lightweight.py`): Web scraping of arXiv using httpx
5. **News Integration** (`news_fetcher.py`): NewsAPI integration with intelligent query generation
6. **Content Extraction** (`content_extractor.py`): Tavily API for full article content extraction
7. **Query Generation** (`query_generator.py`): Smart news query generation based on user profiles
8. **Models** (`models.py`): Pydantic models for type safety across the system
9. **Configuration** (`config.py`): Centralized settings management via Pydantic BaseSettings
10. **State Management** (`state.py`): JSON-based task state persistence
11. **Security** (`security.py`): API key authentication middleware
12. **Caching** (`cache.py`): In-memory cache with TTL for API responses
13. **Metrics** (`metrics.py`): Performance monitoring and API call tracking

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

### Environment Configuration

Required environment variables in `config/.env`:

**Core Configuration:**
- `OPENAI_API_KEY`: OpenAI API access (required)
- `API_KEY`: Authentication for API endpoints (required)
- `OPENAI_MODEL`: Model selection (default: gpt-4o)
- `TOP_N_ARTICLES`: Number of articles to analyze (default: 5)
- `LOG_LEVEL`: Logging level (default: INFO)

**Performance Settings:**
- `CRAWLER_TIMEOUT`: Web crawler timeout in ms (default: 25000)
- `HTTP_TIMEOUT`: HTTP request timeout in seconds (default: 30)
- `TASK_TIMEOUT`: Max digest generation time in seconds (default: 300)
- `AGENT_RETRIES`: LLM retry attempts (default: 2)
- `ANALYSIS_CONTENT_MAX_CHARS`: Max content for analysis (default: 8000)
- `RANKING_INPUT_MAX_ARTICLES`: Max articles for ranking (default: 20)

**News Integration:**
- `NEWSAPI_KEY`: NewsAPI key for news fetching (optional)
- `TAVILY_API_KEY`: Tavily API key for content extraction (optional)
- `NEWS_ENABLED`: Enable news fetching (default: true)
- `NEWS_MAX_ARTICLES`: Max news articles to fetch (default: 50)
- `NEWS_MAX_EXTRACT`: Max articles to extract full content (default: 10)
- `NEWS_CACHE_TTL`: Cache duration in seconds (default: 3600)

**Deployment:**
- `USE_LIGHTWEIGHT`: Use httpx instead of Playwright (default: true)
- `LOGFIRE_TOKEN`: Monitoring service token (optional)

### Testing Approach

Tests use pytest with mocked external API calls:

- Mock `openai.AsyncOpenAI` for deterministic testing
- Mock NewsAPI and Tavily responses for integration tests
- Test both ranking and analysis workflows
- Validate model attributes and JSON parsing
- Test error handling and graceful degradation

### Docker Security

Production containers implement:

- Non-root user (UID 10001)
- Read-only filesystem with tmpfs mounts for `/tmp` and `/data`
- Dropped capabilities except NET_BIND_SERVICE
- Resource limits (CPU: 0.5, Memory: 512M for lightweight)
- Security options (no-new-privileges)
- Health checks at `/digest-status/health`
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
- State persistence uses JSON files in the `data/` directory
- All LLM interactions now use direct OpenAI client (not pydantic-ai)
- Cloud Run deployment uses concurrency=1 due to in-memory state management
- News fetching gracefully degrades if APIs are unavailable
- Content extraction is prioritized for high-relevance articles
- The archived full implementation with Playwright/crawl4ai is in `archived/`

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

# Deploy to Cloud Run with environment variables
export GCP_PROJECT_ID=your-project-id
export NEWSAPI_KEY=your-newsapi-key
export TAVILY_API_KEY=your-tavily-key
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

Error handling ensures the digest continues even if news APIs fail.