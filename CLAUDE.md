# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paperboy is an AI-powered academic paper recommendation system that ranks and analyzes arXiv papers based on user research profiles. It uses OpenAI models to provide personalized digests of relevant research papers.

## Key Commands

### Running the Service

```bash
# Development with Docker Compose (Lightweight version)
docker-compose -f docker-compose.lightweight.yaml up --build

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
```

### Development Setup

```bash
# Install dependencies
pip install -r requirements.lightweight.txt

# Run locally without Docker
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# View API documentation
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

## Architecture Overview

The codebase follows a modular architecture with clear separation of concerns:

1. **API Layer** (`main.py`): FastAPI application handling HTTP endpoints and background tasks
2. **Agent System** (`agent.py`): Core AI logic using Pydantic AI for paper ranking and analysis
3. **Data Fetching** (`fetcher_lightweight.py`): Web scraping of arXiv using httpx (lightweight HTTP client)
4. **Agent Tools** (`agent_tools_lightweight.py`): Article content extraction and LLM-based analysis
5. **Models** (`models.py`): Pydantic models for type safety across the system
6. **Configuration** (`config.py`): Centralized settings management via environment variables

### Key Design Patterns

- **Async/Await**: All I/O operations are async for scalability
- **Background Tasks**: Long-running operations handled via FastAPI BackgroundTasks
- **Type Safety**: Extensive use of Pydantic models for validation
- **Dependency Injection**: Configuration injected via Pydantic BaseSettings
- **Security**: API key authentication middleware on all endpoints

### Critical Dependencies

- `pydantic-ai`: Agent framework for LLM orchestration
- `openai`: GPT model integration for ranking and analysis
- `httpx`: Async HTTP client for web scraping (lightweight alternative)
- `beautifulsoup4`: HTML parsing for content extraction
- `fastapi`: Modern async web framework
- `logfire`: Production monitoring and observability

### Environment Configuration

Required environment variables in `config/.env`:

- `OPENAI_API_KEY`: OpenAI API access (required)
- `API_KEY`: Authentication for API endpoints (required)
- `OPENAI_MODEL`: Model selection (default: gpt-4o)
- `TOP_N_ARTICLES`: Number of articles to analyze (default: 5)
- `LOG_LEVEL`: Logging level (default: INFO)
- `CRAWLER_TIMEOUT`: Web crawler timeout in ms (default: 25000)
- `AGENT_RETRIES`: LLM retry attempts (default: 2)
- `ANALYSIS_CONTENT_MAX_CHARS`: Max content for analysis (default: 8000)
- `RANKING_INPUT_MAX_ARTICLES`: Max articles for ranking (default: 20)
- `USE_LIGHTWEIGHT`: Use httpx instead of Playwright (default: true)
- `HTTP_TIMEOUT`: HTTP request timeout in seconds (default: 30)
- `TASK_TIMEOUT`: Max time for digest generation (default: 300)
- `LOGFIRE_TOKEN`: Monitoring service token (optional)

### Testing Approach

Tests use pytest with mocked OpenAI calls. Key test patterns:

- Mock `openai.AsyncOpenAI` for deterministic testing
- Test both ranking and analysis workflows
- Validate model attributes and JSON parsing

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

1. POST `/generate-digest` → Returns task_id immediately
2. Background: Fetch → Rank → Scrape → Analyze → Generate HTML
3. GET `/digest-status/{task_id}` → Check progress/results
4. Optional: Callbacks sent to provided webhook URL

### Important Notes

- The system now uses the lightweight implementation by default (`fetcher_lightweight.py`, `agent_tools_lightweight.py`) with httpx for better Cloud Run compatibility
- State persistence uses JSON files in the `data/` directory
- All LLM prompts are structured for JSON output (see `agent_prompts.py`)
- The system handles rate limiting and retries automatically
- The full implementation with Playwright/crawl4ai has been archived in `archived_full_implementation/` for future reference
- Cloud Run deployment uses concurrency=1 due to in-memory state management
- Async operations throughout with proper timeout handling
- Semaphore-based concurrency control for web scraping

### Common Development Tasks

```bash
# Check Docker logs
docker-compose logs -f

# Clean up Docker resources
docker-compose down --volumes --remove-orphans
docker system prune

# Run specific tests with verbose output
pytest -v tests/test_agent.py::test_rank_articles

# Check API health
curl http://localhost:8000/digest-status/health

# Test API with authentication
curl -X POST http://localhost:8000/generate-digest \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"user_info": {"name": "Test User", "title": "Researcher", "goals": "ML papers"}}'
```

### Performance Considerations

- The ranking stage is limited to `RANKING_INPUT_MAX_ARTICLES` to prevent context overflow
- Article content is truncated to `ANALYSIS_CONTENT_MAX_CHARS` for analysis
- Background tasks have a `TASK_TIMEOUT` to prevent hanging operations
- HTTP requests use `HTTP_TIMEOUT` for reliability
- Consider using GPT-3.5-turbo for cost optimization during development
