# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paperboy is an AI-powered academic paper recommendation system that ranks and analyzes arXiv papers based on user research profiles. It uses OpenAI models to provide personalized digests of relevant research papers.

## Key Commands

### Running the Service
```bash
# Development with Docker Compose
docker-compose up --build

# Lightweight version (without full scraping)
docker-compose -f docker-compose.lightweight.yaml up --build

# Production deployment
docker build -t paperboy:optimized-v2 -f Dockerfile .
docker run -p 8000:8000 --env-file config/.env paperboy:optimized-v2

# Cloud Run deployment
./deploy_cloudrun.sh
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agent.py

# Run with coverage
pytest --cov=src
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally without Docker
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## Architecture Overview

The codebase follows a modular architecture with clear separation of concerns:

1. **API Layer** (`main.py`): FastAPI application handling HTTP endpoints and background tasks
2. **Agent System** (`agent.py`): Core AI logic using Pydantic AI for paper ranking and analysis
3. **Data Fetching** (`fetcher.py`): Web scraping of arXiv using Crawl4AI with Playwright
4. **Agent Tools** (`agent_tools.py`): Article content extraction and LLM-based analysis
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
- `crawl4ai`: Web scraping with browser automation
- `fastapi`: Modern async web framework
- `logfire`: Production monitoring and observability

### Environment Configuration

Required environment variables in `config/.env`:
- `OPENAI_API_KEY`: OpenAI API access
- `API_KEY`: Authentication for API endpoints
- `OPENAI_MODEL`: Model selection (default: gpt-4o)
- `TOP_N_ARTICLES`: Number of articles to analyze (default: 5)

### Testing Approach

Tests use pytest with mocked OpenAI calls. Key test patterns:
- Mock `openai.AsyncOpenAI` for deterministic testing
- Test both ranking and analysis workflows
- Validate model attributes and JSON parsing

### Docker Security

Production containers implement:
- Non-root user (UID 10001)
- Read-only filesystem with tmpfs mounts
- Dropped capabilities except NET_BIND_SERVICE
- Resource limits (CPU: 1.0, Memory: 2G)

### API Workflow

1. POST `/generate-digest` → Returns task_id immediately
2. Background: Fetch → Rank → Scrape → Analyze → Generate HTML
3. GET `/digest-status/{task_id}` → Check progress/results
4. Optional: Callbacks sent to provided webhook URL

### Important Notes

- The lightweight version (`fetcher_lightweight.py`, `agent_tools_lightweight.py`) uses httpx instead of Playwright for environments without browser support
- State persistence uses JSON files in the `data/` directory
- All LLM prompts are structured for JSON output (see `agent_prompts.py`)
- The system handles rate limiting and retries automatically