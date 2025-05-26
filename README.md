# Article Ranking Agent

A sophisticated AI agent that intelligently finds and ranks relevant academic papers based on user information and research interests. Built with modern Python practices, FastAPI, and robust error handling.

## Project Structure

```
paperboy/
├── src/                    # Source code
│   ├── agent.py           # Main agent logic
│   ├── agent_tools.py     # Agent utility functions/tools
│   ├── agent_prompts.py   # Agent prompts
│   ├── config.py          # Configuration loading (Pydantic BaseSettings)
│   ├── models.py          # Pydantic data models
│   ├── api_models.py      # FastAPI request/response models
│   ├── main.py           # FastAPI application entry point
│   ├── security.py       # API security implementation
│   ├── state.py          # State management (if used)
│   └── __init__.py        # Package initialization
├── tests/                 # Test files
│   ├── test_agent.py
│   ├── test_agent_tools.py
│   ├── integration_test.py
│   ├── test_state.json    # Example state data
│   └── __init__.py
├── config/               # Configuration files directory
│   ├── .env             # Environment variables (loaded by src/config.py)
│   └── .env.example     # Example environment variables
├── data/                # Data files
│   └── agent_state.json # Example agent state persistence
├── notebooks/           # Jupyter notebooks (for experimentation)
├── Dockerfile          # Lightweight Docker configuration (default)
├── Dockerfile.full     # Full Docker configuration with Playwright
├── docker-compose.yaml # Docker Compose configuration
├── .dockerignore       # Docker ignore rules
├── .gitignore          # Git ignore rules
└── docs/               # Documentation files
```

## Features

- **Smart Article Recommendations**

  - Personalized article suggestions based on user profile and research interests
  - Advanced relevance scoring with detailed reasoning
  - Support for large article datasets with automatic truncation

- **FastAPI Integration**

  - RESTful API endpoints for digest generation and status checking
  - Asynchronous processing with background tasks
  - API key authentication for secure access
  - Callback URL support for status updates
  - Swagger/OpenAPI documentation at `/docs`

- **Docker Support**

  - Multi-stage build for optimized image size
  - Development and production configurations
  - Easy deployment with Docker Compose
  - Health check endpoints
  - Automatic environment isolation

- **Robust Architecture**

  - Modular design with clear separation of concerns (`agent`, `tools`, `config`, `models`)
  - Centralized, type-safe configuration via Pydantic `BaseSettings`
  - Comprehensive error handling and logging
  - State management for persistent agent memory
  - Configurable model selection and parameters via environment variables

- **Developer-Friendly**
  - Type hints and Pydantic models for better code quality and validation
  - Extensive test coverage
  - Clear documentation and implementation notes
  - Easy configuration through a `.env` file
  - Hot-reload support in development

## Installation

### Using Docker (Recommended)

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd paperboy
   ```

2. Set up environment variables:

   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your settings
   ```

3. Build and run with Docker Compose:

   ```bash
   docker-compose up --build
   ```

   The API will be available at:

   - API Endpoints: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative Documentation: http://localhost:8000/redoc

### Manual Installation (Not Recommended)

Only use this method if you cannot use Docker for some reason. Docker is the recommended way to run this application as it ensures consistent environments and includes all necessary dependencies.

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd paperboy
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your settings
   ```

## Configuration

Configuration is managed centrally via Pydantic `BaseSettings` in `src/config.py` and loaded from environment variables defined in `config/.env`.

### Environment Variables (`config/.env`)

| Variable                     | Description                                         | Default (`src/config.py`) |
| ---------------------------- | --------------------------------------------------- | ------------------------- |
| `OPENAI_API_KEY`             | OpenAI API key                                      | **Required**              |
| `OPENAI_MODEL`               | OpenAI Model ID                                     | `gpt-4o`                  |
| `ARXIV_FILE`                 | Article data filename (in `data/`) or absolute path | `arxiv_papers.json`       |
| `TOP_N_ARTICLES`             | Number of articles to rank/analyze                  | `5`                       |
| `LOG_LEVEL`                  | Logging level (DEBUG, INFO, etc.)                   | `INFO`                    |
| `CRAWLER_TIMEOUT`            | Web crawler page timeout (ms)                       | `25000`                   |
| `AGENT_RETRIES`              | Pydantic AI agent retry attempts                    | `2`                       |
| `ANALYSIS_CONTENT_MAX_CHARS` | Max characters of article content sent for analysis | `8000`                    |
| `RANKING_INPUT_MAX_ARTICLES` | Max number of raw articles sent to LLM for ranking  | `20`                      |
| `API_KEY`                    | Secret key for API authentication                   | **Required**              |

## Usage

### API Endpoints

The service exposes the following REST API endpoints:

1. **Generate Digest** - `POST /generate-digest`

   ```bash
   curl -X POST http://localhost:8000/generate-digest \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{
       "user_info": {
         "name": "John Doe",
         "title": "AI Researcher",
         "goals": "Looking for papers on LLMs and transformers"
       },
       "target_date": "2024-03-20",
       "top_n_articles": 5,
       "callback_url": "http://your-callback-url.com/webhook"
     }'
   ```

2. **Check Digest Status** - `GET /digest-status/{task_id}`

   ```bash
   curl http://localhost:8000/digest-status/{task_id} \
     -H "X-API-Key: your_api_key"
   ```

3. **Health Check** - `GET /digest-status/health`
   ```bash
   curl http://localhost:8000/digest-status/health
   ```

For full API documentation, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker Commands

- **Start the service:**

  ```bash
  docker-compose up
  ```

- **Rebuild and start:**

  ```bash
  docker-compose up --build
  ```

- **Run in background:**

  ```bash
  docker-compose up -d
  ```

- **View logs:**

  ```bash
  docker-compose logs -f
  ```

- **Stop the service:**

  ```bash
  docker-compose down
  ```

- **Clean up unused resources:**
  ```bash
  docker-compose down --volumes --remove-orphans
  docker system prune
  ```

### Development Mode

For development with hot-reload:

```bash
docker-compose up --build
```

The service will automatically reload when you make changes to the code.

## Development

### Adding New Features

1. Create feature branch:

   ```bash
   git checkout -b feature/your-feature
   ```

2. Implement changes (e.g., add new tools in `src/agent_tools.py`, update models in `src/models.py`).
3. If adding configuration, update `src/config.py` and `config/.env.example`.
4. Add tests in `tests/`.
5. Update `README.md` if necessary.
6. Submit pull request.

### Code Style

- Follow PEP 8 guidelines.
- Use type hints (`pydantic` heavily relies on them).
- Document functions, classes, and Pydantic models clearly.
- Keep functions focused and modular (Single Responsibility Principle).

## Troubleshooting

Common issues and solutions:

1. **`ValidationError` on startup:**

   - Ensure `config/.env` file exists and contains all required variables (like `OPENAI_API_KEY`).
   - Check variable names in `config/.env` match the `validation_alias` in `src/config.py`.
   - Verify API key validity.

2. **Docker Issues:**

   - If the container fails to start, check logs: `docker-compose logs -f`
   - For permission issues: `sudo chown -R $USER:$USER .`
   - If port 8000 is in use: modify the port mapping in `docker-compose.yaml`
   - For cleanup: `docker-compose down --volumes --remove-orphans && docker system prune`

3. **API Authentication Errors:**

   - Ensure the `X-API-Key` header matches the `API_KEY` in your `.env` file
   - Check if the API key is properly set in your environment
   - Verify the header name is exactly `X-API-Key` (case-sensitive)

4. **Memory Issues / Context Limits:**

   - The agent currently loads all articles specified by `ARXIV_FILE` into memory.
   - The number of raw articles sent to the LLM for ranking is limited by `RANKING_INPUT_MAX_ARTICLES`.
   - The amount of content from each article sent for analysis is limited by `ANALYSIS_CONTENT_MAX_CHARS`.
   - If Docker container runs out of memory, adjust memory limits in `docker-compose.yaml`

5. **Performance / Cost:**
   - Consider using a less expensive model like `gpt-3.5-turbo` via `OPENAI_MODEL` in `config/.env`.
   - Adjust `AGENT_RETRIES` if excessive retries are occurring.
   - Use background processing for long-running tasks
   - Implement caching if needed

## Acknowledgments

- OpenAI for API access
- FastAPI and Pydantic maintainers
- Docker and container ecosystem
- Academic paper sources (e.g., ArXiv)
