# Article Ranking Agent

A sophisticated AI agent that intelligently finds and ranks relevant academic papers based on user information and research interests. Built with modern Python practices and robust error handling.

## Project Structure

```
paperboy/
├── src/                    # Source code
│   ├── agent.py           # Main agent logic
│   ├── agent_tools.py     # Agent utility functions/tools
│   ├── agent_prompts.py   # Agent prompts
│   ├── config.py          # Configuration loading (Pydantic BaseSettings)
│   ├── models.py          # Pydantic data models
│   ├── state.py           # State management (if used)
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
└── docs/               # Documentation files
```

## Features

- **Smart Article Recommendations**

  - Personalized article suggestions based on user profile and research interests
  - Advanced relevance scoring with detailed reasoning
  - Support for large article datasets with automatic truncation

- **Robust Architecture**

  - Modular design with clear separation of concerns (`agent`, `tools`, `config`, `models`)
  - Centralized, type-safe configuration via Pydantic `BaseSettings`
  - Comprehensive error handling and logging
  - State management for persistent agent memory (if implemented)
  - Configurable model selection and parameters via environment variables

- **Developer-Friendly**
  - Type hints and Pydantic models for better code quality and validation
  - Extensive test coverage (planned/included)
  - Clear documentation and implementation notes
  - Easy configuration through a `.env` file

## Installation

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd article-ranking-agent # Or your project directory name
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   - Go to the `config/` directory: `cd config`
   - Copy the example `.env` file:
     ```bash
     cp .env.example .env
     ```
   - Go back to the root directory: `cd ..`

5. Edit `config/.env` to configure:
   - `OPENAI_API_KEY`: Your OpenAI API key (required).
   - `OPENAI_MODEL`: The OpenAI model to use (default: `gpt-4o`).
   - `ARXIV_FILE`: Path to the arXiv data file (relative to `data/` directory or absolute, default: `arxiv_papers.json`).
   - `TOP_N_ARTICLES`: Number of top articles to return (default: `5`).
   - `LOG_LEVEL`: Logging level (e.g., `INFO`, `DEBUG`, default: `INFO`).
   - `CRAWLER_TIMEOUT`: Timeout in milliseconds for web crawling (default: `25000`).
   - `AGENT_RETRIES`: Number of times the agent should retry on failure (default: `2`).
   - Additional configuration options as needed (add them to `src/config.py` and `config/.env.example`).
     Configuration is loaded at startup by `src/config.py`.

## Usage

### Basic Usage

Run the agent from the project root directory using the `-m` flag to treat it as a module:

```bash
python -m src.agent
```

The agent will:

1. Load configuration from `config/.env` via `src/config.py`.
2. Load articles from the specified file (default: `data/arxiv_papers.json`).
3. Process user information and research interests (defined in `src/agent.py::main`).
4. Rank articles based on relevance using the configured LLM.
5. Analyze the top ranked articles using the LLM.
6. Generate an HTML summary (`arxiv_digest.html`) in the root directory.

### Advanced Usage

#### Custom User Profile

Edit the `user_info` `UserContext` object in the `main()` function within `src/agent.py`:

```python
# In src/agent.py -> main()
user_info = UserContext(
    name="Your Name",
    title="Your Title",
    goals="Your research interests and goals"
)
```

#### Testing

Run the test suite (assuming tests are set up):

```bash
python -m pytest tests/
```

## Configuration

Configuration is managed centrally via Pydantic `BaseSettings` in `src/config.py` and loaded from environment variables defined in `config/.env`.

### Environment Variables (`config/.env`)

| Variable          | Description                                         | Default (`src/config.py`) |
| ----------------- | --------------------------------------------------- | ------------------------- |
| `OPENAI_API_KEY`  | OpenAI API key                                      | **Required**              |
| `OPENAI_MODEL`    | OpenAI Model ID                                     | `gpt-4o`                  |
| `ARXIV_FILE`      | Article data filename (in `data/`) or absolute path | `arxiv_papers.json`       |
| `TOP_N_ARTICLES`  | Number of articles to rank/analyze                  | `5`                       |
| `LOG_LEVEL`       | Logging level (DEBUG, INFO, etc.)                   | `INFO`                    |
| `CRAWLER_TIMEOUT` | Web crawler page timeout (ms)                       | `25000`                   |
| `AGENT_RETRIES`   | Pydantic AI agent retry attempts                    | `2`                       |

### Model Configuration

The agent supports various OpenAI models configured via the `OPENAI_MODEL` environment variable.

- **`gpt-4o` (Default):** Balances performance and capability.
- **`gpt-4-turbo`:** Strong reasoning capabilities.
- **`gpt-3.5-turbo`:** Faster, more cost-effective for simpler tasks.

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

2. **`ImportError: attempted relative import with no known parent package`:**

   - Ensure you are running the agent from the project root directory using `python -m src.agent`.

3. **Memory Issues:**

   - The agent currently loads all articles specified by `ARXIV_FILE` into memory. Consider processing large files in chunks if memory becomes an issue.
   - The number of articles sent to the LLM for ranking is currently capped at 20 within `src/agent.py::rank_articles`.

4. **Performance / Cost:**
   - Consider using a less expensive model like `gpt-3.5-turbo` via `OPENAI_MODEL` in `config/.env`.
   - Adjust `AGENT_RETRIES` if excessive retries are occurring.

## Acknowledgments

- OpenAI for API access
- Pydantic & Pydantic-AI maintainers
- Contributors and maintainers
- Academic paper sources (e.g., ArXiv)
