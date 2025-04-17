# Article Ranking Agent

A sophisticated AI agent that intelligently finds and ranks relevant academic papers based on user information and research interests. Built with modern Python practices and robust error handling.

## Project Structure

```
paperboy/
├── src/                    # Source code
│   ├── agent.py           # Main agent implementation
│   ├── agent_tools.py     # Agent utility functions
│   ├── state.py           # State management
│   ├── agent_prompts.py   # Agent prompts
│   └── __init__.py        # Package initialization
├── tests/                 # Test files
│   ├── test_agent.py
│   ├── test_agent_tools.py
│   ├── integration_test.py
│   ├── test_state.json
│   └── __init__.py
├── config/               # Configuration files
│   ├── .env             # Environment variables
│   ├── .env.example     # Example environment variables
│   └── settings.py      # Application settings
├── data/                # Data files
│   └── agent_state.json # Agent state persistence
├── notebooks/           # Jupyter notebooks
└── docs/               # Documentation
```

## Features

- **Smart Article Recommendations**

  - Personalized article suggestions based on user profile and research interests
  - Advanced relevance scoring with detailed reasoning
  - Support for large article datasets with automatic truncation

- **Robust Architecture**

  - Modular design with clear separation of concerns
  - Comprehensive error handling and logging
  - State management for persistent agent memory
  - Configurable model selection and parameters

- **Developer-Friendly**
  - Type hints and Pydantic models for better code quality
  - Extensive test coverage
  - Clear documentation and implementation notes
  - Easy configuration through environment variables

## Installation

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd article-ranking-agent
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

   ```bash
   cp .env.example .env
   ```

5. Edit `.env` to configure:
   - `OPENAI_API_KEY`: Your OpenAI API key (required)
   - `OPENAI_MODEL`: The OpenAI model to use (default: gpt-4)
   - `ARXIV_FILE`: Path to the arXiv data file
   - `TOP_N_ARTICLES`: Number of top articles to return
   - Additional configuration options as needed

## Usage

### Basic Usage

Run the agent with:

```bash
python agent.py
```

The agent will:

1. Load articles from the specified file
2. Process user information and research interests
3. Rank articles based on relevance
4. Display top papers with scores and reasoning

### Advanced Usage

#### Custom User Profile

Edit the user information in `agent.py`:

```python
user_info = {
    "name": "Your Name",
    "title": "Your Title",
    "goals": "Your research interests and goals",
    "expertise": ["field1", "field2"],
    "preferences": {
        "publication_types": ["conference", "journal"],
        "date_range": "last_year"
    }
}
```

#### Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Configuration

### Environment Variables

| Variable         | Description       | Default                              |
| ---------------- | ----------------- | ------------------------------------ |
| `OPENAI_API_KEY` | OpenAI API key    | Required                             |
| `OPENAI_MODEL`   | Model to use      | gpt-4                                |
| `ARXIV_FILE`     | Article data file | arxiv_cs_submissions_2025-04-01.json |
| `TOP_N_ARTICLES` | Number of results | 5                                    |
| `LOG_LEVEL`      | Logging level     | INFO                                 |

### Model Configuration

The agent supports various OpenAI models and can be configured for different use cases:

- GPT-4: Best for complex reasoning and detailed analysis
- GPT-3.5: Faster, more cost-effective for simpler tasks

## Development

### Adding New Features

1. Create feature branch:

   ```bash
   git checkout -b feature/your-feature
   ```

2. Implement changes
3. Add tests in `tests/`
4. Submit pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes
- Keep functions focused and modular

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**

   - Ensure `.env` file exists and contains valid API key
   - Check API key permissions

2. **Memory Issues**

   - Adjust `TOP_N_ARTICLES` for smaller datasets
   - Use truncation settings for large articles

3. **Performance**
   - Consider using GPT-gpt-4.1-mini-2025-04-14 for faster results
   - Adjust batch sizes in configuration

## Acknowledgments

- OpenAI for API access
- Contributors and maintainers
- Academic paper sources
