# Article Ranking Agent

A simple, robust AI agent that finds and ranks relevant academic papers based on user information and research interests.

## Features

- Personalized article recommendations based on user profile and research interests
- Robust JSON parsing and validation with Pydantic
- Configurable model selection and parameters
- Handles large article datasets with automatic truncation
- Comprehensive error handling and logging

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```
   cp .env.example .env
   ```
4. Edit `.env` to add your OpenAI API key and customize other settings

## Usage

Run the agent with:

```
python agent.py
```

The agent will:

1. Load articles from the specified file (or use sample data if not found)
2. Rank the articles based on the user's research interests
3. Display the top relevant papers with relevance scores and reasoning

## Configuration

You can configure the agent using environment variables in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: The OpenAI model to use (default: gpt-4o)
- `ARXIV_FILE`: Path to the arXiv data file (default: arxiv_cs_submissions_2025-04-01.json)
- `TOP_N_ARTICLES`: Number of top articles to return (default: 5)

## Customization

To use your own user information, edit the `user_info` dictionary in the `main()` function:

```python
user_info = {
    "name": "Your Name",
    "title": "Your Title",
    "goals": "Your research interests and goals"
}
```

## Implementation Notes

See `project_resources/implementation_notes.md` for details on the implementation approach and learnings.
