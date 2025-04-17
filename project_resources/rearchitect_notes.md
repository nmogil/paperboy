# Project Rearchitecture Guide

## Overview

This guide provides a detailed, step-by-step implementation plan for rearchitecting the project to follow best practices and improve maintainability. The guide is written for junior engineers and includes specific code examples, testing strategies, and common pitfalls to avoid.

## Current Architecture Analysis

### Current Structure

- **agent.py**: Contains agent definitions and some orchestration logic
- **agent_tools.py**: Contains tool implementations and some agent-like functionality

### Problems Identified

1. Agent logic spread across multiple files
2. Unclear separation of concerns
3. Tool functions doing agent-like work
4. Potential circular dependencies
5. Difficult to test and maintain

## Step 1: Project Structure Setup

Create the following directory structure:

```
project/
├── agent.py           # All agent orchestration and coordination logic
├── agent_tools.py     # Pure tools/utilities: atomic, stateless, no decisions
├── agent_prompts.py   # Prompt templates and system instructions
├── tests/
│   ├── test_agent.py
│   └── test_agent_tools.py
├── requirements.txt
├── .env.example
└── README.md
```

## Step 2: Refactor agent_tools.py

### A. Identify Pure Tool Functions

1. List all functions that should be pure tools:
   - `analyze_article`
   - `scrape_article`
   - Any other utility functions

### B. Remove Agent Logic

1. Move any decision-making or orchestration to agent.py
2. Ensure each function is stateless
3. Add proper type hints and docstrings

### C. Example Implementation

```python
# agent_tools.py

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def analyze_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single article for scoring, keywords, and sentiment.

    Args:
        article: Dictionary containing article data

    Returns:
        Dictionary containing analysis results:
        - score: float between 0 and 1
        - keywords: List of important terms
        - sentiment: float between -1 and 1
    """
    try:
        # Pure analysis logic here
        score = compute_article_score(article)
        keywords = extract_keywords(article.get("body", ""))
        sentiment = analyze_sentiment(article.get("body", ""))

        return {
            "score": score,
            "keywords": keywords,
            "sentiment": sentiment
        }
    except Exception as e:
        logger.error(f"Error analyzing article: {e}")
        return {
            "score": 0.0,
            "keywords": [],
            "sentiment": 0.0
        }

def compute_article_score(article: Dict[str, Any]) -> float:
    """Compute a score for an article based on various metrics."""
    # Implementation here
    pass

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from text."""
    # Implementation here
    pass

def analyze_sentiment(text: str) -> float:
    """Analyze sentiment of text, returning value between -1 and 1."""
    # Implementation here
    pass
```

## Step 3: Refactor agent.py

### A. Move All Agent Logic Here

1. Define clear agent classes/functions
2. Handle orchestration and decision-making
3. Use tools from agent_tools.py

### B. Example Implementation

```python
# agent.py

from typing import List, Dict, Any
from agent_tools import analyze_article
import logging

logger = logging.getLogger(__name__)

class ArticleRankAgent:
    """Agent responsible for ranking and analyzing articles."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    async def rank_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank articles based on analysis and user preferences.

        Args:
            articles: List of article dictionaries

        Returns:
            List of articles sorted by relevance score
        """
        try:
            scored_articles = []
            for article in articles:
                # Use pure tool function for analysis
                analysis = analyze_article(article)

                # Agent makes decision about ranking
                scored_articles.append({
                    "article": article,
                    "score": self._compute_final_score(analysis),
                    "analysis": analysis
                })

            # Sort by score
            ranked = sorted(scored_articles, key=lambda x: x["score"], reverse=True)
            return ranked

        except Exception as e:
            logger.error(f"Error ranking articles: {e}")
            return []

    def _compute_final_score(self, analysis: Dict[str, Any]) -> float:
        """Compute final score based on analysis and agent preferences."""
        # Implementation here
        pass
```

## Step 4: Testing Strategy

### A. Unit Tests for Tools

```python
# tests/test_agent_tools.py

import pytest
from agent_tools import analyze_article

def test_analyze_article_basic():
    article = {
        "title": "Test Article",
        "body": "This is a good article about AI."
    }
    result = analyze_article(article)

    assert isinstance(result, dict)
    assert "score" in result
    assert "keywords" in result
    assert "sentiment" in result
    assert 0 <= result["score"] <= 1
    assert isinstance(result["keywords"], list)
    assert -1 <= result["sentiment"] <= 1

def test_analyze_article_empty():
    article = {"title": "Empty", "body": ""}
    result = analyze_article(article)

    assert result["score"] == 0
    assert result["keywords"] == []
    assert result["sentiment"] == 0
```

### B. Integration Tests for Agent

```python
# tests/test_agent.py

import pytest
from agent import ArticleRankAgent

@pytest.mark.asyncio
async def test_rank_articles():
    agent = ArticleRankAgent()
    articles = [
        {"title": "Good Article", "body": "This is excellent content."},
        {"title": "Bad Article", "body": "This is poor content."},
        {"title": "Neutral Article", "body": "This is okay content."}
    ]

    ranked = await agent.rank_articles(articles)

    assert len(ranked) == len(articles)
    assert ranked[0]["score"] >= ranked[1]["score"] >= ranked[2]["score"]
```

## Step 5: Migration Plan

1. **Preparation**

   - Create backup of current code
   - Set up new directory structure
   - Create empty test files

2. **Refactor Tools**

   - Move pure functions to agent_tools.py
   - Add type hints and docstrings
   - Write unit tests

3. **Refactor Agent**

   - Move agent logic to agent.py
   - Update imports
   - Write integration tests

4. **Testing**

   - Run unit tests
   - Run integration tests
   - Fix any issues

5. **Documentation**
   - Update README.md
   - Add docstrings
   - Create .env.example

## Step 6: Common Pitfalls to Avoid

1. **Leaving Agent Logic in Tools**

   - Problem: Tools making decisions or maintaining state
   - Solution: Move all decision-making to agent.py

2. **Circular Dependencies**

   - Problem: agent.py imports from agent_tools.py which imports from agent.py
   - Solution: Keep dependencies one-way (tools → agent)

3. **Missing Error Handling**

   - Problem: Functions failing silently
   - Solution: Add proper try/except blocks and logging

4. **Insufficient Testing**

   - Problem: Not testing edge cases
   - Solution: Write comprehensive tests for both success and failure cases

5. **Poor Type Hints**
   - Problem: Functions with unclear inputs/outputs
   - Solution: Add detailed type hints and docstrings

## Step 7: Best Practices

### For agent_tools.py

- Keep functions pure and stateless
- Add comprehensive type hints
- Include detailed docstrings
- Handle errors gracefully
- Log important events

### For agent.py

- Handle all orchestration and decision-making
- Use clear class/function names
- Document agent behavior
- Handle errors at the agent level
- Log agent decisions

### For Testing

- Write unit tests for all tools
- Write integration tests for agents
- Test edge cases and error conditions
- Use pytest fixtures for common setup

### For Documentation

- Keep README.md up to date
- Document all environment variables
- Include examples in docstrings
- Add comments for complex logic

## Conclusion

This rearchitecture improves:

1. Code organization and maintainability
2. Testability and reliability
3. Separation of concerns
4. Error handling and logging
5. Documentation and type safety

Follow this guide step-by-step to successfully rearchitect your project. If you encounter any issues or need clarification on any step, refer to the examples and best practices provided.
