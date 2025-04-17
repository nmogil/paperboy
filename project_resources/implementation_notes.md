# Article Analysis Agent Implementation Plan

## Overview

This document outlines the implementation plan for a new Pydantic AI Agent that will analyze articles from arXiv. The agent will take the output from the existing ranking agent, scrape each article's content using crawl4ai, and provide personalized analysis including summaries, importance explanations, and action recommendations.

## Revised Implementation Approach

### 1. Cohesive Integration with agent.py

- **Agent-Centered Design:**  
  Build the new Article Analysis Agent as a natural sibling or extension within agent.py, adopting the same dependency injection, tool registration, and Pydantic model schema practices as the current agent.
- **Batch Processing:**  
  Accept a batch of ranked articles (output from the relevance-ranking agent)—use a well-typed Pydantic model for this input.

### 2. Crawl4ai Scraping – Example-Based Usage

- **Dependency Injection:**  
  In agent.py's `Deps` (dependencies) dataclass, instantiate and inject the crawl4ai client/server instance, so it's shared across all agent tools.
- **Leverage Examples:**  
  Use patterns and selector logic from `crawl4ai_examples.md` and `crawl4ai_selectors.md`—create one or more scraping functions/tools that:
  - Take an article URL as input
  - Call crawl4ai (via its SDK, MCP, or HTTP API depending on architecture)
  - Select and extract only the main article content (abstract, full text, sections as applicable) with crawl4ai selectors
  - Return clean, validated text for further processing

### 3. Single End-to-End Pipeline

- **Main API/Handler:**  
  The main agent entrypoint (in agent.py) should:
  - Accept a batch of ranked articles (include fields like url, rank, title, summary, etc.)
  - Use async batch logic for scalable performance (e.g., parallel scraping of multiple articles if allowed by crawl4ai/client/server architecture)
- **Data Flow:**  
  For each article:

  1. Use the scrape tool to fetch and parse article content
  2. Feed text, user goals, and career info into the LLM via a prompt template
  3. Parse and validate the LLM response into a structured, Pydantic-validated result (`ArticleAnalysis`)
  4. Collect all results in a `BatchAnalysis` response

- **Output:**  
  Return a structured analysis per article, including summary, importance, and recommended action.

### 4. Prompt Engineering and LLM Tooling

- **agent_prompts.py:**  
  Write specific prompt templates (in agent_prompts.py) for the analysis LLM tool to:
  - Summarize the article
  - Contextualize its importance for the user's goals
  - Provide tailored, actionable advice
- **LLM Agent Methods:**  
  Use Pydantic AI's tool registration for the analysis step, passing the scraped content and user context. The LLM tool should output strongly-typed objects (with proper field separation, not just markdown).

### 5. Error Handling & Logging

- **Tool-level Error Handling:**  
  Add robust checks (timeouts, data-type assertions, crawl4ai error codes) within scraping and analysis tools. Log all failures but allow overall batch to proceed (collect partial results).
- **Recovery/Retry:**  
  Implement simple retry logic where feasible (e.g., re-request failed scrapings once).

### 6. File/Module Structure (Matching Pydantic AI Patterns)

- **agent.py:**
  - Agent configuration, core orchestration, and main handler.
- **agent_tools.py:**
  - crawl4ai scraping tools, LLM analysis tool.
- **agent_prompts.py:**
  - System prompts, templates, and instruction sets.
- **.env.example:**
  - Required ENV variables for crawl4ai, LLM, etc.
- **requirements.txt:**
  - Dependencies: pydantic, crawl4ai client, requests, bs4, transformers, etc.

### 7. Archon Integration

- **When to Use Archon:**

  - Use Archon for all Pydantic AI-specific implementations, including:
    1. Agent class structure and configuration
    2. Tool registration and implementation
    3. Pydantic model definitions and validation
    4. Prompt engineering and LLM integration
    5. Error handling patterns specific to Pydantic AI

- **Archon Implementation Flow:**

  1. First, use Archon to design and implement the core Pydantic AI agent structure
  2. Then, use Archon to implement the analysis tools and LLM integration
  3. Finally, use Archon to refine the prompt engineering and response parsing

- **Archon Input Requirements:**
  - Provide clear requirements for Pydantic AI implementation
  - Include examples of expected input/output formats
  - Specify any constraints or requirements for the agent's behavior

## Detailed Implementation

### 1. Data Models (in agent.py)

```python
# Extend existing models in agent.py
class ArticleAnalysis(BaseModel):
    """Pydantic model for article analysis results"""
    title: str
    authors: List[str]
    subject: str
    summary: str
    importance: str
    recommended_action: str
    abstract_url: str
    html_url: str
    pdf_url: str
    relevance_score: int
    score_reason: str

class BatchAnalysis(BaseModel):
    """Pydantic model for batch analysis results"""
    analyses: List[ArticleAnalysis]
```

### 2. Crawl4ai Scraping Tool (in agent_tools.py)

```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, LXMLWebScrapingStrategy
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import json
import logging
from bs4 import BeautifulSoup
import asyncio

logger = logging.getLogger("article_analyzer")

async def scrape_article(url: str) -> str:
    """
    Scrape article content using crawl4ai.

    Args:
        url: The URL of the article to scrape

    Returns:
        Cleaned text content of the article
    """
    # Define schema for article content extraction based on actual arXiv HTML structure
    # This should be verified by inspecting the actual HTML
    schema = {
        "name": "ArXiv Article Content",
        "baseSelector": "div.article-content",  # Verify this selector
        "fields": [
            {
                "name": "abstract",
                "selector": "div.abstract",  # Verify this selector
                "type": "text"
            },
            {
                "name": "introduction",
                "selector": "div.introduction",  # Verify this selector
                "type": "text"
            },
            {
                "name": "conclusion",
                "selector": "div.conclusion",  # Verify this selector
                "type": "text"
            }
        ]
    }

    # Create extraction strategy
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)

    # Configure crawler with comprehensive options
    config = CrawlerRunConfig(
        # Use LXML for better performance
        scraping_strategy=LXMLWebScrapingStrategy(),
        # Extraction strategy
        extraction_strategy=extraction_strategy,
        # Content filtering
        word_count_threshold=10,
        excluded_tags=["nav", "footer", "header"],
        exclude_external_links=True,
        # CSS selection for article content
        css_selector="div.article-content",  # Verify this selector
        # Process iframes if needed
        process_iframes=True,
        # Wait for dynamic content if needed
        wait_for="css:div.article-content",
        # No caching for fresh content
        cache_mode=CacheMode.BYPASS
    )

    # Run crawler with proper error handling and retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with AsyncWebCrawler(verbose=True) as crawler:
                result = await crawler.arun(url=url, config=config)

                if not result.success:
                    logger.error(f"Failed to scrape article: {result.error_message}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Wait before retrying
                        continue
                    return ""

                # Handle different response formats
                if result.extracted_content:
                    try:
                        # If using extraction strategy
                        data = json.loads(result.extracted_content)
                        # Combine all text fields
                        content = "\n\n".join([
                            data.get("abstract", ""),
                            data.get("introduction", ""),
                            data.get("conclusion", "")
                        ])
                        return content
                    except json.JSONDecodeError:
                        logger.error("Failed to parse extracted content")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)  # Wait before retrying
                            continue
                        return ""
                elif result.cleaned_html:
                    # If no extraction strategy, use cleaned HTML
                    soup = BeautifulSoup(result.cleaned_html, "html.parser")
                    # Extract main content
                    main_content = soup.select_one("div.article-content")
                    if main_content:
                        return main_content.get_text(separator="\n", strip=True)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Wait before retrying
                        continue
                    return ""
                else:
                    logger.error("No content extracted")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Wait before retrying
                        continue
                    return ""
        except Exception as e:
            logger.error(f"Error scraping article: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # Wait before retrying
                continue
            return ""

    return ""  # Return empty string if all retries failed

# Alternative approach using multiple schemas (based on PROTOTYPE.ipynb)
async def scrape_article_alternative(url: str) -> str:
    """
    Alternative approach to scrape article content using multiple schemas.

    Args:
        url: The URL of the article to scrape

    Returns:
        Cleaned text content of the article
    """
    # Schema for metadata
    metadata_schema = {
        "name": "ArXiv Article Metadata",
        "baseSelector": "dl#articles > dd",
        "fields": [
            {
                "name": "raw_title",
                "selector": "div.list-title",
                "type": "text",
                "default": None
            },
            {
                "name": "authors",
                "selector": "div.list-authors a",
                "type": "list",
                "fields": [
                    {"name": "author_name", "type": "text"}
                ]
            },
            {
                "name": "raw_subjects",
                "selector": "div.list-subjects",
                "type": "text",
                "default": None
            }
        ]
    }

    # Schema for article content
    content_schema = {
        "name": "ArXiv Article Content",
        "baseSelector": "div.article-content",
        "fields": [
            {
                "name": "abstract",
                "selector": "div.abstract",
                "type": "text"
            },
            {
                "name": "introduction",
                "selector": "div.introduction",
                "type": "text"
            },
            {
                "name": "conclusion",
                "selector": "div.conclusion",
                "type": "text"
            }
        ]
    }

    # Create extraction strategies
    metadata_strategy = JsonCssExtractionStrategy(metadata_schema, verbose=True)
    content_strategy = JsonCssExtractionStrategy(content_schema, verbose=True)

    # Configure crawler
    config = CrawlerRunConfig(
        scraping_strategy=LXMLWebScrapingStrategy(),
        extraction_strategy=content_strategy,
        word_count_threshold=10,
        excluded_tags=["nav", "footer", "header"],
        exclude_external_links=True,
        process_iframes=True,
        cache_mode=CacheMode.BYPASS
    )

    try:
        async with AsyncWebCrawler(verbose=True) as crawler:
            # First get metadata
            metadata_result = await crawler.arun(url=url, config=CrawlerRunConfig(
                extraction_strategy=metadata_strategy,
                cache_mode=CacheMode.BYPASS
            ))

            # Then get content
            content_result = await crawler.arun(url=url, config=config)

            if not content_result.success:
                logger.error(f"Failed to scrape article content: {content_result.error_message}")
                return ""

            # Process content
            if content_result.extracted_content:
                try:
                    data = json.loads(content_result.extracted_content)
                    content = "\n\n".join([
                        data.get("abstract", ""),
                        data.get("introduction", ""),
                        data.get("conclusion", "")
                    ])
                    return content
                except json.JSONDecodeError:
                    logger.error("Failed to parse extracted content")
                    return ""
            elif content_result.cleaned_html:
                soup = BeautifulSoup(content_result.cleaned_html, "html.parser")
                main_content = soup.select_one("div.article-content")
                if main_content:
                    return main_content.get_text(separator="\n", strip=True)
                return ""
            else:
                logger.error("No content extracted")
                return ""
    except Exception as e:
        logger.error(f"Error scraping article: {e}")
        return ""

# Batch processing function for parallel scraping
async def scrape_articles_batch(articles: List[dict], max_concurrent: int = 5) -> List[str]:
    """
    Scrape multiple articles in parallel with rate limiting.

    Args:
        articles: List of article dictionaries with html_url field
        max_concurrent: Maximum number of concurrent scraping tasks

    Returns:
        List of article contents
    """
    # Create semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent)

    async def scrape_with_semaphore(article):
        async with semaphore:
            return await scrape_article(article["html_url"])

    # Create tasks for all articles
    tasks = [scrape_with_semaphore(article) for article in articles]

    # Run tasks in parallel and gather results
    contents = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results, handling exceptions
    processed_contents = []
    for i, content in enumerate(contents):
        if isinstance(content, Exception):
            logger.error(f"Error scraping article {i}: {content}")
            processed_contents.append("")
        else:
            processed_contents.append(content)

    return processed_contents
```

### 3. Analysis Tool (in agent_tools.py)

```python
from pydantic_ai import RunContext

async def analyze_article(ctx: RunContext, article_text: str, user_context: UserContext, article_metadata: RankedArticle) -> dict:
    """
    Analyze article content using LLM.

    Args:
        ctx: Run context
        article_text: Scraped article content
        user_context: User context (goals, career)
        article_metadata: Article metadata from ranking agent

    Returns:
        Structured analysis result
    """
    # Format prompt with article content and user context
    prompt = ARTICLE_ANALYSIS_PROMPT.format(
        goals=user_context.goals,
        title=user_context.title,
        name=user_context.name,
        title=article_metadata.title,
        authors=", ".join(article_metadata.authors),
        subject=article_metadata.subject,
        content=article_text
    )

    # Run LLM
    response = await ctx.agent.run(prompt)

    # Parse and structure response
    # This is a simplified example - actual implementation would need more robust parsing
    try:
        # Assuming response is structured as sections
        sections = response.split("\n\n")
        summary = sections[0] if len(sections) > 0 else ""
        importance = sections[1] if len(sections) > 1 else ""
        action = sections[2] if len(sections) > 2 else ""

        return {
            "summary": summary,
            "importance": importance,
            "recommended_action": action
        }
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return {
            "summary": "Failed to generate summary",
            "importance": "Failed to determine importance",
            "recommended_action": "No action recommended"
        }
```

### 4. Prompt Template (in agent_prompts.py)

```python
ARTICLE_ANALYSIS_PROMPT = """
Analyze the following arXiv article for a user with these goals: {goals}
and career: {title} at {name}.

Please provide:
1. A concise summary of the article (2-3 paragraphs)
2. Why this article is important given the user's goals and career
3. A specific action item or next step the user should consider

Article title: {title}
Authors: {authors}
Subject: {subject}

Article content:
{content}
"""
```

### 5. Main Agent Function (in agent.py)

```python
async def analyze_ranked_articles(
    user_info: Dict[str, str],
    ranked_articles: List[RankedArticle],
    top_n: int = 5
) -> List[ArticleAnalysis]:
    """
    Analyze ranked articles for a user.

    Args:
        user_info: User information (name, title, goals)
        ranked_articles: List of ranked articles from the ranking agent
        top_n: Number of top articles to analyze

    Returns:
        List of article analyses
    """
    # Limit to top N articles
    articles_to_analyze = ranked_articles[:top_n]

    # Create user context
    user_context = UserContext(
        name=user_info["name"],
        title=user_info["title"],
        goals=user_info["goals"]
    )

    # Initialize results
    analyses = []

    # Process each article
    for article in articles_to_analyze:
        try:
            # Scrape article content
            article_content = await scrape_article(article.html_url)

            if not article_content:
                logger.warning(f"Failed to scrape content for article: {article.title}")
                continue

            # Analyze article
            analysis_result = await analyze_article(
                article_text=article_content,
                user_context=user_context,
                article_metadata=article
            )

            # Create analysis object
            analysis = ArticleAnalysis(
                title=article.title,
                authors=article.authors,
                subject=article.subject,
                summary=analysis_result["summary"],
                importance=analysis_result["importance"],
                recommended_action=analysis_result["recommended_action"],
                abstract_url=article.abstract_url,
                html_url=article.html_url,
                pdf_url=article.pdf_url,
                relevance_score=article.relevance_score,
                score_reason=article.score_reason
            )

            analyses.append(analysis)

        except Exception as e:
            logger.error(f"Error analyzing article {article.title}: {e}")
            # Continue with next article

    return analyses
```

## Integration with Existing Code

The implementation above integrates with the existing agent.py by:

1. **Extending the existing models** - Adding ArticleAnalysis and BatchAnalysis to complement the existing RankedArticle model
2. **Reusing the user_info structure** - Using the same user_info dictionary format from the ranking agent
3. **Leveraging the same LLM configuration** - Using the same LLM setup for consistency
4. **Following the same async pattern** - Maintaining the async/await pattern for consistency
5. **Using the same logging approach** - Following the existing logging patterns

## Implementation Steps

1. **Use Archon to Design Core Agent Structure**

   - Create the ArticleAnalysis and BatchAnalysis Pydantic models
   - Design the agent class structure
   - Implement dependency injection
   - Set up tool registration

2. **Use Archon to Implement Analysis Tools**

   - Implement the analyze_article tool
   - Set up LLM integration
   - Configure response parsing

3. **Use Archon to Refine Prompt Engineering**

   - Design and implement the ARTICLE_ANALYSIS_PROMPT
   - Set up structured response parsing
   - Implement error handling for LLM responses

4. **Implement Crawl4ai Integration**

   - Create the scraping tools
   - Implement error handling and retry logic
   - Set up batch processing

5. **Test and Refine**
   - Test with sample articles
   - Refine the implementation based on test results
   - Optimize performance

## References

- crawl4ai_examples.md: Examples of using crawl4ai for web scraping
- crawl4ai_selectors.md: Information on content selection and filtering
- PROTOTYPE.ipynb: Example implementation of arXiv article scraping
- agent.py: Existing ranking agent implementation
