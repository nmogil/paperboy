# agent_tools.py

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from src.agent_prompts import ARTICLE_ANALYSIS_PROMPT
import random
import re
from pydantic_ai import Agent

# Set up logging for this module
logger = logging.getLogger("arxiv_agent_tools")
logger.setLevel(logging.INFO)

# Create console handler with formatting if it doesn't exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Also ensure crawl4ai logger is set to DEBUG
crawl4ai_logger = logging.getLogger("crawl4ai")
crawl4ai_logger.setLevel(logging.DEBUG)
if not crawl4ai_logger.handlers:
    crawl4ai_logger.addHandler(console_handler)

# ========== MODELS ==========
class ArticleContent(BaseModel):
    """Model for article content structure"""
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    keywords: Optional[str] = None
    sections: List[Dict[str, str]] = Field(default_factory=list)

class ArticleAnalysisResult(BaseModel):
    """Model for article analysis results"""
    score: float = Field(ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)
    sentiment: float = Field(ge=-1.0, le=1.0)

# ========== SCHEMA ==========
schema_content = {
    "name": "ArXiv Article Metadata",
    "baseSelector": "dl#articles > dd",  # Target the <dd> element directly
    "fields": [
        {
            "name": "title",
            "selector": "div.list-title",  # Selector relative to <dd>
            "type": "text",
            "default": None
        },
        {
            "name": "authors",
            "selector": "div.list-authors a",  # Selector relative to <dd>
            "type": "list",
            "fields": [
                {"name": "author_name", "type": "text"}
            ]
        },
        {
            "name": "abstract",
            "selector": "div.list-subjects",  # Selector relative to <dd>
            "type": "text",
            "default": None
        },
        {
            "name": "primary_subject",
            "selector": "div.list-subjects span.primary-subject",  # Selector relative to <dd>
            "type": "text",
            "default": None
        },
        {
            "name": "comments",
            "selector": "div.list-comments",  # Selector relative to <dd>
            "type": "text",
            "default": None
        },
        {
            "name": "journal_ref",
            "selector": "div.list-journal-ref",  # Selector relative to <dd>
            "type": "text",
            "default": None
        }
    ]
}

# ========== TOOLS ==========
async def scrape_article(crawler: AsyncWebCrawler, url: str) -> str:
    """
    Scrape content from an arXiv article's HTML page.

    Args:
        crawler: An active AsyncWebCrawler instance.
        url: The HTML or abstract URL of the article.

    Returns:
        String with main article content, or empty on error.
    """
    if "/abs/" in url:
        url = url.replace("/abs/", "/html/")
    c_schema = {
        "name": "ArXiv Article Content",
        "baseSelector": "article.ltx_document",
        "fields": [
            {"name": "title", "selector": "h1.ltx_title.ltx_title_document", "type": "text"},
            {"name": "authors", "selector": "div.ltx_authors span.ltx_personname", "type": "list",
                "fields": [{"name": "author_name", "type": "text"}]},
            {"name": "abstract", "selector": "div.ltx_abstract p.ltx_p", "type": "text"},
        ]
    }
    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(c_schema, verbose=False),
        cache_mode=CacheMode.BYPASS,
        page_timeout=25000 # Consider increasing if timeouts persist
    )

    try:
        # Use the provided crawler instance
        result = await crawler.arun(url=url, config=config)
        if result.success and result.extracted_content:
            import json
            try:
                parsed_data = json.loads(result.extracted_content)
            except json.JSONDecodeError as json_err:
                logger.warning(f"Failed to parse extracted JSON for {url}: {json_err}")
                # Fall through to check cleaned_html
                parsed_data = None
            
            article_dict = None
            if isinstance(parsed_data, list) and parsed_data:
                 # If it's a non-empty list, take the first item
                 if isinstance(parsed_data[0], dict):
                     article_dict = parsed_data[0]
                 else:
                     logger.warning(f"First item in extracted list for {url} is not a dictionary: {type(parsed_data[0])}")
            elif isinstance(parsed_data, dict):
                 # If it's already a dictionary
                 article_dict = parsed_data
            else:
                 logger.warning(f"Parsed extracted content for {url} is not a list or dict: {type(parsed_data)}")
            
            if article_dict:
                 # Now safely access fields using .get()
                 title = article_dict.get('title', '')
                 authors_list = article_dict.get('authors', [])
                 authors_str = ', '.join(a.get('author_name', '') for a in authors_list if isinstance(a, dict))
                 abstract = article_dict.get('abstract', '')
                 
                 content_parts = filter(None, [
                     f"Title: {title}" if title else None,
                     f"Authors: {authors_str}" if authors_str else None,
                     f"Abstract: {abstract}" if abstract else None
                 ])
                 return '\n\n'.join(content_parts) # Use double newline for readability
            # If article_dict is None, fall through to check cleaned_html
                 
        if result.cleaned_html:
            soup = BeautifulSoup(result.cleaned_html, "html.parser")
            title = soup.select_one("h1.title, h1.ltx_title_document")
            abstract = soup.select_one("blockquote.abstract, div.ltx_abstract p.ltx_p")
            return '\n'.join(filter(None, [
                f"Title: {title.get_text(strip=True) if title else ''}",
                f"Abstract: {abstract.get_text(strip=True) if abstract else ''}"
            ]))
        logger.warning(f"Article scrape failed for {url}; no content extracted.")
        return ""
    except Exception as e:
        logger.error(f"Scraping error for {url}: {e}", exc_info=True)
        return ""

async def scrape_articles_batch(urls: List[str], max_concurrent: int = 3) -> Dict[str, str]:
    """
    Scrape multiple articles concurrently with rate limiting.
    
    Args:
        urls: List of article URLs to scrape
        max_concurrent: Maximum number of concurrent scraping tasks
        
    Returns:
        Dictionary mapping URLs to their content
    """
    logger.info(f"Starting batch scraping of {len(urls)} articles with max_concurrent={max_concurrent}")
    
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_with_semaphore(url: str) -> Tuple[str, str]:
        """Scrape a single article with semaphore control"""
        async with semaphore:
            try:
                # Add a small random delay to avoid overwhelming the server
                delay = random.uniform(0.5, 2.0)
                logger.debug(f"Waiting {delay:.2f}s before scraping {url}")
                await asyncio.sleep(delay)
                
                content = await scrape_article(url)
                if not content:
                    logger.warning(f"No content retrieved for {url}")
                    return url, ""
                return url, content
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}", exc_info=True)
                return url, ""
    
    # Create tasks for all URLs
    tasks = [scrape_with_semaphore(url) for url in urls]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Convert results to dictionary
    content_dict = dict(results)
    
    # Log summary
    successful = sum(1 for content in content_dict.values() if content)
    logger.info(f"Batch scraping completed: {successful}/{len(urls)} articles successfully scraped")
    
    return content_dict

async def analyze_article(
    agent: Agent,
    article_content: str,
    user_context: Any,
    article_metadata: Dict[str, Any]
) -> Dict[str, str]:
    """
    Analyze an article using the LLM agent.
    Formats the prompt using ARTICLE_ANALYSIS_PROMPT and runs the agent.
    Returns the structured analysis result.

    Args:
        agent: The arxiv_agent instance (or any compatible agent).
        article_content: Main body text.
        user_context: UserContext object.
        article_metadata: Metadata from RankedArticle.

    Returns:
        Dict with keys: summary, importance, recommended_action
    """
    prompt = ARTICLE_ANALYSIS_PROMPT.format(
        goals=user_context.goals,
        title=user_context.title,
        name=user_context.name,
        article_title=article_metadata["title"],
        authors=", ".join(article_metadata["authors"]),
        subject=article_metadata["subject"],
        content=article_content
    )
    # For direct use: this method expects an agent to be called elsewhere.
    # In usage with the Agent, the agent.run() will be called with this prompt.
    
    # Run the agent with the formatted prompt
    try:
        # The agent.run result might be an AgentRunResult object or just the string
        # We expect a string based on previous tests, but handle both
        response = await agent.run(prompt)
        
        response_text = ""
        if hasattr(response, 'output') and isinstance(response.output, str):
             response_text = response.output
        elif isinstance(response, str):
             response_text = response
        else:
             logger.warning(f"Unexpected response type from agent.run for analysis: {type(response)}")
             # Attempt to stringify
             try:
                 response_text = str(response.output if hasattr(response, 'output') else response)
             except Exception:
                 response_text = ""
        
        if not response_text:
             raise ValueError("Agent returned empty or non-string response for analysis")
             
        # Parse the structured response (Summary\n\nImportance\n\nAction)
        parts = response_text.strip().split('\n\n', 2)
        summary = parts[0].replace("Summary", "").strip() if len(parts) > 0 else ""
        importance = parts[1].replace("Importance", "").strip() if len(parts) > 1 else ""
        action = parts[2].replace("Recommended Action", "").strip() if len(parts) > 2 else ""
        
        return {
            "summary": summary,
            "importance": importance,
            "recommended_action": action
        }
    except Exception as e:
        logger.error(f"Error running LLM for analysis via agent: {e}", exc_info=True)
        return {
            "summary": "Failed to analyze article",
            "importance": "",
            "recommended_action": ""
        } 