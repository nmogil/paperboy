# agent_tools.py

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# Import settings
from .config import settings

from src.models import UserContext, ArticleAnalysis
from src.agent_prompts import ARTICLE_ANALYSIS_PROMPT
import random
import re
from pydantic_ai import Agent
from pydantic import ValidationError
from pydantic_ai import ModelRetry

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
        page_timeout=settings.crawler_timeout # Use setting from config
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
    user_context: UserContext,
    article_metadata: Dict[str, Any]
) -> ArticleAnalysis:
    """
    Analyze an article using the LLM agent.
    Formats the prompt using ARTICLE_ANALYSIS_PROMPT and runs the agent.
    Returns the structured analysis result as an ArticleAnalysis object.

    Args:
        agent: The arxiv_agent instance (or any compatible agent).
        article_content: Main body text.
        user_context: UserContext object (imported type).
        article_metadata: Metadata from RankedArticle (passed as dict).

    Returns:
        ArticleAnalysis object containing analysis and metadata.
    """
    logger.debug(f"Analyzing article: {article_metadata.get('title', 'Unknown Title')}")

    # Prepare the prompt - USE CORRECT KEYWORDS FOR .format()
    analysis_prompt = ARTICLE_ANALYSIS_PROMPT.format(
        user_name=user_context.name,
        user_title=user_context.title,
        user_goals=user_context.goals,
        article_title=article_metadata.get("title", "N/A"),
        article_authors=", ".join(article_metadata.get("authors", [])),
        article_subject=article_metadata.get("subject", "N/A"),
        article_content=article_content[:settings.analysis_content_max_chars] # Use setting
    )

    try:
        # Run the agent with the analysis prompt and ArticleAnalysis as the response model
        res = await agent.run(
            analysis_prompt,
            response_model=ArticleAnalysis # Expect Pydantic AI to handle validation
        )
        
        # Pydantic AI should return the validated model directly in res.output
        if isinstance(res.output, ArticleAnalysis):
             analysis_result = res.output
             # Ensure inherited fields are populated correctly from metadata
             # (agent might not return them, or might hallucinate)
             analysis_result.title = article_metadata.get("title", analysis_result.title)
             analysis_result.authors = article_metadata.get("authors", analysis_result.authors)
             analysis_result.subject = article_metadata.get("subject", analysis_result.subject)
             analysis_result.abstract_url = article_metadata.get("abstract_url", analysis_result.abstract_url)
             analysis_result.html_url = article_metadata.get("html_url", analysis_result.html_url)
             analysis_result.pdf_url = article_metadata.get("pdf_url", analysis_result.pdf_url)
             analysis_result.relevance_score = article_metadata.get("relevance_score", analysis_result.relevance_score)
             analysis_result.score_reason = article_metadata.get("score_reason", analysis_result.score_reason)
             
             logger.info(f"Successfully analyzed article: {analysis_result.title}")
             return analysis_result
        else:
            logger.error(f"Analysis agent output was not ArticleAnalysis as expected. Type: {type(res.output)}. Output: {res.output}")
            # Attempt to parse if it's a string that looks like JSON
            if isinstance(res.output, str):
                try:
                    import json
                    data = json.loads(res.output)
                    if isinstance(data, dict):
                         # Manually create the object, merging metadata
                         merged_data = {**article_metadata, **data}
                         try:
                              validated_analysis = ArticleAnalysis(**merged_data)
                              logger.warning("Fallback JSON parsing succeeded for analysis.")
                              return validated_analysis
                         except ValidationError as val_err_fallback:
                              logger.error(f"Fallback analysis validation failed: {val_err_fallback}")
                              raise TypeError(f"LLM failed to return valid ArticleAnalysis structure. Fallback validation failed.") from val_err_fallback
                    else:
                         raise TypeError(f"LLM failed to return valid ArticleAnalysis structure. Fallback JSON was not a dict.")
                except json.JSONDecodeError as json_err:
                    logger.error(f"Fallback JSON parsing failed for analysis: {json_err}")
                    raise TypeError(f"LLM failed to return valid ArticleAnalysis structure. Output was not valid JSON.") from json_err
                except Exception as parse_err:
                     logger.error(f"Unexpected error during fallback parsing: {parse_err}", exc_info=True)
                     raise TypeError(f"LLM failed to return valid ArticleAnalysis structure. Got unexpected error during fallback.") from parse_err
            else:
                 raise TypeError(f"LLM failed to return valid ArticleAnalysis structure. Got: {type(res.output)}")

    except ValidationError as e:
        # This might catch validation errors during the agent.run call itself
        logger.error(f"Analysis LLM output failed schema validation: {e}", exc_info=True)
        # Let Pydantic AI's retry handle this if configured upstream
        raise ModelRetry(f"Analysis schema validation failed: {e}") from e
    except Exception as e:
        logger.error(f"Error during article analysis for '{article_metadata.get('title')}': {e}", exc_info=True)
        # Return metadata with error analysis if exception occurs
        # This makes the overall process more robust, even if one analysis fails
        return ArticleAnalysis(
            **article_metadata,
            summary=f"Analysis failed: {type(e).__name__} - {str(e)}",
            importance="Unknown",
            recommended_action="Error during analysis"
        ) 