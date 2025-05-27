# agent_tools_lightweight.py

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import httpx
import random
import re
from pydantic_ai import Agent
from pydantic import ValidationError
from pydantic_ai import ModelRetry

# Import settings
from .config import settings

from src.models import UserContext, ArticleAnalysis
from src.agent_prompts import ARTICLE_ANALYSIS_PROMPT

# Set up logging for this module
logger = logging.getLogger("arxiv_agent_tools_lightweight")
logger.setLevel(logging.INFO)

# Create console handler with formatting if it doesn't exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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

# ========== TOOLS ==========
async def scrape_article(client: httpx.AsyncClient, url: str) -> str:
    """
    Scrape content from an arXiv article's HTML page.

    Args:
        client: An active httpx.AsyncClient instance.
        url: The HTML or abstract URL of the article.

    Returns:
        String with main article content, or empty on error.
    """
    # Convert abstract URL to HTML URL if needed
    if "/abs/" in url:
        url = url.replace("/abs/", "/html/")
    
    try:
        # Fetch with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                break
            except httpx.HTTPError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                    return ""
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to extract from structured HTML article
        article_elem = soup.find('article', {'class': 'ltx_document'})
        if article_elem:
            # Extract title
            title_elem = article_elem.find('h1', {'class': ['ltx_title', 'ltx_title_document']})
            title = title_elem.get_text(strip=True) if title_elem else ''
            
            # Extract authors
            authors = []
            authors_div = article_elem.find('div', {'class': 'ltx_authors'})
            if authors_div:
                for author_span in authors_div.find_all('span', {'class': 'ltx_personname'}):
                    authors.append(author_span.get_text(strip=True))
            
            # Extract abstract
            abstract = ''
            abstract_div = article_elem.find('div', {'class': 'ltx_abstract'})
            if abstract_div:
                abstract_p = abstract_div.find('p', {'class': 'ltx_p'})
                if abstract_p:
                    abstract = abstract_p.get_text(strip=True)
            
            # Format content
            content_parts = []
            if title:
                content_parts.append(f"Title: {title}")
            if authors:
                content_parts.append(f"Authors: {', '.join(authors)}")
            if abstract:
                content_parts.append(f"Abstract: {abstract}")
            
            if content_parts:
                return '\n\n'.join(content_parts)
        
        # Fallback: Try to extract from abstract page if HTML page fails
        if "/html/" in url:
            abs_url = url.replace("/html/", "/abs/")
            logger.info(f"HTML extraction failed, trying abstract page: {abs_url}")
            
            try:
                abs_response = await client.get(abs_url, timeout=30.0)
                abs_response.raise_for_status()
                abs_soup = BeautifulSoup(abs_response.text, 'html.parser')
                
                # Extract from abstract page
                title_elem = abs_soup.find('h1', {'class': 'title'})
                title = title_elem.get_text(strip=True).replace('Title:', '').strip() if title_elem else ''
                
                abstract_elem = abs_soup.find('blockquote', {'class': 'abstract'})
                abstract = abstract_elem.get_text(strip=True).replace('Abstract:', '').strip() if abstract_elem else ''
                
                authors_elem = abs_soup.find('div', {'class': 'authors'})
                authors = authors_elem.get_text(strip=True).replace('Authors:', '').strip() if authors_elem else ''
                
                content_parts = []
                if title:
                    content_parts.append(f"Title: {title}")
                if authors:
                    content_parts.append(f"Authors: {authors}")
                if abstract:
                    content_parts.append(f"Abstract: {abstract}")
                
                if content_parts:
                    return '\n\n'.join(content_parts)
            except Exception as e:
                logger.error(f"Fallback to abstract page failed: {e}")
        
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
    
    async def scrape_with_semaphore(client: httpx.AsyncClient, url: str) -> Tuple[str, str]:
        """Scrape a single article with semaphore control"""
        async with semaphore:
            try:
                # Add a small random delay to avoid overwhelming the server
                delay = random.uniform(0.5, 2.0)
                logger.debug(f"Waiting {delay:.2f}s before scraping {url}")
                await asyncio.sleep(delay)
                
                content = await scrape_article(client, url)
                if not content:
                    logger.warning(f"No content retrieved for {url}")
                    return url, ""
                return url, content
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}", exc_info=True)
                return url, ""
    
    # Use a single client for all requests
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        # Create tasks for all URLs
        tasks = [scrape_with_semaphore(client, url) for url in urls]
        
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

# Maintain compatibility with original agent_tools
extract_article_content = scrape_article