# agent_tools.py

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from agent_prompts import ARTICLE_ANALYSIS_PROMPT
import random
import re

# Set up logging for this module
logger = logging.getLogger("article_analyzer")
logger.setLevel(logging.DEBUG)

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
async def scrape_article(url: str) -> str:
    """
    Scrape article content from arXiv URL.
    
    Args:
        url: The URL of the article to scrape
        
    Returns:
        The article content as a string
    """
    logger.info(f"Starting to scrape article from URL: {url}")
    
    # Convert abstract URL to HTML URL if needed
    if "/abs/" in url:
        url = url.replace("/abs/", "/html/")
        logger.info(f"Converted abstract URL to HTML URL: {url}")
    
    # Define the schema for extracting article content
    schema = {
        "name": "ArXiv Article Content",
        "baseSelector": "article.ltx_document",  # Main article container
        "fields": [
            {
                "name": "title",
                "selector": "h1.ltx_title.ltx_title_document",
                "type": "text",
                "default": None
            },
            {
                "name": "authors",
                "selector": "div.ltx_authors span.ltx_personname",
                "type": "list",
                "fields": [
                    {"name": "author_name", "type": "text"}
                ]
            },
            {
                "name": "abstract",
                "selector": "div.ltx_abstract p.ltx_p",
                "type": "text",
                "default": None
            },
            {
                "name": "keywords",
                "selector": "div.ltx_keywords",
                "type": "text",
                "default": None
            },
            {
                "name": "sections",
                "selector": "section.ltx_section",
                "type": "list",
                "fields": [
                    {
                        "name": "heading",
                        "selector": "h2.ltx_title_section",
                        "type": "text"
                    },
                    {
                        "name": "content",
                        "selector": "div.ltx_para p.ltx_p",
                        "type": "text"
                    }
                ]
            }
        ]
    }
    
    logger.debug("Created extraction schema for arXiv article page")
    
    # Configure crawler
    config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(schema, verbose=True),
        word_count_threshold=10,
        excluded_tags=["nav", "footer", "header", "script", "style"],
        exclude_external_links=True,
        process_iframes=False,
        wait_for="article.ltx_document",  # Wait for the article content to load
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000
    )
    
    logger.debug("Configured crawler with waiting for article content")
    
    # Try to scrape the article with retries
    max_retries = 1  # Reduced from 3 to 1 for faster testing
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to scrape article")
            async with AsyncWebCrawler(verbose=True) as crawler:
                logger.debug("Created AsyncWebCrawler instance")
                result = await crawler.arun(url=url, config=config)
                logger.debug(f"Crawler run completed. Success: {result.success}")
                
                if not result.success:
                    logger.error(f"Failed to scrape article (attempt {attempt + 1}): {result.error_message}")
                    if attempt < max_retries - 1:
                        logger.info(f"Waiting 2 seconds before retry {attempt + 2}")
                        await asyncio.sleep(2)
                        continue
                    return ""
                
                if result.extracted_content:
                    try:
                        logger.debug("Attempting to parse extracted content as JSON")
                        data = json.loads(result.extracted_content)
                        content_parts = []
                        
                        # Handle both list and dictionary responses
                        if isinstance(data, list):
                            # Take the first item if it's a list
                            data = data[0] if data else {}
                        
                        # Extract title
                        if data.get("title"):
                            logger.debug(f"Found title: {data['title'][:50]}...")
                            content_parts.append(f"Title: {data['title'].strip()}")
                        else:
                            logger.warning("No title found in extracted content")
                        
                        # Extract authors
                        if data.get("authors"):
                            authors = [author.get("author_name", "") for author in data["authors"]]
                            logger.debug(f"Found {len(authors)} authors")
                            content_parts.append(f"Authors: {', '.join(authors)}")
                        else:
                            logger.warning("No authors found in extracted content")
                        
                        # Extract subjects
                        if data.get("primary_subject"):
                            content_parts.append(f"Primary Subject: {data['primary_subject'].strip()}")
                        if data.get("keywords"):
                            keywords = [kw.strip() for kw in data["keywords"].split(",")]
                            content_parts.append(f"Keywords: {', '.join(keywords)}")
                        
                        # Extract abstract
                        if data.get("abstract"):
                            logger.debug("Found abstract")
                            content_parts.extend(["", "Abstract:", data["abstract"].strip()])
                        else:
                            logger.warning("No abstract found in extracted content")
                        
                        if content_parts:
                            logger.info(f"Successfully extracted {len(content_parts)} content parts")
                            return "\n\n".join(content_parts)
                        else:
                            logger.warning("No content parts extracted from structured data")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse extracted content as JSON: {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        return ""
                
                # If we have cleaned HTML but no extracted content, try to extract manually
                elif result.cleaned_html:
                    logger.info("Attempting manual extraction from cleaned HTML")
                    soup = BeautifulSoup(result.cleaned_html, "html.parser")
                    content_parts = []
                    
                    # Extract title (try both formats)
                    title = soup.select_one("h1.title, h1.ltx_title_document")
                    if title:
                        logger.debug(f"Found title manually: {title.get_text(strip=True)[:50]}...")
                        content_parts.append(f"Title: {title.get_text(strip=True)}")
                    else:
                        logger.warning("No title found in manual extraction")
                    
                    # Extract authors (try both formats)
                    authors = soup.select("div.authors a, div.ltx_authors span.ltx_personname")
                    if authors:
                        logger.debug(f"Found {len(authors)} authors manually")
                        content_parts.append(f"Authors: {', '.join(a.get_text(strip=True) for a in authors)}")
                    else:
                        logger.warning("No authors found in manual extraction")
                    
                    # Extract abstract (try both formats)
                    abstract = soup.select_one("blockquote.abstract, div.ltx_abstract p.ltx_p")
                    if abstract:
                        logger.debug("Found abstract manually")
                        content_parts.extend(["", "Abstract:", abstract.get_text(strip=True)])
                    else:
                        logger.warning("No abstract found in manual extraction")
                    
                    if content_parts:
                        logger.info(f"Successfully extracted {len(content_parts)} content parts manually")
                        return "\n\n".join(content_parts)
                    
                    logger.warning("No content found in manual extraction")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return ""
                else:
                    logger.error("No content extracted (neither structured nor HTML)")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return ""
                
        except Exception as e:
            logger.error(f"Error scraping article (attempt {attempt + 1}): {str(e)}", exc_info=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            return ""
    
    logger.error("All scraping attempts failed")
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

async def analyze_article(ctx: Any, article_text: str, user_context: Any, article_metadata: Dict[str, Any]) -> dict:
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
        article_title=article_metadata["title"],
        authors=", ".join(article_metadata["authors"]),
        subject=article_metadata["subject"],
        content=article_text
    )

    # Run LLM
    try:
        # Check if ctx is an Agent object or has an agent attribute
        if hasattr(ctx, 'run'):
            response = await ctx.run(prompt)
        elif hasattr(ctx, 'agent') and hasattr(ctx.agent, 'run'):
            response = await ctx.agent.run(prompt)
        else:
            logger.error(f"Invalid context object: {ctx}")
            return {
                "summary": "Failed to generate summary - invalid context object",
                "importance": "Failed to determine importance",
                "recommended_action": "No action recommended"
            }

        # Extract response data
        if hasattr(response, 'data'):
            response_text = response.data
        elif isinstance(response, str):
            response_text = response
        else:
            logger.error(f"Unexpected response type: {type(response)}")
            return {
                "summary": "Failed to generate summary - unexpected response type",
                "importance": "Failed to determine importance",
                "recommended_action": "No action recommended"
            }

    except Exception as e:
        logger.error(f"Error running LLM for analysis: {e}")
        return {
            "summary": "Failed to generate summary",
            "importance": "Failed to determine importance",
            "recommended_action": "No action recommended"
        }

    # Parse and structure response
    try:
        # Clean up the response text
        response_text = response_text.strip()
        
        # Try to extract sections using regex patterns
        import re
        
        # Look for explicit section headers
        summary_match = re.search(r'Summary\s*(.*?)(?=\s*Importance|\s*Recommended Action|\Z)', 
                                 response_text, re.DOTALL | re.IGNORECASE)
        importance_match = re.search(r'Importance\s*(.*?)(?=\s*Recommended Action|\Z)', 
                                    response_text, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r'Recommended Action\s*(.*?)(?=\Z)', 
                                response_text, re.DOTALL | re.IGNORECASE)
        
        # If we found explicit headers, use those sections
        if summary_match and importance_match and action_match:
            summary = summary_match.group(1).strip()
            importance = importance_match.group(1).strip()
            action = action_match.group(1).strip()
        else:
            # Fall back to splitting by double newlines
            sections = response_text.split("\n\n")
            
            # Remove any section headers that might be in the text
            cleaned_sections = []
            for section in sections:
                # Remove common section headers
                cleaned = re.sub(r'^(Summary|Importance|Recommended Action):?\s*', '', 
                                section.strip(), flags=re.IGNORECASE)
                cleaned_sections.append(cleaned)
            
            # Assign sections based on position
            summary = cleaned_sections[0] if len(cleaned_sections) > 0 else ""
            importance = cleaned_sections[1] if len(cleaned_sections) > 1 else ""
            action = cleaned_sections[2] if len(cleaned_sections) > 2 else ""
            
            # If we still don't have content, try to extract from the original text
            if not summary and not importance and not action:
                # Last resort: try to extract meaningful content from the original text
                paragraphs = response_text.split("\n")
                if len(paragraphs) >= 3:
                    summary = paragraphs[0]
                    importance = paragraphs[1]
                    action = paragraphs[2]
                elif len(paragraphs) == 2:
                    summary = paragraphs[0]
                    importance = paragraphs[1]
                    action = "No specific action recommended."
                elif len(paragraphs) == 1:
                    summary = paragraphs[0]
                    importance = "Importance not specified."
                    action = "No specific action recommended."
                else:
                    summary = "No summary available."
                    importance = "Importance not specified."
                    action = "No specific action recommended."

        # Clean up any remaining section headers
        summary = re.sub(r'^(Summary|Importance|Recommended Action):?\s*', '', 
                        summary.strip(), flags=re.IGNORECASE)
        importance = re.sub(r'^(Summary|Importance|Recommended Action):?\s*', '', 
                           importance.strip(), flags=re.IGNORECASE)
        action = re.sub(r'^(Summary|Importance|Recommended Action):?\s*', '', 
                       action.strip(), flags=re.IGNORECASE)

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

def _normalize_article_data(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize article data to handle synonym keys and ensure consistent structure.
    
    Args:
        article: Raw article dictionary
        
    Returns:
        Normalized article dictionary
    """
    normalized = article.copy()
    
    # Handle subject/subjects synonym
    if "subjects" in normalized and "subject" not in normalized:
        normalized["subject"] = normalized["subjects"]
    elif "subject" not in normalized:
        normalized["subject"] = "Not specified"
    
    # Ensure authors is a list
    if "authors" not in normalized and "author" in normalized:
        if isinstance(normalized["author"], str):
            normalized["authors"] = [normalized["author"]]
        else:
            normalized["authors"] = normalized["author"]
    elif "authors" not in normalized:
        normalized["authors"] = ["Unknown"]
    
    # Ensure relevance_score is an integer
    if "relevance_score" in normalized and isinstance(normalized["relevance_score"], str):
        try:
            normalized["relevance_score"] = int(normalized["relevance_score"])
        except ValueError:
            normalized["relevance_score"] = 0
    
    # Ensure all URLs are present
    arxiv_id = None
    
    # Try to extract arxiv_id from various sources
    if "arxiv_id" in normalized:
        arxiv_id = normalized["arxiv_id"]
    elif "abstract_url" in normalized:
        match = re.search(r'/abs/([^/]+)', normalized["abstract_url"])
        if match:
            arxiv_id = match.group(1)
    elif "html_url" in normalized:
        match = re.search(r'/html/([^/]+)', normalized["html_url"])
        if match:
            arxiv_id = match.group(1)
    elif "pdf_url" in normalized:
        match = re.search(r'/pdf/([^/]+)', normalized["pdf_url"])
        if match:
            arxiv_id = match.group(1)
    
    # If we found an arxiv_id, use it to construct missing URLs
    if arxiv_id:
        if not normalized.get("abstract_url"):
            normalized["abstract_url"] = f"https://arxiv.org/abs/{arxiv_id}"
        if not normalized.get("html_url"):
            normalized["html_url"] = f"https://arxiv.org/html/{arxiv_id}"
        if not normalized.get("pdf_url"):
            normalized["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    else:
        # If no arxiv_id found, set default URLs
        if not normalized.get("abstract_url"):
            normalized["abstract_url"] = ""
        if not normalized.get("html_url"):
            normalized["html_url"] = ""
        if not normalized.get("pdf_url"):
            normalized["pdf_url"] = ""
    
    # Ensure all required fields are present
    required_fields = ["title", "authors", "subject", "relevance_score",
                      "abstract_url", "html_url", "pdf_url"]
    for field in required_fields:
        if field not in normalized:
            if field == "relevance_score":
                normalized[field] = 0
            else:
                normalized[field] = ""
            logger.warning(f"Missing required field '{field}' in article data")
    
    return normalized 