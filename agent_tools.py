# agent_tools.py

import json
import logging
import asyncio
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from agent_prompts import ARTICLE_ANALYSIS_PROMPT

logger = logging.getLogger("article_analyzer")

# Schema for article content based on actual arXiv HTML structure
schema_content = {
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

async def scrape_article(url: str) -> str:
    """
    Scrape article content using crawl4ai.

    Args:
        url: The URL of the article to scrape

    Returns:
        Cleaned text content of the article
    """
    # Create extraction strategy
    extraction_strategy = JsonCssExtractionStrategy(schema_content, verbose=True)

    # Configure crawler
    config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        word_count_threshold=10,
        excluded_tags=["nav", "footer", "header", "script", "style"],
        exclude_external_links=True,
        process_iframes=True,
        wait_for="css:article.ltx_document",
        cache_mode=CacheMode.BYPASS
    )

    # Run crawler with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with AsyncWebCrawler(verbose=True) as crawler:
                result = await crawler.arun(url=url, config=config)

                if not result.success:
                    logger.error(f"Failed to scrape article: {result.error_message}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return ""

                if result.extracted_content:
                    try:
                        data = json.loads(result.extracted_content)
                        # Build a structured article text
                        content_parts = []
                        
                        # Check if data is a dictionary or a list
                        if isinstance(data, dict):
                            # Add title
                            if data.get("title"):
                                content_parts.append(f"Title: {data['title'].strip()}")
                            
                            # Add authors
                            if data.get("authors"):
                                authors = [author.get("author_name", "") for author in data["authors"]]
                                content_parts.append(f"Authors: {', '.join(authors)}")
                            
                            # Add abstract
                            if data.get("abstract"):
                                content_parts.extend(["", "Abstract:", data["abstract"].strip()])
                            
                            # Add keywords
                            if data.get("keywords"):
                                content_parts.extend(["", "Keywords:", data["keywords"].strip()])
                            
                            # Add sections
                            if data.get("sections"):
                                for section in data["sections"]:
                                    heading = section.get("heading", "").strip()
                                    content = section.get("content", "").strip()
                                    if heading and content:
                                        content_parts.extend(["", heading, content])
                        elif isinstance(data, list):
                            # Handle case where data is a list
                            logger.warning("Extracted content is a list, not a dictionary")
                            # Try to extract meaningful content from the list
                            for item in data:
                                if isinstance(item, dict):
                                    if item.get("title"):
                                        content_parts.append(f"Title: {item['title'].strip()}")
                                    if item.get("abstract"):
                                        content_parts.extend(["", "Abstract:", item["abstract"].strip()])
                        
                        if content_parts:
                            return "\n\n".join(content_parts)
                        else:
                            logger.warning("No content parts extracted from structured data")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse extracted content: {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return ""
                elif result.cleaned_html:
                    # Fallback to BeautifulSoup if structured extraction fails
                    soup = BeautifulSoup(result.cleaned_html, "html.parser")
                    content_parts = []
                    
                    # Try to get title
                    title = soup.select_one("h1.ltx_title.ltx_title_document")
                    if title:
                        content_parts.append(f"Title: {title.get_text(strip=True)}")
                    
                    # Try to get authors
                    authors = soup.select("div.ltx_authors span.ltx_personname")
                    if authors:
                        content_parts.append(f"Authors: {', '.join(a.get_text(strip=True) for a in authors)}")
                    
                    # Try to get abstract
                    abstract = soup.select_one("div.ltx_abstract p.ltx_p")
                    if abstract:
                        content_parts.extend(["", "Abstract:", abstract.get_text(strip=True)])
                    
                    # Try to get keywords
                    keywords = soup.select_one("div.ltx_keywords")
                    if keywords:
                        content_parts.extend(["", "Keywords:", keywords.get_text(strip=True)])
                    
                    # Try to get sections
                    sections = soup.select("section.ltx_section")
                    for section in sections:
                        heading = section.select_one("h2.ltx_title_section")
                        content = section.select("div.ltx_para p.ltx_p")
                        if heading and content:
                            content_parts.extend([
                                "",
                                heading.get_text(strip=True),
                                "\n".join(p.get_text(strip=True) for p in content)
                            ])
                    
                    if content_parts:
                        return "\n\n".join(content_parts)
                    
                    # Last resort: try to get any meaningful content
                    body = soup.select_one("article.ltx_document")
                    if body:
                        return body.get_text(separator="\n", strip=True)
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return ""
                else:
                    logger.error("No content extracted")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return ""
        except Exception as e:
            logger.error(f"Error scraping article: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            return ""

    return ""

async def scrape_articles_batch(articles: List[Dict[str, Any]], max_concurrent: int = 5) -> List[str]:
    """
    Scrape multiple articles in parallel with rate limiting.

    Args:
        articles: List of article dictionaries with html_url field
        max_concurrent: Maximum number of concurrent scraping tasks

    Returns:
        List of article contents
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def scrape_with_semaphore(article):
        async with semaphore:
            return await scrape_article(article["html_url"])

    tasks = [scrape_with_semaphore(article) for article in articles]
    contents = await asyncio.gather(*tasks, return_exceptions=True)

    processed_contents = []
    for i, content in enumerate(contents):
        if isinstance(content, Exception):
            logger.error(f"Error scraping article {i}: {content}")
            processed_contents.append("")
        else:
            processed_contents.append(content)

    return processed_contents

async def analyze_article(ctx: Any, article_text: str, user_context: Any, article_metadata: Any) -> dict:
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
        article_title=article_metadata.title,
        authors=", ".join(article_metadata.authors),
        subject=article_metadata.subject,
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
        # Assuming response is structured as sections
        sections = response_text.split("\n\n")
        summary = sections[0] if len(sections) > 0 else ""
        importance = sections[1] if len(sections) > 1 else ""
        action = sections[2] if len(sections) > 2 else ""

        return {
            "summary": summary.strip(),
            "importance": importance.strip(),
            "recommended_action": action.strip()
        }
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}")
        return {
            "summary": "Failed to generate summary",
            "importance": "Failed to determine importance",
            "recommended_action": "No action recommended"
        } 