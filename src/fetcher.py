"""
ArXiv CS submissions fetcher module.
Fetches and processes the latest Computer Science articles from arXiv's daily catchup page.
"""

import os
import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# from crawl4ai.config import BrowserConfig # Comment out or remove old import
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, JsonCssExtractionStrategy, BrowserConfig # Try importing BrowserConfig from top-level crawl4ai

logger = logging.getLogger(__name__)

# CSS Selectors for extracting article data
schema_dd = {
    "name": "ArXiv Article Metadata",
    "baseSelector": "dl#articles > dd", # Target the <dd> element directly
    "fields": [
        {
            "name": "raw_title",
            "selector": "div.list-title", # Selector relative to <dd>
            "type": "text",
            "default": None
        },
        {
            "name": "authors",
            "selector": "div.list-authors a", # Selector relative to <dd>
            "type": "list",
            "fields": [
                {"name": "author_name", "type": "text"}
            ]
        },
        {
            "name": "raw_subjects",
            "selector": "div.list-subjects", # Selector relative to <dd>
            "type": "text",
            "default": None
        },
        {
            "name": "primary_subject",
            "selector": "div.list-subjects span.primary-subject", # Selector relative to <dd>
            "type": "text",
            "default": None
        },
        {
            "name": "raw_comments",
            "selector": "div.list-comments", # Selector relative to <dd>
            "type": "text",
            "default": None
        },
        {
            "name": "raw_journal_ref",
            "selector": "div.list-journal-ref", # Selector relative to <dd>
            "type": "text",
            "default": None
        }
    ]
}

# Schema for extracting IDs and Links from <dt>
schema_dt = {
    "name": "ArXiv Article Links",
    "baseSelector": "dl#articles > dt", # Target the <dt> element
    "fields": [
        {
            "name": "arxiv_id_from_href",
            "selector": "a[title='Abstract']",
            "type": "attribute",
            "attribute": "href",
            "regex": r"/abs/([^/]+)",  # More permissive regex to catch various ID formats
            "default": None
        },
        {
            "name": "arxiv_id_from_text",
            "selector": "a[title='Abstract']",
            "type": "text",
            "regex": r"arXiv:([^\s]+)",  # More permissive regex
            "default": None
        },
        {
            "name": "abstract_url_rel",
            "selector": "a[title='Abstract']",
            "type": "attribute",
            "attribute": "href",
            "default": None
        },
        {
            "name": "pdf_url_rel",
            "selector": "a[title='Download PDF']",
            "type": "attribute",
            "attribute": "href",
            "default": None
        },
        {
            "name": "html_url",
            "selector": "a[title='View HTML']",
            "type": "attribute",
            "attribute": "href",
            "default": None
        }
    ]
}

def _process_and_merge_articles(dt_data_list: List[Dict[str, Any]], dd_data_list: List[Dict[str, Any]], base_url: str) -> List[Dict[str, Any]]:
    """
    Process and merge the extracted dt (titles) and dd (details) data into a list of article dictionaries.
    
    Args:
        dt_data_list: List of dictionaries containing title data
        dd_data_list: List of dictionaries containing details data
        base_url: Base URL for arxiv links
    
    Returns:
        List of processed article dictionaries
    """
    merged_articles = []
    num_items = min(len(dt_data_list), len(dd_data_list))
    logger.info(f"Attempting to merge {num_items} dt/dd pairs.")
    
    id_extraction_failures = 0
    
    for i in range(num_items):
        dt_data = dt_data_list[i]
        dd_data = dd_data_list[i]
        processed = {}

        # --- Improved ArXiv ID Handling ---
        id_from_href_raw = dt_data.get('arxiv_id_from_href')
        id_from_text_raw = dt_data.get('arxiv_id_from_text')
        canonical_id = None
        
        # Try to extract the ID using various methods
        for id_source in [id_from_href_raw, id_from_text_raw]:
            if not id_source:
                continue
                
            # Try standard format YYMM.NNNNN or YYMM.NNNNNvN
            std_match = re.search(r"(\d{4}\.\d{5}(?:v\d+)?)", id_source)
            if std_match:
                canonical_id = std_match.group(1)
                break
                
            # Try alternate format for older papers
            alt_match = re.search(r"([a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", id_source)
            if alt_match:
                canonical_id = alt_match.group(1)
                break
                
            # Last resort: try to extract any sequence that might be an ID
            last_resort_match = re.search(r"([^/\s]+)(?:v\d+)?$", id_source)
            if last_resort_match:
                canonical_id = last_resort_match.group(1)
                break

        if not canonical_id:
            id_extraction_failures += 1
            context_hint = dd_data.get('raw_title', 'N/A')[:50] or dt_data.get('abstract_url_rel', 'N/A') or f'Entry {i+1}'
            logger.warning(f"Failed to extract valid canonical arXiv ID. Context hint: {context_hint}")
            logger.warning(f"  - href source: {id_from_href_raw}")
            logger.warning(f"  - text source: {id_from_text_raw}")
            continue

        processed['arxiv_id'] = f"arXiv:{canonical_id}"

        # --- URL Construction (from dt_data) ---
        abstract_rel = dt_data.get('abstract_url_rel')
        processed['abstract_url'] = f"{base_url}{abstract_rel}" if abstract_rel else f"{base_url}/abs/{canonical_id}"

        pdf_rel = dt_data.get('pdf_url_rel')
        if pdf_rel:
            processed['pdf_url'] = f"{base_url}{pdf_rel}"

        html_url = dt_data.get('html_url')
        if html_url:
            if isinstance(html_url, str) and not html_url.startswith(('http://', 'https://')):
                if not html_url.startswith('/'):
                    html_url = '/' + html_url
                processed['html_url'] = f"{base_url}{html_url}"
            else:
                processed['html_url'] = html_url

        # --- Text Cleaning (from dd_data) ---
        raw_title = dd_data.get('raw_title', '')
        processed['title'] = re.sub(r'^Title:\s*', '', raw_title, flags=re.IGNORECASE).strip() if raw_title else None

        raw_subjects = dd_data.get('raw_subjects', '')
        processed['subjects'] = re.sub(r'^Subjects:\s*', '', raw_subjects, flags=re.IGNORECASE).strip() if raw_subjects else None

        raw_comments = dd_data.get('raw_comments', '')
        processed['comments'] = re.sub(r'^Comments:\s*', '', raw_comments, flags=re.IGNORECASE).strip() if raw_comments else None

        raw_journal_ref = dd_data.get('raw_journal_ref', '')
        processed['journal_ref'] = re.sub(r'^Journal-ref:\s*', '', raw_journal_ref, flags=re.IGNORECASE).strip() if raw_journal_ref else None

        processed['primary_subject'] = dd_data.get('primary_subject')

        # --- Author List (from dd_data) ---
        author_list = dd_data.get('authors', [])
        processed['authors'] = [auth.get('author_name') for auth in author_list if auth.get('author_name')]

        # --- Ensure essential keys exist ---
        processed.setdefault('title', None)
        processed.setdefault('subjects', None)
        processed.setdefault('primary_subject', None)
        processed.setdefault('comments', None)
        processed.setdefault('journal_ref', None)
        processed.setdefault('authors', [])
        processed.setdefault('pdf_url', None)
        processed.setdefault('html_url', None)

        merged_articles.append(processed)
    
    # Print summary of failures
    if id_extraction_failures:
        logger.warning(f"Total ID extraction failures: {id_extraction_failures} out of {num_items} ({id_extraction_failures/num_items*100:.1f}%)")
    
    return merged_articles

async def fetch_arxiv_cs_submissions(target_date: str, crawler: Optional[AsyncWebCrawler] = None) -> List[Dict[str, Any]]:
    """
    Fetch Computer Science submissions from arXiv for a specific date.
    
    Args:
        target_date: Date string in YYYY-MM-DD format
        crawler: Optional AsyncWebCrawler instance to use. If None, a new one will be created.
    
    Returns:
        List of dictionaries containing article information
    """
    try:
        # Validate date format
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Invalid date format for {target_date}. Expected YYYY-MM-DD")
            return []

        target_url = f"https://arxiv.org/catchup/cs/{target_date}"
        logger.info(f"Fetching arXiv CS submissions from {target_url}")
        
        playwright_launch_args = [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--disable-software-rasterizer',
            '--disable-background-networking',
            '--disable-default-apps',
            '--disable-extensions',
            '--disable-sync',
            '--disable-translate',
            '--metrics-recording-only',
            '--mute-audio',
            '--no-first-run',
            '--safebrowsing-disable-auto-update',
            '--disable-dbus',
            '--no-zygote'
        ]
        browser_config = BrowserConfig(extra_args=playwright_launch_args)
        
        # Create extraction strategies
        strategy_dd = JsonCssExtractionStrategy(schema_dd, verbose=False)
        strategy_dt = JsonCssExtractionStrategy(schema_dt, verbose=False)
        
        # Configure crawler runs
        config_dd = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=strategy_dd
        )
        config_dt = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=strategy_dt
        )
        
        # Execute crawls
        if crawler is None:
            # Create a new crawler if none provided
            async with AsyncWebCrawler(verbose=False, config=browser_config) as new_crawler:
                return await _execute_crawls(new_crawler, target_url, config_dd, config_dt)
        else:
            # Use the provided crawler
            return await _execute_crawls(crawler, target_url, config_dd, config_dt)
            
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching articles: {str(e)}", exc_info=True)
        return [] 

async def _execute_crawls(crawler: AsyncWebCrawler, target_url: str, config_dd: CrawlerRunConfig, config_dt: CrawlerRunConfig) -> List[Dict[str, Any]]:
    """Helper function to execute the crawls with a given crawler instance."""
    logger.info("Running crawler for <dd> elements...")
    result_dd = await crawler.arun(url=target_url, config=config_dd)
    if not result_dd.success or not result_dd.extracted_content:
        logger.error(f"Failed to fetch details from {target_url}")
        return []
    
    logger.info("Running crawler for <dt> elements...")
    result_dt = await crawler.arun(url=target_url, config=config_dt)
    if not result_dt.success or not result_dt.extracted_content:
        logger.error(f"Failed to fetch titles from {target_url}")
        return []
    
    try:
        dd_data_list = json.loads(result_dd.extracted_content)
        dt_data_list = json.loads(result_dt.extracted_content)
        logger.info(f"Successfully extracted {len(dd_data_list)} <dd> entries and {len(dt_data_list)} <dt> entries.")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        return []
    
    # Process and merge the data
    articles = _process_and_merge_articles(dt_data_list, dd_data_list, base_url="https://arxiv.org")
    logger.info(f"Successfully processed {len(articles)} articles")
    return articles 