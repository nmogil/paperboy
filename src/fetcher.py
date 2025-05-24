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
    Fetch and process arXiv CS submissions for a given date.
    
    Args:
        target_date (str): Date in YYYY-MM-DD format
        crawler (Optional[AsyncWebCrawler]): Optional crawler instance to reuse
        
    Returns:
        List[Dict[str, Any]]: List of processed article data
    """
    try:
        target_url = f"https://arxiv.org/catchup/cs/{target_date}"
        logger.info(f"Fetching arXiv CS submissions from {target_url}")
        
        playwright_launch_args = [
            # --- Core Security & Sandboxing ---
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--single-process',  # Use single process for better stability in containers
            '--no-zygote',       # Disable zygote process
            
            # --- GPU & Rendering (optimized for headless) ---
            '--disable-gpu',
            '--disable-gpu-sandbox',
            '--disable-gpu-compositing',
            '--disable-software-rasterizer',
            '--disable-accelerated-2d-canvas',
            '--disable-accelerated-jpeg-decoding',
            '--disable-accelerated-mjpeg-decode',
            '--disable-accelerated-video-decode',
            '--disable-accelerated-video-encode',
            '--disable-features=VizDisplayCompositor,UseSkiaRenderer,DefaultANGLEVulkan,Vulkan,Metal,SkiaGraphite,TranslateUI',
            '--disable-webgl',
            '--disable-webgl2',
            '--use-gl=swiftshader',
            
            # --- Memory & Process Management ---
            '--memory-pressure-off',
            '--max_old_space_size=4096',
            '--disk-cache-size=0',
            '--media-cache-size=0',
            '--aggressive-cache-discard',
            
            # --- Network & Connectivity ---
            '--disable-background-networking',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-features=TranslateUI,BlinkGenPropertyTrees',
            
            # --- Extensions & Apps ---
            '--disable-extensions',
            '--disable-default-apps',
            '--disable-component-extensions-with-background-pages',
            '--disable-component-update',
            
            # --- Automation & Dev Tools ---
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
            '--no-first-run',
            '--no-default-browser-check',
            '--disable-infobars',
            
            # --- Audio & Media ---
            '--mute-audio',
            '--disable-audio-output',
            
            # --- Security Features ---
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--disable-ipc-flooding-protection',
            '--ignore-certificate-errors',
            '--ignore-ssl-errors',
            '--ignore-certificate-errors-spki-list',
            
            # --- DBus & System Services ---
            '--disable-dbus',
            '--no-service-autorun',
            '--disable-breakpad',
            '--disable-client-side-phishing-detection',
            
            # --- UI & Display ---
            '--headless=new',
            '--hide-scrollbars',
            '--force-color-profile=srgb',
            '--window-size=1280,720',
            '--virtual-time-budget=300000',  # 5 minute budget for virtual time
            
            # --- Performance Optimizations ---
            '--disable-hang-monitor',
            '--disable-prompt-on-repost',
            '--disable-popup-blocking',
            '--disable-sync',
            '--disable-translate',
            '--metrics-recording-only',
            '--password-store=basic',
            '--use-mock-keychain',
            '--safebrowsing-disable-auto-update',
            
            # --- Logging (disabled for production) ---
            '--disable-logging',
            '--log-level=3',  # Only fatal errors
        ]

        # Configure browser with updated settings
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True,
            extra_args=playwright_launch_args,
            verbose=False,  # Disable verbose logging for production
            viewport_width=1280,  # Set viewport width
            viewport_height=720,  # Set viewport height
            ignore_https_errors=True,  # Ignore HTTPS errors
            text_mode=True,  # Enable text mode for faster processing
            light_mode=True,  # Enable light mode for better performance
        )
        
        # Create extraction strategies (verbose was already correctly False on these)
        strategy_dd = JsonCssExtractionStrategy(schema_dd, verbose=False)
        strategy_dt = JsonCssExtractionStrategy(schema_dt, verbose=False)
        
        # Configure crawler runs with verbose=False
        config_dd = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=strategy_dd,
            page_timeout=120000,
            verbose=False # Explicitly set verbose
        )
        config_dt = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            extraction_strategy=strategy_dt,
            page_timeout=120000,
            verbose=False # Explicitly set verbose
        )
        
        # If no crawler is provided, create and manage one
        if crawler is None:
            try:
                logger.info("Creating new AsyncWebCrawler instance...")
                async with AsyncWebCrawler(verbose=False, config=browser_config) as new_crawler:
                    logger.info("AsyncWebCrawler initialized successfully")
                    return await _execute_crawls(new_crawler, target_url, config_dd, config_dt)
            except Exception as browser_error:
                logger.error(f"Failed to initialize browser: {browser_error}")
                # Try with minimal configuration as fallback
                logger.info("Attempting fallback with minimal browser configuration...")
                minimal_args = [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--headless',
                    '--disable-gpu',
                    '--single-process',
                    '--no-zygote',
                    '--disable-dbus',
                    '--disable-extensions',
                    '--disable-logging'
                ]
                fallback_config = BrowserConfig(
                    browser_type="chromium",
                    headless=True,
                    extra_args=minimal_args,
                    verbose=False,
                )
                try:
                    async with AsyncWebCrawler(verbose=False, config=fallback_config) as fallback_crawler:
                        logger.info("Fallback AsyncWebCrawler initialized successfully")
                        return await _execute_crawls(fallback_crawler, target_url, config_dd, config_dt)
                except Exception as fallback_error:
                    logger.error(f"Fallback browser initialization also failed: {fallback_error}")
                    raise browser_error  # Re-raise original error
        else:
            # If a crawler is provided, use it directly
            return await _execute_crawls(crawler, target_url, config_dd, config_dt)

    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching articles: {str(e)}", exc_info=True)
        return [] 

async def _execute_crawls(crawler: AsyncWebCrawler, target_url: str, config_dd: CrawlerRunConfig, config_dt: CrawlerRunConfig) -> List[Dict[str, Any]]:
    """Helper function to execute the two crawls (dt and dd) and merge results."""
    logger.info(f"Executing crawls for target URL: {target_url}")

    # --- Diagnostic Navigation ---
    try:
        # logger.info("Attempting to navigate to about:blank for diagnostics...") # Removed
        # diag_config_blank = CrawlerRunConfig(page_timeout=30000, cache_mode=CacheMode.BYPASS) # Removed
        # result_blank = await crawler.arun(url="about:blank", config=diag_config_blank) # Removed
        # if result_blank.success: # Removed
        #     logger.info("Successfully navigated to about:blank.") # Removed
        # else: # Removed
        #     logger.error(f"Failed to navigate to about:blank: {result_blank.error_message or 'No error message'}") # Removed
            # Potentially raise an error or return empty if this is critical # Removed
            # For now, we'll log and continue to see if google.com works # Removed

        logger.info("Attempting to navigate to https://www.google.com for diagnostics...")
        diag_config_google = CrawlerRunConfig(page_timeout=120000, cache_mode=CacheMode.BYPASS) # Longer timeout for external site
        result_google = await crawler.arun(url="https://www.google.com", config=diag_config_google)
        if result_google.success:
            logger.info("Successfully navigated to https://www.google.com.")
        else:
            logger.error(f"Failed to navigate to https://www.google.com: {result_google.error_message or 'No error message'}")
            # Potentially raise an error or return empty

    except Exception as e:
        logger.error(f"Error during diagnostic navigation: {e}", exc_info=True)
        # Decide if this should halt further execution. For now, log and continue.

    # --- Actual Crawls ---
    # Create tasks for both crawls
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