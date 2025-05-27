"""
ArXiv CS submissions fetcher module - Lightweight version.
Fetches and processes the latest Computer Science articles from arXiv's daily catchup page.
Uses httpx and BeautifulSoup instead of Playwright for minimal resource usage.
"""

import os
import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

async def fetch_arxiv_cs_submissions(target_date: str, client: Optional[httpx.AsyncClient] = None) -> List[Dict[str, Any]]:
    """
    Fetch Computer Science submissions from arXiv for a specific date.
    
    Args:
        target_date: Date string in YYYY-MM-DD format
        client: Optional httpx.AsyncClient instance to use. If None, a new one will be created.
    
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
        
        # Create client if not provided
        if client is None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as new_client:
                return await _fetch_and_parse(new_client, target_url)
        else:
            return await _fetch_and_parse(client, target_url)
            
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching articles: {str(e)}", exc_info=True)
        return []

async def _fetch_and_parse(client: httpx.AsyncClient, url: str) -> List[Dict[str, Any]]:
    """Helper function to fetch and parse arXiv page."""
    try:
        # Fetch the page with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.get(url)
                response.raise_for_status()
                break
            except httpx.HTTPError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                    return []
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the article list
        dl_articles = soup.find('dl', {'id': 'articles'})
        if not dl_articles:
            logger.error(f"Could not find article list on {url}")
            return []
        
        # Extract dt (links) and dd (metadata) elements
        dt_elements = dl_articles.find_all('dt')
        dd_elements = dl_articles.find_all('dd')
        
        if len(dt_elements) != len(dd_elements):
            logger.warning(f"Mismatch between dt ({len(dt_elements)}) and dd ({len(dd_elements)}) elements")
        
        articles = []
        for dt, dd in zip(dt_elements, dd_elements):
            article = _parse_article(dt, dd)
            if article:
                articles.append(article)
        
        logger.info(f"Successfully parsed {len(articles)} articles from {url}")
        return articles
        
    except Exception as e:
        logger.error(f"Error parsing arXiv page: {e}", exc_info=True)
        return []

def _parse_article(dt_element, dd_element) -> Optional[Dict[str, Any]]:
    """Parse a single article from dt and dd elements."""
    try:
        article = {}
        
        # Extract from dt element (links)
        abstract_link = dt_element.find('a', {'title': 'Abstract'})
        if abstract_link:
            href = abstract_link.get('href', '')
            # Extract arXiv ID from href
            id_match = re.search(r'/abs/([^/]+)', href)
            if id_match:
                arxiv_id = id_match.group(1)
                article['arxiv_id'] = f"arXiv:{arxiv_id}"
                article['abstract_url'] = f"https://arxiv.org{href}"
            
        pdf_link = dt_element.find('a', {'title': 'Download PDF'})
        if pdf_link:
            article['pdf_url'] = f"https://arxiv.org{pdf_link.get('href', '')}"
        
        html_link = dt_element.find('a', {'title': 'View HTML'})
        if html_link:
            article['html_url'] = f"https://arxiv.org{html_link.get('href', '')}"
        
        # Extract from dd element (metadata)
        title_div = dd_element.find('div', {'class': 'list-title'})
        if title_div:
            title = title_div.get_text(strip=True)
            article['title'] = re.sub(r'^Title:\s*', '', title, flags=re.IGNORECASE)
        
        authors_div = dd_element.find('div', {'class': 'list-authors'})
        if authors_div:
            authors = []
            for author_link in authors_div.find_all('a'):
                author_name = author_link.get_text(strip=True)
                if author_name:
                    authors.append(author_name)
            article['authors'] = authors
        
        subjects_div = dd_element.find('div', {'class': 'list-subjects'})
        if subjects_div:
            subjects = subjects_div.get_text(strip=True)
            article['subjects'] = re.sub(r'^Subjects:\s*', '', subjects, flags=re.IGNORECASE)
            
            # Extract primary subject
            primary_span = subjects_div.find('span', {'class': 'primary-subject'})
            if primary_span:
                article['primary_subject'] = primary_span.get_text(strip=True)
        
        comments_div = dd_element.find('div', {'class': 'list-comments'})
        if comments_div:
            comments = comments_div.get_text(strip=True)
            article['comments'] = re.sub(r'^Comments:\s*', '', comments, flags=re.IGNORECASE)
        
        journal_ref_div = dd_element.find('div', {'class': 'list-journal-ref'})
        if journal_ref_div:
            journal_ref = journal_ref_div.get_text(strip=True)
            article['journal_ref'] = re.sub(r'^Journal-ref:\s*', '', journal_ref, flags=re.IGNORECASE)
        
        # Ensure all expected fields exist
        article.setdefault('title', None)
        article.setdefault('subjects', None)
        article.setdefault('primary_subject', None)
        article.setdefault('comments', None)
        article.setdefault('journal_ref', None)
        article.setdefault('authors', [])
        article.setdefault('pdf_url', None)
        article.setdefault('html_url', None)
        
        # Only return if we have essential fields
        if article.get('arxiv_id') and article.get('title'):
            return article
        else:
            logger.warning("Article missing essential fields (arxiv_id or title)")
            return None
            
    except Exception as e:
        logger.error(f"Error parsing article: {e}", exc_info=True)
        return None

# Maintain compatibility with original fetcher
fetch_arxiv_page = fetch_arxiv_cs_submissions