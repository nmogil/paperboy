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
        try:
            datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Invalid date format for {target_date}. Expected YYYY-MM-DD")
            return []

        target_url = f"https://arxiv.org/catchup/cs/{target_date}"
        logger.info(f"Fetching arXiv CS submissions from {target_url}")
        
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
                await asyncio.sleep(2 ** attempt)
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        dl_articles = soup.find('dl', {'id': 'articles'})
        if not dl_articles:
            logger.error(f"Could not find article list on {url}")
            return []
        
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
        
        abstract_link = dt_element.find('a', {'title': 'Abstract'})
        if abstract_link:
            href = abstract_link.get('href', '')
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
        
        article.setdefault('title', None)
        article.setdefault('subjects', None)
        article.setdefault('primary_subject', None)
        article.setdefault('comments', None)
        article.setdefault('journal_ref', None)
        article.setdefault('authors', [])
        article.setdefault('pdf_url', None)
        article.setdefault('html_url', None)
        
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

class ArxivFetcher:
    """Arxiv fetcher with connection pooling and improved performance."""
    
    def __init__(self):
        # Optimized HTTP client configuration for Cloud Run
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,      # Quick connection timeout
                read=25.0,        # Allow time for large responses
                write=10.0,       # Quick write timeout
                pool=2.0          # Quick pool timeout
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=40,
                keepalive_expiry=30.0
            ),
            http2=True,  # Enable HTTP/2 for better performance
        )
        self.semaphore = asyncio.Semaphore(5)
    
    async def fetch_arxiv_papers(self, category: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv API for a specific category."""
        try:
            api_url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results={max_results}&sortBy=lastUpdatedDate&sortOrder=descending"
            
            async with self.semaphore:
                response = await self.client.get(api_url)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'xml')
            entries = soup.find_all('entry')
            
            articles = []
            for entry in entries:
                article = self._parse_api_entry(entry)
                if article:
                    articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles for category {category}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching papers for category {category}: {e}")
            return []
    
    def _parse_api_entry(self, entry) -> Optional[Dict[str, Any]]:
        """Parse a single entry from arXiv API response."""
        try:
            article = {}
            
            title_elem = entry.find('title')
            if title_elem:
                article['title'] = title_elem.get_text(strip=True)
            
            id_elem = entry.find('id')
            if id_elem:
                article['url'] = id_elem.get_text(strip=True)
                article['abstract_url'] = article['url']
                
                id_match = re.search(r'arxiv.org/abs/([^v]+)', article['url'])
                if id_match:
                    arxiv_id = id_match.group(1)
                    article['pdf_url'] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            authors = []
            for author in entry.find_all('author'):
                name = author.find('name')
                if name:
                    authors.append(name.get_text(strip=True))
            article['authors'] = authors
            
            categories = []
            primary_category = entry.find('arxiv:primary_category')
            if primary_category:
                article['subject'] = primary_category.get('term', '')
                categories.append(article['subject'])
            
            for category in entry.find_all('category'):
                term = category.get('term', '')
                if term and term not in categories:
                    categories.append(term)
            article['categories'] = categories
            
            summary_elem = entry.find('summary')
            if summary_elem:
                article['abstract'] = summary_elem.get_text(strip=True)
            
            if article.get('title') and article.get('url'):
                return article
            return None
            
        except Exception as e:
            logger.error(f"Error parsing API entry: {e}")
            return None
    
    async def fetch_article_content(self, url: str) -> str:
        """Fetch and extract article content with timeout."""
        try:
            async with self.semaphore:
                response = await self.client.get(url, timeout=10.0)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            content_areas = [
                soup.find('div', {'class': 'abstract'}),
                soup.find('blockquote', {'class': 'abstract'}),
                soup.find('div', {'id': 'abs'}),
                soup.find('div', {'class': 'paper-content'}),
                soup.find('article'),
                soup.find('main')
            ]
            
            for area in content_areas:
                if area:
                    text = area.get_text(separator='\n', strip=True)
                    if len(text) > 8000:
                        text = text[:8000] + "..."
                    return text
            
            if soup.body:
                text = soup.body.get_text(separator='\n', strip=True)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                text = '\n'.join(lines[:100])
                return text[:8000] if len(text) > 8000 else text
            
            return "No content could be extracted from the page."
            
        except Exception as e:
            logger.error(f"Failed to fetch article content from {url}: {e}")
            return ""
    
    async def close(self):
        """Clean up client connections."""
        await self.client.aclose()