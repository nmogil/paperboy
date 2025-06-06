import asyncio
from typing import List, Dict, Any, Optional
import httpx
import logfire
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .metrics import NewsMetrics

class ContentExtractionError(Exception):
    """Content extraction failed"""
    pass

class TavilyExtractor:
    """Extracts full content with rate limiting and batch processing."""
    
    def __init__(self):
        if not settings.tavily_api_key:
            raise ValueError("Tavily API key not configured")
        
        self.api_key = settings.tavily_api_key
        self.base_url = "https://api.tavily.com/extract"
        self.semaphore = asyncio.Semaphore(settings.extract_max_concurrent)
        self._request_count = 0
        self._request_limit = 100  # Assumed daily limit
        
        # Optimized HTTP client configuration for Cloud Run
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,      # Quick connection timeout
                read=settings.extract_timeout or 25.0,  # Use extract_timeout if set
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
    
    @NewsMetrics.track_content_extraction
    async def extract_articles(
        self,
        articles: List[Dict[str, Any]],
        priority_indices: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract full content with intelligent prioritization.
        
        Args:
            articles: List of article dicts with 'url' field
            priority_indices: Indices of high-priority articles to extract first
            
        Returns:
            Articles with added 'full_content' field
        """
        if self._request_count >= self._request_limit:
            logfire.warn("Tavily request limit reached, skipping extraction")
            return self._add_preview_content(articles)
        
        # Prioritize extraction
        if priority_indices:
            priority_articles = [articles[i] for i in priority_indices if i < len(articles)]
            other_articles = [a for i, a in enumerate(articles) if i not in priority_indices]
            ordered_articles = priority_articles + other_articles
        else:
            ordered_articles = articles
        
        # Extract up to limit
        max_to_extract = min(
            settings.news_max_extract,
            self._request_limit - self._request_count,
            len(ordered_articles)
        )
        
        to_extract = ordered_articles[:max_to_extract]
        not_extracted = ordered_articles[max_to_extract:]
        
        # Batch extraction
        extracted = await self._batch_extract(to_extract)
        
        # Add preview content for non-extracted
        not_extracted = self._add_preview_content(not_extracted)
        
        # Maintain original order
        result = []
        extracted_map = {a['url']: a for a in extracted}
        not_extracted_map = {a['url']: a for a in not_extracted}
        
        for article in articles:
            url = article['url']
            if url in extracted_map:
                result.append(extracted_map[url])
            elif url in not_extracted_map:
                result.append(not_extracted_map[url])
            else:
                result.append(article)
        
        return result
    
    async def _batch_extract(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract articles in batches."""
        tasks = []
        for article in articles:
            task = self._extract_with_semaphore(article)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        extracted_articles = []
        for article, result in zip(articles, results):
            if isinstance(result, Exception):
                logfire.error("Failed to extract content", extra={"url": article['url'], "error": str(result)})
                article['full_content'] = article.get('content_preview', '')
                article['extraction_error'] = str(result)
            else:
                article['full_content'] = result
                article['extraction_success'] = True
            extracted_articles.append(article)
        
        successful = sum(1 for a in extracted_articles if a.get('extraction_success'))
        self._request_count += successful
        logfire.info("Content extraction completed", extra={"successful": successful, "total": len(articles)})
        
        return extracted_articles
    
    async def _extract_with_semaphore(self, article: Dict[str, Any]) -> str:
        """Extract single article with rate limiting."""
        async with self.semaphore:
            await asyncio.sleep(0.5)  # Rate limiting
            return await self._extract_single(article['url'])
    
    @retry(
        stop=stop_after_attempt(2),  # Fewer retries to conserve quota
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def _extract_single(self, url: str) -> str:
        """Extract content from a single URL."""
        try:
            response = await self.client.post(
                self.base_url,
                json={
                    "api_key": self.api_key,
                    "urls": [url],
                    "include_images": False,
                    "include_links": False,
                    "include_formatting": False  # Reduce response size
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                result = data['results'][0]
                # Try both 'raw_content' and 'content' fields for compatibility
                content = result.get('raw_content', '') or result.get('content', '')
                
                # Validate content
                if len(content) < 100:
                    raise ContentExtractionError(f"Extracted content too short: {len(content)} chars for {url}")
                
                logfire.info("Successfully extracted content", extra={
                    "url": url,
                    "content_length": len(content)
                })
                
                return content
            else:
                raise ContentExtractionError(f"No content extracted for {url}")
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._request_limit = self._request_count  # Update limit
                raise ContentExtractionError("Rate limit exceeded")
            raise
    
    def _add_preview_content(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add preview content as fallback."""
        for article in articles:
            if not article.get('full_content'):
                preview = article.get('content_preview', '')
                desc = article.get('description', '')
                article['full_content'] = f"{desc}\n\n{preview}" if desc else preview
                article['extraction_success'] = False
        return articles
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()