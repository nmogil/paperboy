import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import httpx
import logfire
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import settings
from .cache import SimpleCache  # New cache module
from .metrics import NewsMetrics

class NewsAPIError(Exception):
    """Custom exception for NewsAPI errors"""
    pass

class RateLimitError(NewsAPIError):
    """Rate limit exceeded"""
    pass

class NewsAPIFetcher:
    """Fetches news articles from NewsAPI with caching and rate limiting."""
    
    def __init__(self):
        if not settings.newsapi_key:
            raise ValueError("NewsAPI key not configured")
        
        self.api_key = settings.newsapi_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache = SimpleCache(ttl=settings.news_cache_ttl)
        self._rate_limiter = asyncio.Semaphore(5)  # Max 5 concurrent requests
    
    @NewsMetrics.track_api_call("NewsAPI")
    async def fetch_news(
        self,
        queries: List[str],
        target_date: Optional[str] = None,
        max_articles: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles based on queries with intelligent deduplication.
        """
        # Check cache first
        cache_key = f"news:{','.join(sorted(queries))}:{target_date or 'latest'}"
        cached = await self.cache.get(cache_key)
        if cached:
            logfire.info("News cache hit", extra={"cache_key": cache_key})
            return cached
        
        # Set date range with better logic
        from_date, to_date = self._calculate_date_range(target_date)
        
        # Fetch articles with rate limiting
        all_articles = []
        seen_urls: Set[str] = set()
        
        # Smart query batching
        queries_to_fetch = self._optimize_queries(queries)
        
        for query in queries_to_fetch:
            try:
                async with self._rate_limiter:
                    articles = await self._fetch_for_query(
                        query, from_date, to_date,
                        max(10, max_articles // len(queries_to_fetch))
                    )
                    
                    # Deduplicate
                    for article in articles:
                        if article['url'] not in seen_urls:
                            seen_urls.add(article['url'])
                            all_articles.append(article)
            
            except RateLimitError:
                logfire.warn("Rate limit hit for NewsAPI", extra={"query": query})
                await asyncio.sleep(5)  # Back off
                continue
            except Exception as e:
                logfire.error("Failed to fetch news", extra={"query": query, "error": str(e)})
        
        # Sort by relevance and recency
        all_articles.sort(
            key=lambda x: (x.get('relevance_score', 0), x.get('publishedAt', '')),
            reverse=True
        )
        
        result = all_articles[:max_articles]
        
        # Cache the result
        await self.cache.set(cache_key, result)
        
        logfire.info("Fetched news articles", extra={"count": len(result)})
        return result
    
    def _calculate_date_range(self, target_date: Optional[str]) -> tuple[str, str]:
        """Calculate date range based on target date."""
        if target_date:
            try:
                date_obj = datetime.strptime(target_date, "%Y-%m-%d")
                current_date = datetime.now()
                
                # Check if target date is in the future
                if date_obj > current_date:
                    logfire.warn("Target date is in the future, using current date instead", extra={
                        "target_date": target_date,
                        "current_date": current_date.strftime("%Y-%m-%d")
                    })
                    # Use current date instead of future date
                    from_date = current_date.strftime("%Y-%m-%d")
                    to_date = current_date.strftime("%Y-%m-%d")
                else:
                    # Use the exact target date for both from and to
                    from_date = target_date
                    to_date = target_date
                
                logfire.info("Using date range for news search", extra={
                    "original_target_date": target_date,
                    "from_date": from_date,
                    "to_date": to_date
                })
                
            except ValueError as e:
                logfire.error("Invalid date format, using yesterday", extra={
                    "target_date": target_date,
                    "error": str(e)
                })
                # Fallback to yesterday if date parsing fails
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                from_date = yesterday
                to_date = yesterday
        else:
            # Default to yesterday for latest news
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            from_date = yesterday
            to_date = yesterday
            
            logfire.info("Using yesterday for latest news search", extra={
                "from_date": from_date,
                "to_date": to_date
            })
        
        return from_date, to_date
    
    def _optimize_queries(self, queries: List[str]) -> List[str]:
        """Optimize queries to avoid redundancy."""
        # Remove duplicate terms across queries
        unique_terms = set()
        optimized = []
        
        for query in queries:
            terms = set(query.lower().split())
            if not terms.issubset(unique_terms):
                unique_terms.update(terms)
                optimized.append(query)
        
        return optimized[:10]  # Max 10 queries to avoid rate limits
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, NewsAPIError))
    )
    async def _fetch_for_query(
        self,
        query: str,
        from_date: str,
        to_date: str,
        page_size: int
    ) -> List[Dict[str, Any]]:
        """Fetch articles for a single query with retry logic."""
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': settings.news_language,
            'sortBy': settings.news_sort_by,
            'searchIn': settings.news_search_in,
            'pageSize': min(page_size, 100),
            'apiKey': self.api_key
        }
        
        # Add optional filters
        if settings.news_sources:
            params['sources'] = settings.news_sources
        if settings.news_exclude_domains:
            params['excludeDomains'] = settings.news_exclude_domains
        
        # Log the API call for debugging
        logfire.info("Making NewsAPI request", extra={
            "query": query,
            "from_date": from_date,
            "to_date": to_date,
            "page_size": page_size,
            "api_url": f"{self.base_url}?q={query}&from={from_date}&to={to_date}"
        })
        
        response = await self.client.get(self.base_url, params=params)
        
        # Handle rate limiting
        if response.status_code == 429:
            raise RateLimitError("NewsAPI rate limit exceeded")
        
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') != 'ok':
            raise NewsAPIError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
        
        logfire.info("NewsAPI response", extra={
            "query": query,
            "total_results": data.get('totalResults', 0),
            "articles_returned": len(data.get('articles', []))
        })
        
        articles = []
        for article in data.get('articles', []):
            # Calculate relevance score based on query match
            relevance_score = self._calculate_relevance(article, query)
            
            articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', ''),
                'author': article.get('author', ''),
                'url_to_image': article.get('urlToImage', ''),
                'type': 'news',
                'subject': 'news',
                'content_preview': article.get('content', '')[:200],
                'relevance_score': relevance_score
            })
        
        return articles
    
    def _calculate_relevance(self, article: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for an article."""
        query_terms = set(query.lower().split())
        score = 0.0
        
        # Check title (highest weight)
        title = (article.get('title', '') or '').lower()
        title_matches = sum(1 for term in query_terms if term in title)
        score += title_matches * 0.5
        
        # Check description
        desc = (article.get('description', '') or '').lower()
        desc_matches = sum(1 for term in query_terms if term in desc)
        score += desc_matches * 0.3
        
        # Normalize to 0-1
        return min(score / len(query_terms), 1.0)
    
    async def close(self):
        """Clean up client connections."""
        await self.client.aclose()