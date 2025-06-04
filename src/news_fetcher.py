import asyncio
import httpx
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import logfire
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time

from .config import settings
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
        
        # Time-based rate limiting: NewsAPI free tier allows ~100 requests per day
        # We'll be conservative and limit to 1 request per 2 seconds (1800 per hour max)
        self._last_request_time = 0
        self._min_request_interval = 2.0  # seconds between requests
        self._rate_limit_lock = asyncio.Lock()
        
        # Concurrent request limiter
        self._concurrent_limiter = asyncio.Semaphore(3)  # Reduced to 3 concurrent
        
        # Circuit breaker state tracking
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0
        
        # Cache to avoid duplicate requests
        self.cache = None
    
    async def _wait_for_rate_limit(self):
        """Ensure we don't exceed time-based rate limits."""
        async with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < self._min_request_interval:
                wait_time = self._min_request_interval - time_since_last
                logfire.info("Rate limiting: waiting {wait_time:.2f}s", wait_time=wait_time)
                await asyncio.sleep(wait_time)
            
            self._last_request_time = time.time()

    def _check_circuit_breaker(self):
        """Check if circuit breaker should prevent requests."""
        if not self._circuit_open:
            return True
        
        if time.time() > self._circuit_open_until:
            # Reset circuit breaker
            self._circuit_open = False
            self._consecutive_failures = 0
            logfire.info("NewsAPI circuit breaker reset")
            return True
        
        logfire.warn("NewsAPI circuit breaker is open, blocking request")
        return False

    def _handle_success(self):
        """Handle successful request - reset failure counters."""
        self._consecutive_failures = 0
        if self._circuit_open:
            self._circuit_open = False
            logfire.info("NewsAPI circuit breaker closed after successful request")

    def _handle_failure(self, is_rate_limit: bool = False):
        """Handle failed request - update circuit breaker state."""
        self._consecutive_failures += 1
        
        if is_rate_limit and self._consecutive_failures >= 3:
            # Open circuit breaker for rate limit issues
            self._circuit_open = True
            self._circuit_open_until = time.time() + 300  # 5 minutes
            logfire.warn("NewsAPI circuit breaker opened due to rate limiting for 5 minutes")
        elif self._consecutive_failures >= 5:
            # Open circuit breaker for other failures
            self._circuit_open = True
            self._circuit_open_until = time.time() + 60  # 1 minute
            logfire.warn("NewsAPI circuit breaker opened due to failures for 1 minute")

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
        # Check circuit breaker first
        if not self._check_circuit_breaker():
            logfire.warn("NewsAPI circuit breaker is open, returning empty results")
            return []
        
        # Check cache first
        cache_key = f"news:{','.join(sorted(queries))}:{target_date or 'latest'}"
        if self.cache:
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
        
        # Limit number of queries to respect rate limits
        max_queries = 3 if self._consecutive_failures > 0 else 5
        queries_to_fetch = queries_to_fetch[:max_queries]
        
        logfire.info("Fetching news", extra={
            "original_queries": len(queries),
            "optimized_queries": len(queries_to_fetch),
            "max_articles": max_articles
        })
        
        for i, query in enumerate(queries_to_fetch):
            try:
                # Rate limiting and concurrent limiting
                async with self._concurrent_limiter:
                    await self._wait_for_rate_limit()
                    
                    articles = await self._fetch_for_query(
                        query, from_date, to_date,
                        max(10, max_articles // len(queries_to_fetch))
                    )
                    
                    self._handle_success()
                    
                    # Deduplicate
                    for article in articles:
                        if article['url'] not in seen_urls:
                            seen_urls.add(article['url'])
                            all_articles.append(article)
            
            except RateLimitError:
                logfire.warn("Rate limit hit for NewsAPI", extra={"query": query})
                self._handle_failure(is_rate_limit=True)
                
                # If we hit rate limit, stop making more requests
                logfire.warn("Stopping further NewsAPI requests due to rate limit")
                break
            except Exception as e:
                logfire.error("Failed to fetch news", extra={"query": query, "error": str(e)})
                self._handle_failure(is_rate_limit=False)
                continue
        
        # If we got no successful requests, consider this a failure
        if len(all_articles) == 0:
            logfire.warn("No successful NewsAPI requests")
            return []
        
        # Sort by relevance and recency
        all_articles.sort(
            key=lambda x: (x.get('relevance_score', 0), x.get('publishedAt', '')),
            reverse=True
        )
        
        result = all_articles[:max_articles]
        
        # Cache the result only if we have a cache and got some articles
        if result and self.cache:
            # Shorter cache time if we had failures
            successful_requests = len([q for q in queries_to_fetch if q])  # Rough estimate
            cache_ttl = 1800 if successful_requests == len(queries_to_fetch) else 600
            await self.cache.set(cache_key, result, ttl=cache_ttl)
        
        logfire.info("Fetched news articles", extra={
            "count": len(result),
            "successful_requests": len(queries_to_fetch)
        })
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
                'title': article.get('title') or '',
                'description': article.get('description') or '',
                'url': article.get('url') or '',
                'published_at': article.get('publishedAt') or '',
                'source': (article.get('source') or {}).get('name') or '',
                'author': article.get('author') or '',
                'url_to_image': article.get('urlToImage') or '',
                'type': 'news',
                'subject': 'news',
                'content_preview': (article.get('content') or '')[:200],
                'relevance_score': relevance_score
            })
        
        return articles
    
    def _calculate_relevance(self, article: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for an article."""
        query_terms = set(query.lower().split())
        score = 0.0
        
        # Check title (highest weight)
        title = article.get('title') or ''
        title = title.lower() if title else ''
        title_matches = sum(1 for term in query_terms if term in title)
        score += title_matches * 0.5
        
        # Check description
        desc = article.get('description') or ''
        desc = desc.lower() if desc else ''
        desc_matches = sum(1 for term in query_terms if term in desc)
        score += desc_matches * 0.3
        
        # Normalize to 0-1
        return min(score / len(query_terms), 1.0)
    
    async def close(self):
        """Clean up client connections."""
        await self.client.aclose()