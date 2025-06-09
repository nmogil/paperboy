"""
Fetch Sources Service for Paperboy - Daily content fetching and storage.
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from supabase import Client
import logfire
import json

from .fetcher_lightweight import fetch_arxiv_cs_submissions
from .news_fetcher import NewsAPIFetcher
from .query_generator import QueryGenerator
from .circuit_breaker import ServiceCircuitBreakers, CircuitOpenError
from .config import settings
from .models import TaskStatus


class FetchSourcesService:
    """Service for fetching and storing daily sources."""
    
    def __init__(self, supabase_client: Client, circuit_breakers: ServiceCircuitBreakers):
        self.supabase_client = supabase_client
        self.circuit_breakers = circuit_breakers
        
        # Initialize news components only if enabled
        self.news_fetcher = None
        self.query_generator = None
        
        if settings.news_enabled:
            try:
                if settings.newsapi_key and settings.tavily_api_key:
                    self.news_fetcher = NewsAPIFetcher()
                    self.query_generator = QueryGenerator()
                    logfire.info("News functionality enabled in fetch service")
                else:
                    logfire.warn("News enabled but API keys not configured")
            except Exception as e:
                logfire.error("Failed to initialize news components in fetch service", extra={"error": str(e)})
    
    async def fetch_and_store_sources(self, source_date: str, task_id: str, callback_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch and store daily sources for a given date.
        
        Args:
            source_date: Date string in YYYY-MM-DD format
            task_id: Unique task identifier
            callback_url: Optional callback URL for async completion
            
        Returns:
            Dictionary with fetch results and metadata
        """
        try:
            # Validate date format
            try:
                datetime.strptime(source_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid date format: {source_date}. Expected YYYY-MM-DD")
            
            # Check if sources already exist for this date
            existing = await self._check_existing_sources(source_date)
            if existing and existing.get('fetch_status') == 'completed':
                logfire.info("Sources already exist for date", extra={"source_date": source_date})
                return {
                    "source_date": source_date,
                    "status": "completed",
                    "message": "Sources already exist for this date",
                    "arxiv_count": len(existing.get('arxiv_papers', [])),
                    "news_count": len(existing.get('news_articles', []))
                }
            
            # Update fetch task status to processing
            await self._update_fetch_task(task_id, "processing", "Fetching sources...")
            
            # Create or update daily_sources record
            await self._create_or_update_daily_sources(source_date, "processing")
            
            # Fetch ArXiv papers
            logfire.info("Fetching ArXiv papers", extra={"source_date": source_date})
            arxiv_papers = await self._fetch_arxiv_papers_with_breaker(source_date)
            
            # Fetch news articles with generic queries
            logfire.info("Fetching news articles", extra={"source_date": source_date})
            news_articles = await self._fetch_news_articles_with_breaker(source_date)
            
            # Store the fetched sources
            fetch_metadata = {
                "arxiv_count": len(arxiv_papers),
                "news_count": len(news_articles),
                "fetched_at": datetime.now().isoformat(),
                "task_id": task_id
            }
            
            await self._store_daily_sources(source_date, arxiv_papers, news_articles, fetch_metadata)
            
            # Update fetch task to completed
            result = {
                "source_date": source_date,
                "arxiv_count": len(arxiv_papers),
                "news_count": len(news_articles),
                "total_count": len(arxiv_papers) + len(news_articles)
            }
            
            await self._update_fetch_task(task_id, "completed", "Sources fetched successfully", result)
            
            logfire.info("Successfully fetched and stored sources", extra={
                "source_date": source_date,
                "arxiv_count": len(arxiv_papers),
                "news_count": len(news_articles)
            })
            
            # Send callback if provided
            if callback_url:
                await self._send_callback(callback_url, task_id, "completed", result)
            
            return {
                "source_date": source_date,
                "status": "completed",
                "message": "Sources fetched and stored successfully",
                "arxiv_count": len(arxiv_papers),
                "news_count": len(news_articles)
            }
            
        except Exception as e:
            error_msg = f"Failed to fetch sources: {str(e)}"
            logfire.error("Fetch sources failed", extra={"error": str(e), "source_date": source_date})
            
            # Update fetch task to failed
            await self._update_fetch_task(task_id, "failed", error_msg)
            
            # Update daily_sources to failed
            await self._update_daily_sources_status(source_date, "failed", error_msg)
            
            # Send callback if provided
            if callback_url:
                await self._send_callback(callback_url, task_id, "failed", error_msg)
            
            raise
    
    async def _fetch_arxiv_papers_with_breaker(self, source_date: str) -> List[Dict[str, Any]]:
        """Fetch ArXiv papers with circuit breaker protection."""
        breaker = self.circuit_breakers.get('arxiv')
        
        try:
            return await breaker.call(self._fetch_arxiv_papers, source_date)
        except CircuitOpenError:
            logfire.warn("ArXiv circuit breaker open, returning empty list")
            return []
    
    async def _fetch_arxiv_papers(self, source_date: str) -> List[Dict[str, Any]]:
        """Fetch ArXiv papers for the given date."""
        try:
            articles = await fetch_arxiv_cs_submissions(source_date)
            
            # Normalize articles - ensure consistent structure
            for article in articles:
                if 'primary_subject' in article and 'subject' not in article:
                    article['subject'] = article['primary_subject']
                
                article.setdefault('subject', 'cs.AI')
                article.setdefault('authors', ['Unknown'])
                article.setdefault('type', 'paper')
            
            logfire.info("Fetched ArXiv papers", extra={"count": len(articles), "source_date": source_date})
            return articles
            
        except Exception as e:
            logfire.error("Failed to fetch ArXiv papers", extra={"error": str(e), "source_date": source_date})
            raise
    
    async def _fetch_news_articles_with_breaker(self, source_date: str) -> List[Dict[str, Any]]:
        """Fetch news articles with circuit breaker protection."""
        if not self.news_fetcher or not settings.news_enabled:
            logfire.info("News fetching disabled or not configured")
            return []
        
        try:
            # Use QueryGenerator to get queries (will always return ["AI"])
            dummy_user_info = {"name": "fetch_service", "title": "system", "goals": ""}
            queries = await self.query_generator.generate_queries(
                dummy_user_info, 
                target_date=source_date
            )
            
            # Fetch news with circuit breaker
            newsapi_breaker = self.circuit_breakers.get('newsapi')
            news_articles = await newsapi_breaker.call(
                self.news_fetcher.fetch_news,
                queries=queries,
                target_date=source_date,
                max_articles=settings.news_max_articles
            )
            
            # Normalize news articles
            for article in news_articles:
                article.setdefault('type', 'news')
                article.setdefault('subject', 'news')
                article.setdefault('relevance_score', 0.0)
            
            logfire.info("Fetched news articles", extra={"count": len(news_articles), "source_date": source_date})
            return news_articles
            
        except CircuitOpenError:
            logfire.warn("News API circuit breaker open, returning empty list")
            return []
        except Exception as e:
            logfire.error("Failed to fetch news articles", extra={"error": str(e), "source_date": source_date})
            return []  # Don't fail the entire fetch if news fails
    
    async def _check_existing_sources(self, source_date: str) -> Optional[Dict[str, Any]]:
        """Check if sources already exist for the given date."""
        try:
            response = self.supabase_client.table('daily_sources').select("*").eq('source_date', source_date).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logfire.error("Failed to check existing sources", extra={"error": str(e), "source_date": source_date})
            return None
    
    async def _create_or_update_daily_sources(self, source_date: str, status: str) -> None:
        """Create or update daily_sources record."""
        try:
            # First try to update existing record
            existing = await self._check_existing_sources(source_date)
            
            data = {
                "fetch_status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            if existing:
                # Update existing record
                self.supabase_client.table('daily_sources').update(data).eq('source_date', source_date).execute()
            else:
                # Create new record
                data.update({
                    "source_date": source_date,
                    "arxiv_papers": [],
                    "news_articles": [],
                    "fetch_metadata": {},
                    "created_at": datetime.now().isoformat()
                })
                self.supabase_client.table('daily_sources').insert(data).execute()
                
            logfire.info("Created/updated daily_sources record", extra={"source_date": source_date, "status": status})
            
        except Exception as e:
            logfire.error("Failed to create/update daily_sources", extra={"error": str(e), "source_date": source_date})
            raise
    
    async def _store_daily_sources(
        self, 
        source_date: str, 
        arxiv_papers: List[Dict[str, Any]], 
        news_articles: List[Dict[str, Any]], 
        fetch_metadata: Dict[str, Any]
    ) -> None:
        """Store the fetched sources in daily_sources table."""
        try:
            data = {
                "arxiv_papers": arxiv_papers,
                "news_articles": news_articles,
                "fetch_status": "completed",
                "fetch_metadata": fetch_metadata,
                "updated_at": datetime.now().isoformat()
            }
            
            # Retry logic for SSL errors with exponential backoff
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    # Run the synchronous Supabase operation in a thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self.supabase_client.table('daily_sources').update(data).eq('source_date', source_date).execute()
                    )
                    break
                except Exception as e:
                    error_str = str(e)
                    if ("SSLV3_ALERT_BAD_RECORD_MAC" in error_str or 
                        "SSL" in error_str or 
                        "ReadError" in error_str) and attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
                        logfire.info(f"SSL/Network error on attempt {attempt + 1}, retrying in {wait_time}s...", 
                                   extra={"error": error_str, "attempt": attempt + 1})
                        await asyncio.sleep(wait_time)
                        continue
                    raise
            
            logfire.info("Stored daily sources", extra={
                "source_date": source_date,
                "arxiv_count": len(arxiv_papers),
                "news_count": len(news_articles)
            })
            
        except Exception as e:
            logfire.error("Failed to store daily sources", extra={"error": str(e), "source_date": source_date})
            raise
    
    async def _update_daily_sources_status(self, source_date: str, status: str, error: Optional[str] = None) -> None:
        """Update the status of a daily_sources record."""
        try:
            data = {
                "fetch_status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            if error:
                data["error"] = error
            
            # Run in executor to handle potential SSL issues
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.supabase_client.table('daily_sources').update(data).eq('source_date', source_date).execute()
            )
            
        except Exception as e:
            logfire.error("Failed to update daily_sources status", extra={"error": str(e), "source_date": source_date})
    
    async def _update_fetch_task(self, task_id: str, status: str, message: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Update fetch task status."""
        try:
            data = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            if result:
                data["result"] = result
            else:
                data["error"] = message if status == "failed" else None
            
            self.supabase_client.table('fetch_tasks').update(data).eq('task_id', task_id).execute()
            
            logfire.info("Updated fetch task", extra={"task_id": task_id, "status": status})
            
        except Exception as e:
            logfire.error("Failed to update fetch task", extra={"error": str(e), "task_id": task_id})
    
    async def _send_callback(self, callback_url: str, task_id: str, status: str, result: Any) -> None:
        """Send callback notification."""
        import httpx
        
        payload = {
            "task_id": task_id,
            "status": status,
            "result": result if status == "completed" else None,
            "error": result if status == "failed" else None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()
            logfire.info("Fetch callback sent successfully", extra={"callback_url": callback_url, "task_id": task_id})
        except Exception as e:
            logfire.error("Failed to send fetch callback", extra={"error": str(e), "task_id": task_id})


class DailySourcesManager:
    """Manager for retrieving and working with daily sources."""
    
    def __init__(self, supabase_client: Client):
        self.supabase_client = supabase_client
    
    async def get_sources_for_date(self, source_date: str) -> Optional[Dict[str, Any]]:
        """Get sources for a specific date."""
        try:
            response = self.supabase_client.table('daily_sources').select("*").eq('source_date', source_date).execute()
            
            if response.data and len(response.data) > 0:
                sources = response.data[0]
                if sources.get('fetch_status') == 'completed':
                    return sources
                else:
                    logfire.warn("Sources exist but not completed", extra={"source_date": source_date, "status": sources.get('fetch_status')})
                    return None
            
            logfire.warn("No sources found for date", extra={"source_date": source_date})
            return None
            
        except Exception as e:
            logfire.error("Failed to get sources for date", extra={"error": str(e), "source_date": source_date})
            return None
    
    async def get_latest_sources(self) -> Optional[Dict[str, Any]]:
        """Get the latest available sources."""
        try:
            response = self.supabase_client.table('daily_sources').select("*").eq('fetch_status', 'completed').order('source_date', desc=True).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            logfire.warn("No completed sources found")
            return None
            
        except Exception as e:
            logfire.error("Failed to get latest sources", extra={"error": str(e)})
            return None
    
    async def list_available_dates(self, limit: int = 30) -> List[str]:
        """List available source dates."""
        try:
            response = self.supabase_client.table('daily_sources').select("source_date").eq('fetch_status', 'completed').order('source_date', desc=True).limit(limit).execute()
            
            if response.data:
                return [item['source_date'] for item in response.data]
            
            return []
            
        except Exception as e:
            logfire.error("Failed to list available dates", extra={"error": str(e)})
            return []