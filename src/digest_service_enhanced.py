"""
Enhanced DigestService with circuit breakers and caching.
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict
import logfire
import httpx
import json


from .models import TaskStatus, DigestStatus, RankedArticle, ArticleAnalysis, ContentType
from .llm_client import LLMClient
from .fetcher_lightweight import ArxivFetcher
from .state import TaskStateManager
from .config import settings
from .news_fetcher import NewsAPIFetcher
from .content_extractor import TavilyExtractor
from .query_generator import QueryGenerator
from .circuit_breaker import ServiceCircuitBreakers, CircuitOpenError





class EnhancedDigestService:
    """Main service for generating paper digests with enhanced reliability."""

    def __init__(self):
        self.llm_client = LLMClient()
        self.fetcher = ArxivFetcher()
        self.state_manager = TaskStateManager()
        
        # Always initialize circuit breakers with reasonable defaults
        # This will be injected from main.py if available, otherwise use in-memory fallback
        try:
            self.circuit_breakers = ServiceCircuitBreakers()  # In-memory fallback
            logfire.info("Circuit breakers initialized with in-memory fallback")
        except Exception as e:
            logfire.error("Failed to initialize circuit breakers: {error}", error=str(e))
            self.circuit_breakers = None
        
        self.cache = None  # Will be injected from main.py
        
        # Initialize news components only if enabled
        self.news_fetcher = None
        self.content_extractor = None
        self.query_generator = None
        
        if settings.news_enabled:
            try:
                if settings.newsapi_key and settings.tavily_api_key:
                    self.news_fetcher = NewsAPIFetcher()
                    self.content_extractor = TavilyExtractor()
                    self.query_generator = QueryGenerator()
                    logfire.info("News functionality enabled")
                else:
                    logfire.warn("News enabled but API keys not configured")
            except Exception as e:
                logfire.error("Failed to initialize news components", extra={"error": str(e)})

    def _generate_tldr_summary(self, articles: List[ArticleAnalysis], user_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate TL;DR bullet points for the executive summary."""
        tldr_items = []

        try:
            # Separate papers and news for balanced representation
            papers = [a for a in articles if a.type != ContentType.NEWS]
            news = [a for a in articles if a.type == ContentType.NEWS]

            # Take top items from each category
            top_papers = sorted(papers, key=lambda x: x.relevance_score, reverse=True)[:3]
            top_news = sorted(news, key=lambda x: x.relevance_score, reverse=True)[:2]

            # Combine and sort by relevance
            top_articles = sorted(top_papers + top_news, key=lambda x: x.relevance_score, reverse=True)

            for article in top_articles:
                # Extract the most important point based on type
                if article.type == ContentType.NEWS:
                    summary = f"<strong>{article.title}:</strong> {self._simplify_content(article.summary.split('.')[0], 100)}"
                else:
                    # For papers, use the first key finding
                    if article.key_findings:
                        summary = f"<strong>{self._simplify_content(article.title, 60)}:</strong> {article.key_findings[0]}"
                    else:
                        summary = f"<strong>{self._simplify_content(article.title, 60)}:</strong> {article.summary.split('.')[0]}"

                tldr_items.append({
                    'summary': summary,
                    'relevance_score': article.relevance_score,
                    'type': article.type.value if hasattr(article.type, 'value') else str(article.type)
                })

            return tldr_items
        except Exception as e:
            logfire.error(f"Error generating TL;DR summary: {e}")
            # Return simple fallback
            return [{
                'summary': f"<strong>{articles[0].title if articles else 'No articles'}:</strong> Summary unavailable",
                'relevance_score': 0,
                'type': 'paper'
            }]

    def _calculate_total_reading_time(self, articles: List[ArticleAnalysis]) -> int:
        """Calculate total reading time based on article types."""
        try:
            total_time = 0
            for article in articles:
                if article.type == ContentType.NEWS:
                    total_time += 3  # 3 minutes for news
                else:
                    total_time += 5  # 5 minutes for papers
            return max(1, total_time)  # At least 1 minute
        except Exception as e:
            logfire.error(f"Error calculating reading time: {e}")
            return len(articles) * 4  # Default 4 minutes per article

    def _categorize_articles_by_relevance(self, articles: List[ArticleAnalysis]) -> Dict[str, List[ArticleAnalysis]]:
        """Categorize articles into relevance buckets."""
        categories = {
            'critical': [],     # 90-100 score
            'important': [],    # 70-89 score
            'interesting': [],  # 50-69 score
            'quick_scan': []    # Below 50 or remaining items
        }

        try:
            for article in articles:
                score = article.relevance_score
                if score >= 90:
                    categories['critical'].append(article)
                elif score >= 70:
                    categories['important'].append(article)
                elif score >= 50:
                    categories['interesting'].append(article)
                else:
                    categories['quick_scan'].append(article)
        except Exception as e:
            logfire.error(f"Error categorizing articles: {e}")
            # Return all articles as 'interesting' as fallback
            categories['interesting'] = articles

        return categories

    def _simplify_content(self, text: str, max_length: int = 150) -> str:
        """Simplify and truncate content for better readability."""
        if not text:
            return ""

        try:
            # Remove jargon and simplify
            replacements = {
                'This paper presents': 'Researchers found',
                'The study demonstrates': 'The study shows',
                'We propose': 'This introduces',
                'methodology': 'method',
                'utilization': 'use',
                'furthermore': 'also',
                'therefore': 'so',
                'however': 'but',
                'novel approach': 'new method',
                'state-of-the-art': 'latest',
                'empirical evidence': 'test results'
            }

            for old, new in replacements.items():
                text = text.replace(old, new)
                text = text.replace(old.lower(), new.lower())

            # Truncate to max length at sentence boundary
            if len(text) > max_length:
                # Find the last period within max_length
                truncate_point = text.rfind('.', 0, max_length)
                if truncate_point > 0:
                    text = text[:truncate_point + 1]
                else:
                    # No period found, truncate at word boundary
                    truncate_point = text.rfind(' ', 0, max_length - 3)
                    text = text[:truncate_point] + '...' if truncate_point > 0 else text[:max_length-3] + '...'

            return text.strip()
        except Exception as e:
            logfire.error(f"Error simplifying content: {e}")
            return text[:max_length] if len(text) > max_length else text

    async def generate_digest(self, task_id: str, user_info: Dict[str, Any], callback_url: str = None, target_date: str = None, top_n_articles: int = None, digest_sources: Optional[Dict[str, bool]] = None) -> None:
        """Generate a complete digest for the user with circuit breaker protection."""
        try:
            # Set default digest sources if not provided
            if digest_sources is None:
                digest_sources = {"arxiv": True, "news_api": settings.news_enabled}
            
            await self.state_manager.update_task(
                task_id,
                DigestStatus(status=TaskStatus.PROCESSING, message="Fetching content...")
            )

            # Fetch papers if enabled
            articles = []
            if digest_sources.get("arxiv", True):
                await self.state_manager.update_task(
                    task_id,
                    DigestStatus(status=TaskStatus.PROCESSING, message="Fetching papers...")
                )
                articles = await self._fetch_papers_with_breaker(user_info.get('categories', ['cs.AI', 'cs.LG']), target_date)
                logfire.info("Fetched ArXiv articles", extra={"count": len(articles)})

            # Fetch news if enabled
            news_articles = []
            if digest_sources.get("news_api", False) and self.news_fetcher and self.content_extractor and self.query_generator:
                await self.state_manager.update_task(
                    task_id,
                    DigestStatus(status=TaskStatus.PROCESSING, message="Fetching relevant news...")
                )
                news_articles = await self._fetch_news_with_breaker(user_info, target_date)
                logfire.info("Fetched news articles", extra={"count": len(news_articles)})

            # Combine articles and news
            all_content = articles + news_articles

            if not all_content:
                sources_requested = [k for k, v in digest_sources.items() if v]
                await self._complete_task(task_id, f"No content found for requested sources: {', '.join(sources_requested)}", callback_url)
                return

            await self.state_manager.update_task(
                task_id,
                DigestStatus(status=TaskStatus.PROCESSING, message=f"Ranking {len(all_content)} items...")
            )

            # Rank content with circuit breaker
            top_n = top_n_articles if top_n_articles is not None else settings.top_n_articles
            ranked_articles = await self._rank_content_with_breaker(all_content, user_info, top_n)

            if not ranked_articles:
                await self._complete_task(task_id, "No relevant content found", callback_url)
                return

            await self.state_manager.update_task(
                task_id,
                DigestStatus(
                    status=TaskStatus.PROCESSING,
                    message=f"Analyzing top {len(ranked_articles)} items..."
                )
            )

            # Analyze articles with circuit breaker
            analyzed_articles = await self._analyze_papers_with_breaker(ranked_articles, user_info)

            # Generate HTML digest
            digest_html = self._generate_html(analyzed_articles, user_info)

            await self._complete_task(task_id, digest_html, callback_url, analyzed_articles)

        except CircuitOpenError as e:
            error_msg = f"Service temporarily unavailable: {str(e)}"
            logfire.error("Circuit breaker open: {error}", error=str(e))
            await self.state_manager.update_task(
                task_id,
                DigestStatus(status=TaskStatus.FAILED, message=error_msg)
            )
            if callback_url:
                await self._send_callback(callback_url, task_id, "failed", error_msg)
        except Exception as e:
            logfire.error("Digest generation failed: {error}", error=str(e))
            await self.state_manager.update_task(
                task_id,
                DigestStatus(status=TaskStatus.FAILED, message=str(e))
            )
            if callback_url:
                await self._send_callback(callback_url, task_id, "failed", str(e))

    async def _fetch_papers_with_breaker(self, categories: List[str], target_date: str = None) -> List[Dict[str, Any]]:
        """Fetch papers with circuit breaker protection."""
        breaker = self.circuit_breakers.get('arxiv')
        
        try:
            return await breaker.call(self._fetch_papers, categories, target_date)
        except CircuitOpenError:
            logfire.warn("ArXiv circuit breaker open, returning empty list")
            return []

    async def _fetch_papers(self, categories: List[str], target_date: str = None) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv with caching."""
        # Check cache first
        cache_key = f"arxiv:{','.join(categories)}:{target_date or 'latest'}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached:
                logfire.info("ArXiv cache hit", extra={"cache_key": cache_key})
                return cached
        
        from .fetcher_lightweight import fetch_arxiv_cs_submissions
        
        if not target_date:
            target_date = "2025-05-01"
        
        try:
            all_articles = await fetch_arxiv_cs_submissions(target_date)
            
            if not all_articles:
                logfire.warn("No articles found for date {target_date}", target_date=target_date)
                return []
            
            # Normalize articles
            for article in all_articles:
                if 'primary_subject' in article and 'subject' not in article:
                    article['subject'] = article['primary_subject']
                
                article.setdefault('subject', 'cs.AI')
                article.setdefault('authors', ['Unknown'])
            
            # Cache the result
            if self.cache:
                await self.cache.set(cache_key, all_articles, use_persistent=True)
            
            logfire.info("Fetched {count} articles from arXiv", count=len(all_articles))
            return all_articles
            
        except Exception as e:
            logfire.error("Failed to fetch papers: {error}", error=str(e))
            raise

    async def _fetch_news_with_breaker(self, user_info: Dict[str, Any], target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch news with circuit breaker protection."""
        if not settings.news_enabled:
            return []
        
        try:
            # Generate queries with circuit breaker
            queries = await self._generate_queries_with_breaker(user_info, target_date)
            if not queries:
                return []
            
            # Fetch news with circuit breaker
            newsapi_breaker = self.circuit_breakers.get('newsapi')
            raw_news = await newsapi_breaker.call(
                self.news_fetcher.fetch_news,
                queries=queries,
                target_date=target_date,
                max_articles=settings.news_max_articles
            )
            
            if not raw_news:
                return []
            
            # Rank news articles
            top_news = await self._rank_news_articles(raw_news, user_info, top_n=5)
            
            if not top_news:
                return []
            
            # Extract content with circuit breaker
            tavily_breaker = self.circuit_breakers.get('tavily')
            extracted_news = await tavily_breaker.call(
                self.content_extractor.extract_articles,
                top_news,
                priority_indices=None
            )
            
            return extracted_news
            
        except CircuitOpenError as e:
            logfire.warn("News service circuit breaker open: {error}", error=str(e))
            return []
        except Exception as e:
            logfire.error("Failed to fetch news: {error}", error=str(e))
            return []

    async def _generate_queries_with_breaker(self, user_info: Dict[str, Any], target_date: Optional[str]) -> List[str]:
        """Generate queries with circuit breaker protection."""
        breaker = self.circuit_breakers.get('openai')
        
        try:
            return await breaker.call(
                self.query_generator.generate_queries,
                user_info,
                target_date
            )
        except CircuitOpenError:
            # Fallback to simple queries
            logfire.warn("Query generation circuit breaker open, using fallback queries")
            return self._get_fallback_queries(user_info)

    def _get_fallback_queries(self, user_info: Dict[str, Any]) -> List[str]:
        """Generate simple fallback queries without LLM."""
        queries = []
        
        # Add news interest if specified
        news_interest = user_info.get('news_interest')
        if news_interest:
            queries.append(news_interest)
        
        # Add goals-based queries
        goals = user_info.get('goals', '')
        if 'AI' in goals.upper():
            queries.append('artificial intelligence news')
        if 'machine learning' in goals.lower():
            queries.append('machine learning breakthrough')
        
        # Default queries
        if not queries:
            queries = ['AI technology news', 'tech industry updates']
        
        return queries[:5]

    async def _rank_content_with_breaker(
        self,
        content: List[Dict[str, Any]],
        user_info: Dict[str, Any],
        top_n: int
    ) -> List[RankedArticle]:
        """Rank content with circuit breaker protection."""
        breaker = self.circuit_breakers.get('openai')
        
        try:
            return await breaker.call(self._rank_content, content, user_info, top_n)
        except CircuitOpenError:
            logfire.error("Ranking circuit breaker open, using fallback ranking")
            # Simple fallback: return first N items
            fallback_ranked = []
            for i, item in enumerate(content[:top_n]):
                try:
                    article = RankedArticle(
                        title=item.get('title', 'Unknown'),
                        authors=item.get('authors', ['Unknown']),
                        subject=item.get('subject', 'cs.AI'),
                        score_reason="Circuit breaker active - default ranking",
                        relevance_score=100 - (i * 10),  # Decreasing scores
                        abstract_url=item.get('abstract_url', item.get('url', 'https://example.com')),
                        html_url=item.get('html_url'),
                        pdf_url=item.get('pdf_url'),
                        type=ContentType.NEWS if item.get('type') == 'news' else ContentType.PAPER
                    )
                    fallback_ranked.append(article)
                except Exception as e:
                    logfire.error("Failed to create fallback ranked article: {error}", error=str(e))
            
            return fallback_ranked

    async def _analyze_papers_with_breaker(
        self,
        ranked_articles: List[RankedArticle],
        user_info: Dict[str, Any]
    ) -> List[ArticleAnalysis]:
        """Analyze papers with circuit breaker protection."""
        breaker = self.circuit_breakers.get('openai')
        analyses = []
        
        for article in ranked_articles:
            try:
                # Fetch content first (with ArXiv breaker if applicable)
                if article.type == ContentType.NEWS:
                    content = getattr(article, 'full_content', '') or getattr(article, 'content_preview', '')
                    if not content:
                        content = f"Title: {article.title}\n\nDescription: {article.score_reason}"
                else:
                    arxiv_breaker = self.circuit_breakers.get('arxiv')
                    try:
                        content = await arxiv_breaker.call(
                            self.fetcher.fetch_article_content,
                            str(article.abstract_url)
                        )
                    except CircuitOpenError:
                        content = f"Title: {article.title}\n\nContent unavailable due to service issues."
                
                # Prepare metadata
                metadata = self._prepare_article_metadata(article)
                
                # Analyze with circuit breaker
                analysis = await breaker.call(
                    self.llm_client.analyze_article,
                    content,
                    metadata,
                    user_info
                )
                
                # Preserve type-specific fields
                if article.type == ContentType.NEWS:
                    analysis.type = ContentType.NEWS
                    analysis.source = getattr(article, 'source', '')
                    analysis.published_at = getattr(article, 'published_at', '')
                    analysis.url_to_image = getattr(article, 'url_to_image', '')
                
                analyses.append(analysis)
                
            except CircuitOpenError:
                # Create minimal analysis when circuit is open
                logfire.warn("Analysis circuit breaker open for article: {title}", title=article.title)
                fallback_analysis = self._create_fallback_analysis(article)
                analyses.append(fallback_analysis)
            except Exception as e:
                logfire.error("Failed to analyze article {title}: {error}", title=article.title, error=str(e))
        
        return analyses

    def _prepare_article_metadata(self, article: RankedArticle) -> Dict[str, Any]:
        """Prepare article metadata for analysis."""
        if article.type == ContentType.NEWS:
            return {
                'title': article.title,
                'url': str(article.abstract_url),
                'score': article.relevance_score,
                'reasoning': article.score_reason,
                'authors': article.authors,
                'subject': article.subject,
                'abstract_url': str(article.abstract_url),
                'type': 'news',
                'source': getattr(article, 'source', None),
                'published_at': getattr(article, 'published_at', None),
                'url_to_image': getattr(article, 'url_to_image', None),
                'relevance_score': article.relevance_score,
                'score_reason': article.score_reason,
                'html_url': None,
                'pdf_url': None
            }
        else:
            return {
                'title': article.title,
                'url': str(article.abstract_url),
                'score': article.relevance_score,
                'reasoning': article.score_reason,
                'authors': article.authors,
                'subject': article.subject,
                'abstract_url': str(article.abstract_url),
                'html_url': str(article.html_url) if article.html_url else None,
                'pdf_url': str(article.pdf_url) if article.pdf_url else None,
                'type': 'paper',
                'relevance_score': article.relevance_score,
                'score_reason': article.score_reason
            }

    def _create_fallback_analysis(self, article: RankedArticle) -> ArticleAnalysis:
        """Create a minimal analysis when circuit breaker is open."""
        return ArticleAnalysis(
            title=article.title,
            authors=article.authors,
            subject=article.subject,
            abstract_url=article.abstract_url,
            html_url=article.html_url,
            pdf_url=article.pdf_url,
            relevance_score=article.relevance_score,
            score_reason=article.score_reason,
            summary="Analysis unavailable due to service issues.",
            importance="Unable to determine importance at this time.",
            recommended_action="Save for later review when service is restored.",
            key_findings=["Service temporarily unavailable"],
            relevance_to_user="Relevance score: " + str(article.relevance_score),
            technical_details="Technical analysis unavailable.",
            potential_applications="Unable to analyze applications.",
            critical_notes=None,
            follow_up_suggestions=None,
            type=article.type
        )

    # Include all the original methods that don't need modification
    async def _rank_news_articles(self, news_articles: List[Dict[str, Any]], user_info: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """Use OpenAI to rank and filter news articles before content extraction."""
        # [Original implementation remains the same]
        if not news_articles:
            return []

        try:
            system_prompt = f"""You are an expert news analyst. Your task is to rank news articles based on their relevance to a user's interests and work.

Below is a list of news articles in JSON format. Select the {top_n} most relevant articles based on the user profile.

Focus on:
1. Articles directly related to the user's work, company, or industry
2. Breaking news or developments in their field of interest
3. Technology trends that align with their goals
4. Industry insights that would be valuable for their role

Your response MUST be a valid JSON array containing EXACTLY {top_n} selected articles. Each article should maintain its original structure but you can add a "relevance_reasoning" field explaining why it's relevant.

Return ONLY the JSON array, no other text."""

            user_prompt = f"""User profile:
Name: {user_info.get('name', 'User')}
Title: {user_info.get('title', '')}
Goals: {user_info.get('goals', '')}
News Interest: {user_info.get('news_interest', '')}

News Articles to rank:
{json.dumps(news_articles, indent=2)}"""

            response = await self.llm_client._call_llm(
                system_prompt,
                user_prompt,
                temperature=0.3,
                max_tokens=4000
            )

            ranked_articles = json.loads(response)
            
            if not isinstance(ranked_articles, list):
                logfire.error("LLM returned non-list for news ranking, using original articles")
                return news_articles[:top_n]
            
            logfire.info("Successfully ranked news articles with AI", extra={
                "original_count": len(news_articles),
                "filtered_count": len(ranked_articles)
            })
            
            return ranked_articles[:top_n]

        except Exception as e:
            logfire.error("Failed to rank news articles with AI", extra={"error": str(e)})
            sorted_articles = sorted(
                news_articles, 
                key=lambda x: x.get('relevance_score', 0), 
                reverse=True
            )
            return sorted_articles[:top_n]

    # Include _rank_content, _analyze_papers, _generate_html, _generate_papers_section, 
    # _generate_news_section, _complete_task, and _send_callback methods from original
    # (These remain the same as in the original digest_service.py)
    
    async def _rank_content(self, content: List[Dict[str, Any]], user_info: Dict[str, Any], top_n: int) -> List[RankedArticle]:
        """[Original implementation]"""
        if not content:
            return []
        
        papers = [item for item in content if item.get('type') != 'news']
        news = [item for item in content if item.get('type') == 'news']
        
        logfire.info("Content breakdown for ranking", extra={"papers": len(papers), "news": len(news), "total": len(content)})
        
        max_articles = getattr(settings, 'ranking_input_max_articles', 20)
        
        if news and papers:
            news_ratio = min(0.3, len(news) / len(content))
            max_news = max(len(news), int(max_articles * news_ratio))
            max_papers = max_articles - max_news
            content_subset = news[:max_news] + papers[:max_papers]
            
            logfire.info("Balanced content subset for ranking", extra={"papers_in_subset": min(max_papers, len(papers)), "news_in_subset": min(max_news, len(news)), "total_subset": len(content_subset)})
        elif news:
            content_subset = news[:max_articles]
        elif papers:
            content_subset = papers[:max_articles]
        else:
            content_subset = []
        
        if not content_subset:
            return []
        
        try:
            ranked = await self.llm_client.rank_mixed_content(content_subset, user_info, top_n, weight_recency=True)
            logfire.info("Final ranking completed", extra={"input_count": len(content_subset), "output_count": len(ranked), "requested_top_n": top_n})
            return ranked
        except Exception as e:
            logfire.error("Failed to rank mixed content", extra={"error": str(e)})
            if papers:
                return await self._rank_papers(papers, user_info, top_n)
            return []

    async def _rank_papers(self, articles: List[Dict[str, Any]], user_info: Dict[str, Any], top_n: int = None) -> List[RankedArticle]:
        """[Original implementation]"""
        from .config import settings
        max_articles = getattr(settings, 'ranking_input_max_articles', 20)
        articles_subset = articles[:max_articles]
        
        logfire.info("Ranking {subset_count} articles (limited from {total_count} total)", subset_count=len(articles_subset), total_count=len(articles))

        top_n_articles = top_n if top_n is not None else settings.top_n_articles
        
        ranked = await self.llm_client.rank_articles(articles_subset, user_info, top_n_articles)

        logfire.info("LLM returned {count} ranked articles", count=len(ranked))
        return ranked

    async def _analyze_papers(self, ranked_articles: List[RankedArticle], user_info: Dict[str, Any]) -> List[ArticleAnalysis]:
        """[Original implementation]"""
        batch_size = 3
        analyses = []

        for i in range(0, len(ranked_articles), batch_size):
            batch = ranked_articles[i:i + batch_size]
            batch_tasks = []

            for article in batch:
                task = self._fetch_and_analyze_article(article, user_info)
                batch_tasks.append(task)

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, ArticleAnalysis):
                    analyses.append(result)
                else:
                    logfire.error("Analysis failed: {error}", error=str(result))

            if i + batch_size < len(ranked_articles):
                await asyncio.sleep(1)

        logfire.info("Successfully analyzed {count} articles", count=len(analyses))
        return analyses

    async def _fetch_and_analyze_article(self, article: RankedArticle, user_info: Dict[str, Any]) -> ArticleAnalysis:
        """[Original implementation]"""
        try:
            if article.type == ContentType.NEWS:
                content = getattr(article, 'full_content', '') or getattr(article, 'content_preview', '')
                if not content:
                    content = f"Title: {article.title}\n\nDescription: {article.score_reason}"
                
                metadata = self._prepare_article_metadata(article)
            else:
                content = await self.fetcher.fetch_article_content(str(article.abstract_url))
                if not content:
                    raise ValueError("No content fetched")
                
                metadata = self._prepare_article_metadata(article)

            analysis = await self.llm_client.analyze_article(content, metadata, user_info)
            
            if article.type == ContentType.NEWS:
                analysis.type = ContentType.NEWS
                analysis.source = getattr(article, 'source', '')
                analysis.published_at = getattr(article, 'published_at', '')
                analysis.url_to_image = getattr(article, 'url_to_image', '')
            
            return analysis

        except Exception as e:
            logfire.error("Failed to analyze {title}: {error}", title=article.title, error=str(e))
            raise

    def _generate_html(self, articles: List[ArticleAnalysis], user_info: Dict[str, Any]) -> str:
        """Generate simple, newspaper-style HTML digest using template approach."""
        try:
            return self._generate_html_template(articles, user_info)
        except Exception as e:
            logfire.error(f"Error generating HTML digest: {e}")
            # Simple fallback if there are any issues
            return self._generate_fallback_html(articles, user_info, str(e))

    def _generate_html_template(self, articles: List[ArticleAnalysis], user_info: Dict[str, Any]) -> str:
        """Generate HTML using the original template-based method (fallback)."""
        try:
            # Generate TL;DR summary
            tldr_items = self._generate_tldr_summary(articles, user_info)

            # Calculate stats
            papers = [a for a in articles if a.type != ContentType.NEWS]
            news = [a for a in articles if a.type == ContentType.NEWS]
            total_reading_time = self._calculate_total_reading_time(articles)

            # Categorize articles
            categorized = self._categorize_articles_by_relevance(articles)

            # Build HTML
            html_parts = [self._generate_html_header(user_info, len(papers), len(news), total_reading_time)]

            # Add TL;DR section
            html_parts.append(self._generate_tldr_section(tldr_items, len(articles), total_reading_time))

            # Add categorized content sections
            if categorized['critical'] or categorized['important']:
                html_parts.append(self._generate_priority_section(
                    categorized['critical'] + categorized['important'],
                    "🎯 Directly Relevant to Your Work"
                ))

            if categorized['interesting']:
                html_parts.append(self._generate_priority_section(
                    categorized['interesting'],
                    "📚 Expand Your Knowledge"
                ))

            if categorized['quick_scan']:
                html_parts.append(self._generate_quick_scan_section(categorized['quick_scan']))

            html_parts.append(self._generate_html_footer(len(articles)))

            return ''.join(html_parts)
        except Exception as e:
            logfire.error(f"Error generating template HTML: {e}")
            raise

    def _generate_fallback_html(self, articles: List[ArticleAnalysis], user_info: Dict[str, Any], error_msg: str) -> str:
        """Generate a simple fallback HTML if the main generation fails."""
        current_date = datetime.now().strftime("%B %d, %Y")
        user_name = user_info.get('name', 'User')
        
        fallback_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your AI Digest - {current_date}</title>
    <style>
        body {{ font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
        .error {{ color: #e74c3c; background: #ffeaa7; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .article {{ background: #f8f9fa; padding: 20px; margin: 15px 0; border-radius: 8px; }}
        h1 {{ color: #2c3e50; }}
        h3 {{ color: #34495e; }}
        a {{ color: #3498db; }}
    </style>
</head>
<body>
    <h1>Your AI Digest - {current_date}</h1>
    <p>Hello {user_name},</p>
    <div class="error">
        <strong>Notice:</strong> There was an issue generating the full digest format. 
        Here's a simplified version of your articles.
    </div>
    <div>
        {"".join([f'<div class="article"><h3>{article.title}</h3><p>{article.summary}</p><a href="{article.abstract_url}">Read more</a></div>' for article in articles[:5]])}
    </div>
    <p><em>Technical details: {error_msg}</em></p>
</body>
</html>"""
        return fallback_html

    def _generate_html_header(self, user_info: Dict[str, Any], papers_count: int, news_count: int, total_time: int) -> str:
        """Generate simple, newspaper-style HTML header optimized for email clients."""
        try:
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            user_name = user_info.get('name', 'User')
            user_title = user_info.get('title', 'Researcher')
            user_goals = user_info.get('goals', 'AI Research')

            return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Research Digest - {current_date}</title>
    <style>
        body {{
            font-family: Georgia, Times, serif;
            line-height: 1.6;
            color: #000000;
            background-color: #ffffff;
            margin: 0;
            padding: 20px;
            max-width: 600px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 28px;
            font-weight: bold;
            margin: 0 0 5px 0;
            color: #000000;
            text-align: center;
            border-bottom: 3px solid #000000;
            padding-bottom: 10px;
        }}
        
        h2 {{
            font-size: 18px;
            font-weight: bold;
            margin: 30px 0 15px 0;
            color: #000000;
            border-bottom: 1px solid #cccccc;
            padding-bottom: 5px;
        }}
        
        h3 {{
            font-size: 16px;
            font-weight: bold;
            margin: 20px 0 10px 0;
            color: #000000;
        }}
        
        p {{
            margin: 10px 0;
            font-size: 14px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #000000;
        }}
        
        .date {{
            font-size: 12px;
            color: #666666;
            margin: 5px 0;
        }}
        
        .subtitle {{
            font-size: 14px;
            color: #666666;
            margin: 10px 0;
        }}
        
        .stats {{
            font-size: 12px;
            color: #666666;
            margin: 15px 0;
        }}
        
        .article {{
            margin: 25px 0;
            padding: 15px 0;
            border-bottom: 1px solid #eeeeee;
        }}
        
        .article-title {{
            font-size: 16px;
            font-weight: bold;
            margin: 0 0 8px 0;
            color: #000000;
        }}
        
        .article-meta {{
            font-size: 11px;
            color: #666666;
            margin: 5px 0;
        }}
        
        .summary {{
            font-size: 14px;
            margin: 10px 0;
            color: #333333;
        }}
        
        .relevance {{
            font-size: 13px;
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 3px solid #cccccc;
        }}
        
        .actions {{
            margin: 15px 0;
        }}
        
        .actions a {{
            color: #000000;
            text-decoration: underline;
            font-size: 13px;
            margin-right: 15px;
        }}
        
        .section {{
            margin: 30px 0;
        }}
        
        .quick-item {{
            margin: 8px 0;
            font-size: 13px;
            padding: 5px 0;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #cccccc;
            text-align: center;
            font-size: 12px;
            color: #666666;
        }}
        
        .score {{
            font-size: 11px;
            color: #666666;
            font-weight: normal;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📰 PAPERBOY DIGEST</h1>
        <div class="date">{current_date}</div>
        <div class="subtitle">Personalized for {user_name}, {user_title}</div>
        <div class="stats">{papers_count} Papers • {news_count} News Articles • {total_time} min read</div>
    </div>
"""
        except Exception as e:
            logfire.error(f"Error generating simple HTML header: {e}")
            return f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Paperboy Digest</title></head>
<body style="font-family: Georgia, serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<h1 style="text-align: center; border-bottom: 2px solid black;">📰 PAPERBOY DIGEST</h1>
<p style="text-align: center; color: #666;">{datetime.now().strftime('%B %d, %Y')} • {user_info.get('name', 'User')}</p>
"""

    def _generate_tldr_section(self, tldr_items: List[Dict[str, str]], total_items: int, total_time: int) -> str:
        """Generate simple executive summary section."""
        if not tldr_items:
            return ""

        try:
            bullets = []
            for item in tldr_items:
                bullets.append(f"• {item['summary']}")
            
            return f"""
    <div class="section">
        <h2>TODAY'S HIGHLIGHTS</h2>
        {"<br>".join(bullets)}
    </div>
"""
        except Exception as e:
            logfire.error(f"Error generating simple TL;DR section: {e}")
            return ""

    def _generate_priority_section(self, articles: List[ArticleAnalysis], section_title: str) -> str:
        """Generate simple article section."""
        if not articles:
            return ""

        try:
            # Clean up section title
            clean_title = section_title.replace("🎯 ", "").replace("📚 ", "").replace("🔬 ", "")
            
            articles_html = []
            for article in articles[:4]:  # Limit to 4 articles per section
                
                # Determine priority text
                if article.relevance_score >= 90:
                    priority = "HIGH PRIORITY"
                elif article.relevance_score >= 80:
                    priority = "IMPORTANT"
                else:
                    priority = "NOTEWORTHY"

                # Get article type
                article_type = "NEWS" if article.type == ContentType.NEWS else "RESEARCH"
                
                # Create summary
                if article.type == ContentType.NEWS:
                    summary = self._simplify_content(article.summary, 150)
                else:
                    summary = self._simplify_content(article.importance, 150)

                # Get relevance
                relevance = self._simplify_content(article.relevance_to_user, 200)

                # Key insight
                if article.key_findings and len(article.key_findings) > 0:
                    key_insight = self._simplify_content(article.key_findings[0], 150)
                else:
                    key_insight = self._simplify_content(article.recommended_action, 150)

                # URLs
                abstract_url = str(article.abstract_url) if article.abstract_url else "#"
                pdf_url = str(article.pdf_url) if article.pdf_url else ""

                article_html = f"""
    <div class="article">
        <div class="article-title">{article.title}</div>
        <div class="article-meta">{article_type} • {priority} • Score: {article.relevance_score}/100</div>
        
        <div class="summary">{summary}</div>
        
        <div class="relevance">
            <strong>Why this matters:</strong> {relevance}
        </div>
        
        <div class="summary">
            <strong>Key insight:</strong> {key_insight}
        </div>
        
        <div class="actions">
            <a href="{abstract_url}">Read Article</a>
            {f'<a href="{pdf_url}">Download PDF</a>' if pdf_url and article.type != ContentType.NEWS else ''}
        </div>
    </div>
"""
                articles_html.append(article_html)

            return f"""
    <div class="section">
        <h2>{clean_title.upper()}</h2>
        {''.join(articles_html)}
    </div>
"""
        except Exception as e:
            logfire.error(f"Error generating simple priority section: {e}")
            return ""

    def _generate_quick_scan_section(self, articles: List[ArticleAnalysis]) -> str:
        """Generate simple quick scan section."""
        if not articles:
            return ""

        try:
            items = []
            for article in articles[:8]:  # Show more items in quick scan
                # Truncate title
                title = article.title
                if len(title) > 80:
                    title = title[:77] + "..."
                
                # Get type and score
                article_type = "📰" if article.type == ContentType.NEWS else "📄"
                score = f"({article.relevance_score}/100)"
                
                # URL
                url = str(article.abstract_url) if article.abstract_url else "#"
                
                item_html = f"""
        <div class="quick-item">
            {article_type} <strong>{title}</strong> {score} 
            <a href="{url}">Read →</a>
        </div>
"""
                items.append(item_html)

            return f"""
    <div class="section">
        <h2>QUICK SCAN</h2>
        {''.join(items)}
    </div>
"""
        except Exception as e:
            logfire.error(f"Error generating simple quick scan section: {e}")
            return ""

    def _generate_html_footer(self, total_items: int) -> str:
        """Generate simple footer."""
        try:
            papers_processed = total_items * 4
            time_saved = max(30, total_items * 15)

            return f"""
    <div class="footer">
        <p><strong>Your Research Impact This Week</strong></p>
        <p>{papers_processed}+ papers processed • {total_items} selected for you • ~{time_saved} minutes saved</p>
        <p>Generated by <strong>Paperboy AI</strong> • Accelerating research through intelligent curation</p>
    </div>
</body>
</html>
"""
        except Exception as e:
            logfire.error(f"Error generating simple footer: {e}")
            return "</body></html>"

    def _get_user_focus_area(self) -> str:
        """Extract user's primary focus area from their info."""
        # This would need to be passed through or stored in instance
        return "AI Research"

    async def _complete_task(
        self,
        task_id: str,
        result: str,
        callback_url: str = None,
        articles: List[ArticleAnalysis] = None
    ) -> None:
        """Mark task as completed and send callback if needed."""
        await self.state_manager.update_task(
            task_id,
            DigestStatus(
                status=TaskStatus.COMPLETED,
                message="Digest generated successfully",
                result=result,
                articles=articles
            )
        )

        if callback_url:
            await self._send_callback(callback_url, task_id, "completed", result)

    async def _send_callback(self, callback_url: str, task_id: str, status: str, result: str) -> None:
        """Send callback to webhook URL."""
        payload = {
            "task_id": task_id,
            "status": status,
            "result": result if status == "completed" else None,
            "error": result if status == "failed" else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()
            logfire.info("Callback sent successfully to {url}", url=callback_url)
        except Exception as e:
            logfire.error("Failed to send callback: {error}", error=str(e))
