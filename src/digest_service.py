import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
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

class DigestService:
    """Main service for generating paper digests."""

    def __init__(self):
        self.llm_client = LLMClient()
        self.fetcher = ArxivFetcher()
        self.state_manager = TaskStateManager()
        
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

    async def generate_digest(self, task_id: str, user_info: Dict[str, Any], callback_url: str = None, target_date: str = None, top_n_articles: int = None, digest_sources: Optional[Dict[str, bool]] = None) -> None:
        """Generate a complete digest for the user."""
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
                logfire.info("Fetching papers for categories: {categories}", categories=user_info.get('categories', []))
                articles = await self._fetch_papers(user_info.get('categories', ['cs.AI', 'cs.LG']), target_date)
                logfire.info("Fetched ArXiv articles", extra={"count": len(articles)})

            # Fetch news if enabled
            news_articles = []
            if digest_sources.get("news_api", False) and self.news_fetcher and self.content_extractor and self.query_generator:
                await self.state_manager.update_task(
                    task_id,
                    DigestStatus(status=TaskStatus.PROCESSING, message="Fetching relevant news...")
                )
                news_articles = await self._fetch_news(user_info, target_date)
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

            sample_article = str(all_content[0])[:200] if all_content else 'No articles'
            logfire.info("Sample article for ranking: {sample_article}", sample_article=sample_article)
            top_n = top_n_articles if top_n_articles is not None else settings.top_n_articles
            ranked_articles = await self._rank_content(all_content, user_info, top_n)

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

            analyzed_articles = await self._analyze_papers(ranked_articles, user_info)

            digest_html = self._generate_html(analyzed_articles, user_info)

            await self._complete_task(task_id, digest_html, callback_url, analyzed_articles)

        except Exception as e:
            logfire.error("Digest generation failed: {error}", error=str(e))
            await self.state_manager.update_task(
                task_id,
                DigestStatus(status=TaskStatus.FAILED, message=str(e))
            )
            if callback_url:
                await self._send_callback(callback_url, task_id, "failed", str(e))

    async def _fetch_papers(self, categories: List[str], target_date: str = None) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv using the original catchup page method."""
        from .fetcher_lightweight import fetch_arxiv_cs_submissions
        from datetime import datetime
        
        if not target_date:
            target_date = "2025-05-01"
        
        try:
            all_articles = await fetch_arxiv_cs_submissions(target_date)
            
            if not all_articles:
                logfire.warn("No articles found for date {target_date}", target_date=target_date)
                return []
            
            for article in all_articles:
                if 'primary_subject' in article and 'subject' not in article:
                    article['subject'] = article['primary_subject']
                
                article.setdefault('subject', 'cs.AI')
                article.setdefault('authors', ['Unknown'])
            
            logfire.info("Fetched {count} articles from arXiv catchup page for {target_date}", count=len(all_articles), target_date=target_date)
            return all_articles
            
        except Exception as e:
            logfire.error("Failed to fetch papers from catchup page: {error}", error=str(e))
            return []

    async def _fetch_news(
        self,
        user_info: Dict[str, Any],
        target_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch and extract news articles with AI-powered pre-filtering."""
        if not settings.news_enabled:
            return []

        if not settings.newsapi_key or not settings.tavily_api_key:
            logfire.warn("News APIs not configured, skipping news fetch")
            return []

        try:
            # Generate queries with caching
            queries = await self.query_generator.generate_queries(user_info, target_date)
            logfire.info("Generated news queries", extra={"queries": queries, "user": user_info.get('name')})

            # Fetch raw news articles from NewsAPI (no content extraction yet)
            raw_news_articles = await self.news_fetcher.fetch_news(
                queries=queries,
                target_date=target_date,
                max_articles=settings.news_max_articles
            )

            if not raw_news_articles:
                logfire.info("No news articles found")
                return []

            logfire.info("Fetched raw news articles from NewsAPI", extra={"count": len(raw_news_articles)})

            # Step 1: Have OpenAI rank the raw news articles to get top 5 most relevant
            top_news_articles = await self._rank_news_articles(raw_news_articles, user_info, top_n=5)
            
            if not top_news_articles:
                logfire.info("No relevant news articles after AI filtering")
                return []

            logfire.info("AI filtered to top news articles", extra={"count": len(top_news_articles)})

            # Step 2: Extract content from only the top 5 news articles using Tavily
            extracted_news = await self.content_extractor.extract_articles(
                top_news_articles,
                priority_indices=None  # All are already prioritized
            )

            logfire.info("Extracted content for top news articles", extra={"count": len(extracted_news)})
            return extracted_news

        except Exception as e:
            logfire.error("Failed to fetch news", extra={"error": str(e)})
            # Return empty list to allow digest to continue with just papers
            return []

    async def _rank_news_articles(
        self,
        news_articles: List[Dict[str, Any]],
        user_info: Dict[str, Any],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Use OpenAI to rank and filter news articles before content extraction."""
        if not news_articles:
            return []

        try:
            # Use the LLM to rank news articles based on relevance
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
            # Fallback: return top articles by relevance score
            sorted_articles = sorted(
                news_articles, 
                key=lambda x: x.get('relevance_score', 0), 
                reverse=True
            )
            return sorted_articles[:top_n]

    async def _rank_content(
        self,
        content: List[Dict[str, Any]],
        user_info: Dict[str, Any],
        top_n: int
    ) -> List[RankedArticle]:
        """Rank mixed content with balanced representation before final ranking."""
        if not content:
            return []
        
        # Separate papers and news for balanced mixing
        papers = [item for item in content if item.get('type') != 'news']
        news = [item for item in content if item.get('type') == 'news']
        
        logfire.info("Content breakdown for ranking", extra={
            "papers": len(papers), 
            "news": len(news), 
            "total": len(content)
        })
        
        # Create balanced subset for final ranking
        max_articles = getattr(settings, 'ranking_input_max_articles', 20)
        
        # Ensure fair representation: if we have news, include it in the ranking subset
        if news and papers:
            # Take proportional amounts, but ensure news gets represented
            news_ratio = min(0.3, len(news) / len(content))  # Max 30% news, but at least what we have
            max_news = max(len(news), int(max_articles * news_ratio))  # Include all news articles we extracted
            max_papers = max_articles - max_news
            
            content_subset = news[:max_news] + papers[:max_papers]
            
            logfire.info("Balanced content subset for ranking", extra={
                "papers_in_subset": min(max_papers, len(papers)),
                "news_in_subset": min(max_news, len(news)),
                "total_subset": len(content_subset)
            })
        elif news:
            # Only news articles
            content_subset = news[:max_articles]
        elif papers:
            # Only papers
            content_subset = papers[:max_articles]
        else:
            content_subset = []
        
        if not content_subset:
            return []
        
        try:
            # Use the mixed content ranking method for final selection
            ranked = await self.llm_client.rank_mixed_content(
                content_subset,
                user_info,
                top_n,
                weight_recency=True  # Give slight weight to recent news
            )
            
            logfire.info("Final ranking completed", extra={
                "input_count": len(content_subset),
                "output_count": len(ranked),
                "requested_top_n": top_n
            })
            return ranked
            
        except Exception as e:
            logfire.error("Failed to rank mixed content", extra={"error": str(e)})
            # Fallback to paper-only ranking if mixed ranking fails
            if papers:
                return await self._rank_papers(papers, user_info, top_n)
            return []

    async def _rank_papers(
        self,
        articles: List[Dict[str, Any]],
        user_info: Dict[str, Any],
        top_n: int = None
    ) -> List[RankedArticle]:
        """Rank papers based on user profile."""
        from .config import settings
        max_articles = getattr(settings, 'ranking_input_max_articles', 20)
        articles_subset = articles[:max_articles]
        
        logfire.info("Ranking {subset_count} articles (limited from {total_count} total)", subset_count=len(articles_subset), total_count=len(articles))

        top_n_articles = top_n if top_n is not None else settings.top_n_articles
        
        ranked = await self.llm_client.rank_articles(
            articles_subset,
            user_info,
            top_n_articles
        )

        logfire.info("LLM returned {count} ranked articles", count=len(ranked))
        return ranked

    async def _analyze_papers(
        self,
        ranked_articles: List[RankedArticle],
        user_info: Dict[str, Any]
    ) -> List[ArticleAnalysis]:
        """Analyze top papers in parallel with rate limiting."""
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

    async def _fetch_and_analyze_article(
        self,
        article: RankedArticle,
        user_info: Dict[str, Any]
    ) -> ArticleAnalysis:
        """Fetch article content and analyze it."""
        try:
            # Handle news articles differently from papers
            if article.type == ContentType.NEWS:
                # News articles already have extracted content
                content = getattr(article, 'full_content', '') or getattr(article, 'content_preview', '')
                if not content:
                    # Fallback to description
                    content = f"Title: {article.title}\n\nDescription: {article.score_reason}"
                
                metadata = {
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
                # Research papers - fetch content from ArXiv
                content = await self.fetcher.fetch_article_content(str(article.abstract_url))

                if not content:
                    raise ValueError("No content fetched")

                metadata = {
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

            # Analyze the content using LLM
            analysis = await self.llm_client.analyze_article(content, metadata, user_info)
            
            # Preserve news-specific fields in the analysis
            if article.type == ContentType.NEWS:
                analysis.type = ContentType.NEWS
                analysis.source = getattr(article, 'source', '')
                analysis.published_at = getattr(article, 'published_at', '')
                analysis.url_to_image = getattr(article, 'url_to_image', '')
            
            return analysis

        except Exception as e:
            logfire.error("Failed to analyze {title}: {error}", title=article.title, error=str(e))
            raise

    def _generate_html(
        self,
        articles: List[ArticleAnalysis],
        user_info: Dict[str, Any]
    ) -> str:
        """Generate enhanced HTML digest with improved readability."""
        # Group by type
        papers = [a for a in articles if a.type != ContentType.NEWS]
        news = [a for a in articles if a.type == ContentType.NEWS]
        
        # Extract keywords for highlighting (future enhancement)
        # keywords = self._extract_keywords(user_info)
        
        html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Digest - {date}</title>
    <style>
        /* Base styles */
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px; 
            margin: 0 auto; 
            padding: 20px; 
            line-height: 1.7; 
            color: #2c3e50;
            background: #f8f9fa;
        }}
        
        /* Header */
        .header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 40px 30px; 
            border-radius: 16px; 
            margin-bottom: 40px; 
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .header h1 {{ 
            margin: 0 0 15px 0; 
            font-size: 2.5em; 
            font-weight: 700;
        }}
        .header-meta {{
            font-size: 1.1em;
            opacity: 0.95;
        }}
        
        /* Quick stats */
        .quick-stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .stat-item {{
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        
        /* Content sections */
        .content-section {{ 
            margin-bottom: 50px; 
        }}
        .section-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 3px solid #3498db;
        }}
        .section-header h2 {{ 
            color: #2c3e50; 
            font-size: 1.8em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-count {{
            background: #ecf0f1;
            color: #7f8c8d;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        
        /* Article cards */
        .articles-grid, .news-grid {{ 
            display: flex; 
            flex-direction: column; 
            gap: 25px; 
        }}
        .article {{ 
            background: white; 
            border: 1px solid #e1e8ed; 
            border-radius: 16px;
            padding: 30px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        .article:hover {{ 
            transform: translateY(-3px); 
            box-shadow: 0 8px 24px rgba(0,0,0,0.12); 
        }}
        
        /* Content type indicators */
        .content-type-badge {{
            position: absolute;
            top: 15px;
            right: 15px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .paper-badge {{
            background: #3498db;
            color: white;
        }}
        .news-badge {{
            background: #e74c3c;
            color: white;
        }}
        
        /* Visual distinction */
        .news-article {{ 
            border-left: 5px solid #e74c3c; 
        }}
        .paper-article {{ 
            border-left: 5px solid #3498db; 
        }}
        
        /* Article header */
        .article h3 {{ 
            margin: 0 50px 20px 0; 
            color: #2c3e50; 
            font-size: 1.4em; 
            line-height: 1.4;
            font-weight: 600;
        }}
        
        /* Enhanced metadata */
        .metadata {{ 
            color: #7f8c8d; 
            font-size: 0.9em; 
            margin-bottom: 20px; 
            display: flex; 
            flex-wrap: wrap; 
            gap: 12px;
            align-items: center;
        }}
        .metadata-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .score {{ 
            background: #27ae60; 
            color: white; 
            padding: 5px 14px; 
            border-radius: 20px;
            font-size: 0.85em; 
            font-weight: 600;
        }}
        .high-score {{ background: #27ae60; }}
        .medium-score {{ background: #f39c12; }}
        .low-score {{ background: #95a5a6; }}
        .news-score {{ 
            background: #e74c3c; 
        }}
        .source-tag {{ 
            background: #ecf0f1; 
            color: #7f8c8d; 
            padding: 3px 10px; 
            border-radius: 12px; 
            font-size: 0.8em;
        }}
        .news-source {{ 
            background: #ffeaa7; 
            color: #d63031; 
        }}
        .reading-time {{
            color: #95a5a6;
            font-size: 0.85em;
        }}
        .freshness {{
            color: #e74c3c;
            font-size: 0.85em;
            font-weight: 500;
        }}
        
        /* Content sections */
        .section {{ 
            margin: 20px 0; 
        }}
        .section h4 {{ 
            color: #34495e; 
            font-size: 1.1em; 
            margin-bottom: 10px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        /* Key findings */
        .key-findings {{ 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 12px; 
            margin: 15px 0;
            border: 1px solid #e1e8ed;
        }}
        .key-findings ul {{ 
            margin: 0; 
            padding-left: 25px; 
        }}
        .key-findings li {{ 
            margin-bottom: 10px;
            line-height: 1.6;
        }}
        
        /* News summary */
        .news-summary {{ 
            background: #fff3cd; 
            padding: 20px; 
            border-radius: 12px; 
            margin: 15px 0;
            border: 1px solid #ffeaa7;
        }}
        
        /* Expandable content */
        .expandable {{
            margin-top: 15px;
        }}
        .expand-toggle {{
            color: #3498db;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            user-select: none;
        }}
        .expand-toggle:hover {{
            text-decoration: underline;
        }}
        .expandable-content {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ecf0f1;
        }}
        
        /* Action box */
        .action-box {{ 
            background: #d5f4e6; 
            border: 2px solid #27ae60; 
            border-radius: 12px; 
            padding: 20px; 
            margin-top: 20px;
        }}
        .action-box h4 {{
            margin-top: 0;
            color: #27ae60;
        }}
        .action-box strong {{ 
            color: #27ae60; 
        }}
        
        /* Links */
        .article-links {{ 
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .article-links a {{ 
            color: #3498db; 
            text-decoration: none;
            font-size: 0.95em;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 8px 16px;
            border: 1px solid #3498db;
            border-radius: 8px;
            transition: all 0.2s ease;
        }}
        .article-links a:hover {{ 
            background: #3498db;
            color: white;
        }}
        
        /* Images */
        .news-image {{ 
            width: 100%; 
            max-width: 100%; 
            height: auto; 
            border-radius: 12px; 
            margin: 20px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        /* Keyword highlighting */
        .highlight {{
            background: #fff59d;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
        }}
        
        /* Footer */
        .footer {{ 
            text-align: center; 
            color: #95a5a6; 
            font-size: 0.9em; 
            margin-top: 60px; 
            padding: 30px 20px; 
            border-top: 2px solid #ecf0f1;
        }}
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {{
            body {{ padding: 15px; }}
            .header {{ padding: 30px 20px; }}
            .header h1 {{ font-size: 2em; }}
            .quick-stats {{ gap: 15px; }}
            .article {{ padding: 20px; }}
            .article h3 {{ font-size: 1.2em; margin-right: 40px; }}
            .metadata {{ font-size: 0.85em; }}
            .content-type-badge {{ 
                top: 10px; 
                right: 10px;
                font-size: 0.7em;
            }}
            .section-header {{ flex-direction: column; align-items: flex-start; gap: 10px; }}
            .news-image {{ margin: 15px -20px; width: calc(100% + 40px); }}
        }}
        
        /* Print styles */
        @media print {{
            body {{ background: white; }}
            .header {{ background: none; color: black; border: 2px solid black; }}
            .article {{ box-shadow: none; border: 1px solid black; page-break-inside: avoid; }}
            .expand-toggle {{ display: none; }}
            .expandable-content {{ display: block !important; }}
        }}
        
        /* Utility classes */
        .hidden {{ display: none; }}
        .mt-10 {{ margin-top: 10px; }}
        .mt-20 {{ margin-top: 20px; }}
    </style>
    <script>
        // Expandable content toggle
        function toggleExpand(id) {{
            const content = document.getElementById(id);
            const toggle = document.getElementById(id + '-toggle');
            if (content.classList.contains('hidden')) {{
                content.classList.remove('hidden');
                toggle.innerHTML = '▼ Show less';
            }} else {{
                content.classList.add('hidden');
                toggle.innerHTML = '▶ Show more';
            }}
        }}
        
        // Reading time calculation
        function calculateReadingTime(text) {{
            const wordsPerMinute = 200;
            const words = text.trim().split(/\s+/).length;
            const minutes = Math.ceil(words / wordsPerMinute);
            return minutes;
        }}
        
        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {{
            // Add reading times
            document.querySelectorAll('.article').forEach((article, index) => {{
                const text = article.innerText;
                const time = calculateReadingTime(text);
                const timeEl = article.querySelector('.reading-time');
                if (timeEl) {{
                    timeEl.innerHTML = `⏱ ${{time}} min read`;
                }}
            }});
        }});
    </script>
</head>
<body>
    <div class="header">
        <h1>📚 Your Research Digest</h1>
        <div class="header-meta">
            <p>Generated on {date} for <strong>{user_name}</strong></p>
            <p><strong>Research Focus:</strong> {interests}</p>
        </div>
        <div class="quick-stats">
            <span class="stat-item">📄 {paper_count} Papers</span>
            <span class="stat-item">📰 {news_count} News Articles</span>
            <span class="stat-item">⏱ ~{total_reading_time} min total</span>
        </div>
    </div>
""".format(
            date=datetime.now().strftime("%B %d, %Y"),
            user_name=user_info.get('name', 'Researcher'),
            interests=', '.join(user_info.get('research_interests', user_info.get('categories', ['AI Research']))),
            paper_count=len(papers),
            news_count=len(news),
            total_reading_time=(len(papers) * 5) + (len(news) * 3)  # Rough estimate
        )]

        # Add papers section
        if papers:
            html_parts.append(self._generate_papers_section(papers))

        # Add news section
        if news:
            html_parts.append(self._generate_news_section(news))

        html_parts.append("""
    <div class="footer">
        <p>🤖 Generated by Paperboy AI Research Assistant</p>
        <p>Combining the latest research papers with relevant industry news</p>
    </div>
</body>
</html>
""")

        return ''.join(html_parts)

    def _generate_papers_section(self, papers: List[ArticleAnalysis]) -> str:
        """Generate enhanced HTML section for research papers."""
        section_html = [f"""
    <div class="content-section">
        <div class="section-header">
            <h2>📚 Research Papers</h2>
            <span class="section-count">{len(papers)} articles</span>
        </div>
        <div class="articles-grid">
"""]

        for idx, paper in enumerate(papers):
            # Determine score class
            score_class = 'high-score' if paper.relevance_score >= 80 else 'medium-score' if paper.relevance_score >= 60 else 'low-score'
            
            section_html.append(f"""
            <div class="article paper-article">
                <span class="content-type-badge paper-badge">Paper</span>
                <h3>{paper.title}</h3>
                <div class="metadata">
                    <span class="score {score_class}">Score: {paper.relevance_score}/100</span>
                    <span class="source-tag">📝 {paper.subject}</span>
                    <span class="metadata-item">👥 {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}</span>
                    <span class="reading-time">⏱ 5 min read</span>
                </div>

                <div class="section">
                    <h4>🎯 Why This Matters to You</h4>
                    <p>{paper.relevance_to_user}</p>
                </div>

                <div class="key-findings">
                    <h4>🔍 Key Findings</h4>
                    <ul>
                        {''.join(f'<li>{finding}</li>' for finding in paper.key_findings[:3])}
                    </ul>
                    {f'''<div class="expandable">
                        <span class="expand-toggle" id="paper-{idx}-findings-toggle" onclick="toggleExpand('paper-{idx}-findings')">▶ Show {len(paper.key_findings) - 3} more findings</span>
                        <div id="paper-{idx}-findings" class="expandable-content hidden">
                            <ul>
                                {''.join(f'<li>{finding}</li>' for finding in paper.key_findings[3:])}
                            </ul>
                        </div>
                    </div>''' if len(paper.key_findings) > 3 else ''}
                </div>

                <div class="expandable">
                    <span class="expand-toggle" id="paper-{idx}-details-toggle" onclick="toggleExpand('paper-{idx}-details')">▶ Show technical details</span>
                    <div id="paper-{idx}-details" class="expandable-content hidden">
                        <div class="section">
                            <h4>🔧 Technical Details</h4>
                            <p>{paper.technical_details}</p>
                        </div>
                        
                        <div class="section">
                            <h4>💡 Potential Applications</h4>
                            <p>{paper.potential_applications}</p>
                        </div>
                    </div>
                </div>

                {f'''<div class="section">
                    <h4>⚠️ Critical Notes</h4>
                    <p>{paper.critical_notes}</p>
                </div>''' if paper.critical_notes else ''}

                {f'''<div class="section">
                    <h4>🔗 Next Steps</h4>
                    <p>{paper.follow_up_suggestions}</p>
                </div>''' if paper.follow_up_suggestions else ''}

                <div class="action-box">
                    <h4>📋 Recommended Action</h4>
                    <p><strong>{paper.recommended_action}</strong></p>
                </div>

                <div class="article-links">
                    <a href="{paper.abstract_url}" target="_blank">📄 View Abstract</a>
                    {f'<a href="{paper.pdf_url}" target="_blank">📁 Download PDF</a>' if paper.pdf_url else ''}
                </div>
            </div>
""")

        section_html.append("""
        </div>
    </div>
""")
        return ''.join(section_html)

    def _generate_news_section(self, news: List[ArticleAnalysis]) -> str:
        """Generate enhanced HTML section for news articles."""
        section_html = [f"""
    <div class="content-section">
        <div class="section-header">
            <h2>📰 Industry News</h2>
            <span class="section-count">{len(news)} articles</span>
        </div>
        <div class="news-grid">
"""]

        for idx, article in enumerate(news):
            # Format published date and calculate freshness
            pub_date = ""
            freshness = ""
            if hasattr(article, 'published_at') and article.published_at:
                try:
                    from datetime import datetime, timezone
                    if isinstance(article.published_at, str):
                        date_obj = datetime.fromisoformat(article.published_at.replace('Z', '+00:00'))
                        pub_date = date_obj.strftime("%B %d, %Y")
                        
                        # Calculate freshness
                        now = datetime.now(timezone.utc)
                        diff = now - date_obj
                        if diff.days == 0:
                            hours = diff.seconds // 3600
                            freshness = f"{hours} hours ago" if hours > 0 else "Just now"
                        elif diff.days == 1:
                            freshness = "Yesterday"
                        elif diff.days < 7:
                            freshness = f"{diff.days} days ago"
                except:
                    pub_date = article.published_at

            section_html.append(f"""
            <div class="article news-article">
                <span class="content-type-badge news-badge">News</span>
                <h3>{article.title}</h3>
                <div class="metadata">
                    <span class="score news-score">Score: {article.relevance_score}/100</span>
                    {f'<span class="source-tag news-source">📺 {article.source}</span>' if hasattr(article, 'source') and article.source else ''}
                    {f'<span class="metadata-item">📅 {pub_date}</span>' if pub_date else ''}
                    {f'<span class="freshness">🔥 {freshness}</span>' if freshness else ''}
                    <span class="reading-time">⏱ 3 min read</span>
                </div>

                {f'<img src="{article.url_to_image}" alt="Article image" class="news-image">' if hasattr(article, 'url_to_image') and article.url_to_image else ''}

                <div class="news-summary">
                    <h4>📰 Summary</h4>
                    <p>{article.summary}</p>
                </div>

                <div class="section">
                    <h4>🎯 Relevance to Your Work</h4>
                    <p>{article.relevance_to_user}</p>
                </div>

                <div class="section">
                    <h4>🔍 Key Points</h4>
                    <ul>
                        {''.join(f'<li>{finding}</li>' for finding in article.key_findings)}
                    </ul>
                </div>

                <div class="section">
                    <h4>💼 Importance</h4>
                    <p>{article.importance}</p>
                </div>

                {f'''<div class="section">
                    <h4>🔗 Follow-up</h4>
                    <p>{article.follow_up_suggestions}</p>
                </div>''' if article.follow_up_suggestions else ''}

                <div class="action-box">
                    <h4>📋 Recommended Action</h4>
                    <p><strong>{article.recommended_action}</strong></p>
                </div>

                <div class="article-links">
                    <a href="{article.abstract_url}" target="_blank">🔗 Read Full Article</a>
                </div>
            </div>
""")

        section_html.append("""
        </div>
    </div>
""")
        return ''.join(section_html)

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