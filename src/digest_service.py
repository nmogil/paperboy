import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logfire
import httpx

from .models import TaskStatus, DigestStatus, RankedArticle, ArticleAnalysis
from .llm_client import LLMClient
from .fetcher_lightweight import ArxivFetcher
from .state import TaskStateManager
from .config import settings

class DigestService:
    """Main service for generating paper digests."""

    def __init__(self):
        self.llm_client = LLMClient()
        self.fetcher = ArxivFetcher()
        self.state_manager = TaskStateManager()

    async def generate_digest(self, task_id: str, user_info: Dict[str, Any], callback_url: str = None, target_date: str = None, top_n_articles: int = None) -> None:
        """Generate a complete digest for the user."""
        try:
            await self.state_manager.update_task(
                task_id,
                DigestStatus(status=TaskStatus.PROCESSING, message="Fetching papers...")
            )

            logfire.info("Fetching papers for categories: {categories}", categories=user_info.get('categories', []))
            articles = await self._fetch_papers(user_info.get('categories', ['cs.AI', 'cs.LG']), target_date)

            if not articles:
                await self._complete_task(task_id, "No papers found", callback_url)
                return

            await self.state_manager.update_task(
                task_id,
                DigestStatus(status=TaskStatus.PROCESSING, message=f"Ranking {len(articles)} papers...")
            )

            sample_article = str(articles[0])[:200] if articles else 'No articles'
            logfire.info("Sample article for ranking: {sample_article}", sample_article=sample_article)
            top_n = top_n_articles if top_n_articles is not None else settings.top_n_articles
            ranked_articles = await self._rank_papers(articles, user_info, top_n)

            if not ranked_articles:
                await self._complete_task(task_id, "No relevant papers found", callback_url)
                return

            await self.state_manager.update_task(
                task_id,
                DigestStatus(
                    status=TaskStatus.PROCESSING,
                    message=f"Analyzing top {len(ranked_articles)} papers..."
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
                'pdf_url': str(article.pdf_url),
                'relevance_score': article.relevance_score,
                'score_reason': article.score_reason
            }

            return await self.llm_client.analyze_article(content, metadata, user_info)

        except Exception as e:
            logfire.error("Failed to analyze {title}: {error}", title=article.title, error=str(e))
            raise

    def _generate_html(
        self,
        articles: List[ArticleAnalysis],
        user_info: Dict[str, Any]
    ) -> str:
        """Generate HTML digest."""
        html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Research Digest - {date}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        .header {{ background: #f7f7f7; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .article {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
                   padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .article h2 {{ margin-top: 0; color: #333; }}
        .metadata {{ color: #666; font-size: 0.9em; margin-bottom: 15px; }}
        .score {{ background: #4CAF50; color: white; padding: 4px 8px; border-radius: 4px;
                 font-size: 0.8em; font-weight: bold; }}
        .section {{ margin: 15px 0; }}
        .section h3 {{ color: #555; font-size: 1.1em; margin-bottom: 8px; }}
        ul {{ padding-left: 20px; }}
        li {{ margin-bottom: 5px; }}
        .footer {{ text-align: center; color: #999; font-size: 0.9em; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📚 Your Research Digest</h1>
        <p>Generated on {date} for {user_name}</p>
        <p><strong>Research Interests:</strong> {interests}</p>
    </div>
""".format(
            date=datetime.now().strftime("%B %d, %Y"),
            user_name=user_info.get('name', 'Researcher'),
            interests=', '.join(user_info.get('research_interests', []))
        )]

        for article in articles:
            html_parts.append(f"""
    <div class="article">
        <h2>{article.title}</h2>
        <div class="metadata">
            <span class="score">Score: {article.relevance_score}/100</span>
            <span> | <a href="{article.abstract_url}">View on arXiv</a></span>
            {f'<span> | <a href="{article.pdf_url}">PDF</a></span>' if article.pdf_url else ''}
        </div>

        <div class="section">
            <h3>Why This Matters to You</h3>
            <p>{article.relevance_to_user}</p>
        </div>

        <div class="section">
            <h3>Key Findings</h3>
            <ul>
                {''.join(f'<li>{finding}</li>' for finding in article.key_findings)}
            </ul>
        </div>

        <div class="section">
            <h3>Technical Details</h3>
            <p>{article.technical_details}</p>
        </div>

        <div class="section">
            <h3>Potential Applications</h3>
            <p>{article.potential_applications}</p>
        </div>

        {f'''<div class="section">
            <h3>Critical Notes</h3>
            <p>{article.critical_notes}</p>
        </div>''' if article.critical_notes else ''}

        {f'''<div class="section">
            <h3>Next Steps</h3>
            <p>{article.follow_up_suggestions}</p>
        </div>''' if article.follow_up_suggestions else ''}

        <div class="section">
            <h3>Recommended Action</h3>
            <p><strong>{article.recommended_action}</strong></p>
        </div>
    </div>
""")

        html_parts.append("""
    <div class="footer">
        <p>Generated by Paperboy AI Research Assistant</p>
    </div>
</body>
</html>
""")

        return ''.join(html_parts)

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
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(callback_url, json=payload)
                response.raise_for_status()
            logfire.info("Callback sent successfully to {url}", url=callback_url)
        except Exception as e:
            logfire.error("Failed to send callback: {error}", error=str(e))