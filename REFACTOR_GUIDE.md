# Paperboy Refactoring Guide: Simplification & Performance

This guide provides step-by-step instructions for refactoring the Paperboy codebase to remove unnecessary complexity and improve performance. The main goal is to eliminate the Pydantic AI agent framework and replace it with a simple, direct pipeline.

## Overview of Changes

- Remove Pydantic AI agent framework
- Consolidate multiple files into a simpler structure
- Direct OpenAI API calls instead of agent abstraction
- Improved error handling and retry logic
- Performance optimizations for faster response times

## Prerequisites

Before starting:
1. Create a new branch: `git checkout -b refactor/simplify-architecture`
2. Ensure all tests pass: `pytest`
3. Back up the current working state

## Step 1: Create New Core Files

### 1.1 Create `src/llm_client.py`

This replaces the agent framework with direct OpenAI calls:

```python
import json
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import logfire

from .config import settings
from .models import RankedArticle, ArticleAnalysis

class LLMClient:
    """Simple LLM client with retry logic and structured outputs."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """Make LLM call with retry logic."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logfire.error(f"LLM call failed: {str(e)}")
            raise
    
    async def rank_articles(
        self,
        articles: List[Dict[str, Any]],
        user_info: Dict[str, Any],
        top_n: int
    ) -> List[RankedArticle]:
        """Rank articles based on user profile."""
        system_prompt = """You are an expert research paper analyst. Your task is to rank academic papers based on their relevance to a user's research interests and background.

Output a JSON array of objects with:
- title: Paper title
- score: Relevance score (0-100)
- reasoning: Brief explanation of relevance
- url: Paper URL

Only include papers with score > 40. Order by score descending."""
        
        user_prompt = f"""User Profile:
Research Interests: {', '.join(user_info.get('research_interests', []))}
Institution: {user_info.get('affiliation', 'Unknown')}
Recent Focus: {user_info.get('recent_focus', 'General research')}

Papers to rank (top {top_n} most relevant):
{json.dumps(articles, indent=2)}"""
        
        response = await self._call_llm(system_prompt, user_prompt)
        
        try:
            data = json.loads(response)
            return [RankedArticle(**item) for item in data[:top_n]]
        except (json.JSONDecodeError, ValueError) as e:
            logfire.error(f"Failed to parse ranking response: {e}")
            raise ValueError(f"Invalid LLM response format: {e}")
    
    async def analyze_article(
        self,
        article_content: str,
        article_metadata: Dict[str, Any],
        user_info: Dict[str, Any]
    ) -> ArticleAnalysis:
        """Analyze a single article."""
        system_prompt = """You are an expert at analyzing research papers for busy academics. Create a concise, insightful analysis.

Output JSON with:
- key_findings: List of 3-5 main contributions
- relevance_to_user: How this connects to the user's work
- technical_details: Important methods/results
- potential_applications: How the user might apply this
- critical_notes: Limitations or concerns
- follow_up_suggestions: Related papers or next steps"""
        
        user_prompt = f"""Analyze this paper for a researcher with interests in: {', '.join(user_info.get('research_interests', []))}

Paper: {article_metadata.get('title')}
Authors: {article_metadata.get('authors')}

Content:
{article_content[:8000]}  # Limit content length

Focus on practical insights relevant to the user's research."""
        
        response = await self._call_llm(system_prompt, user_prompt, temperature=0.3)
        
        try:
            data = json.loads(response)
            return ArticleAnalysis(**{**data, **article_metadata})
        except (json.JSONDecodeError, ValueError) as e:
            logfire.error(f"Failed to parse analysis response: {e}")
            raise ValueError(f"Invalid LLM response format: {e}")
```

### 1.2 Create `src/digest_service.py`

This is the main pipeline that orchestrates everything:

```python
import asyncio
from typing import List, Dict, Any
from datetime import datetime
import logfire

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
    
    async def generate_digest(self, task_id: str, user_info: Dict[str, Any], callback_url: str = None) -> None:
        """Generate a complete digest for the user."""
        try:
            # Update status to processing
            await self.state_manager.update_task(
                task_id, 
                DigestStatus(status=TaskStatus.PROCESSING, message="Fetching papers...")
            )
            
            # Step 1: Fetch papers
            logfire.info(f"Fetching papers for categories: {user_info.get('categories', [])}")
            articles = await self._fetch_papers(user_info.get('categories', ['cs.AI', 'cs.LG']))
            
            if not articles:
                await self._complete_task(task_id, "No papers found", callback_url)
                return
            
            # Update status
            await self.state_manager.update_task(
                task_id,
                DigestStatus(status=TaskStatus.PROCESSING, message=f"Ranking {len(articles)} papers...")
            )
            
            # Step 2: Rank papers
            ranked_articles = await self._rank_papers(articles, user_info)
            
            if not ranked_articles:
                await self._complete_task(task_id, "No relevant papers found", callback_url)
                return
            
            # Update status
            await self.state_manager.update_task(
                task_id,
                DigestStatus(
                    status=TaskStatus.PROCESSING, 
                    message=f"Analyzing top {len(ranked_articles)} papers..."
                )
            )
            
            # Step 3: Analyze top papers
            analyzed_articles = await self._analyze_papers(ranked_articles, user_info)
            
            # Step 4: Generate HTML digest
            digest_html = self._generate_html(analyzed_articles, user_info)
            
            # Complete task
            await self._complete_task(task_id, digest_html, callback_url, analyzed_articles)
            
        except Exception as e:
            logfire.error(f"Digest generation failed: {str(e)}")
            await self.state_manager.update_task(
                task_id,
                DigestStatus(status=TaskStatus.FAILED, message=str(e))
            )
            if callback_url:
                # Send failure callback
                await self._send_callback(callback_url, task_id, "failed", str(e))
    
    async def _fetch_papers(self, categories: List[str]) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv."""
        all_articles = []
        
        # Fetch in parallel for speed
        tasks = [self.fetcher.fetch_arxiv_papers(cat) for cat in categories]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            else:
                logfire.error(f"Failed to fetch papers: {result}")
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        return unique_articles
    
    async def _rank_papers(
        self, 
        articles: List[Dict[str, Any]], 
        user_info: Dict[str, Any]
    ) -> List[RankedArticle]:
        """Rank papers based on user profile."""
        # Limit to reasonable number for LLM context
        articles_subset = articles[:50]
        
        ranked = await self.llm_client.rank_articles(
            articles_subset,
            user_info,
            settings.top_n_articles
        )
        
        return ranked
    
    async def _analyze_papers(
        self,
        ranked_articles: List[RankedArticle],
        user_info: Dict[str, Any]
    ) -> List[ArticleAnalysis]:
        """Analyze top papers in parallel."""
        # Fetch content for all articles in parallel
        content_tasks = []
        for article in ranked_articles:
            content_tasks.append(self.fetcher.fetch_article_content(article.url))
        
        contents = await asyncio.gather(*content_tasks, return_exceptions=True)
        
        # Analyze articles in parallel
        analysis_tasks = []
        for article, content in zip(ranked_articles, contents):
            if isinstance(content, str) and content:
                metadata = {
                    'title': article.title,
                    'url': article.url,
                    'score': article.score,
                    'reasoning': article.reasoning
                }
                analysis_tasks.append(
                    self.llm_client.analyze_article(content, metadata, user_info)
                )
        
        analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Filter out failed analyses
        valid_analyses = [a for a in analyses if isinstance(a, ArticleAnalysis)]
        
        return valid_analyses
    
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
            <span class="score">Score: {article.score}/100</span>
            <span> | <a href="{article.url}">View on arXiv</a></span>
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
        # Implementation depends on your callback requirements
        pass
```

## Step 2: Update Existing Files

### 2.1 Simplify `src/fetcher_lightweight.py`

Remove agent-related code and optimize:

```python
# Add at the top of the file
import asyncio
from typing import List, Dict, Any

# Update the class to be more efficient
class ArxivFetcher:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def fetch_arxiv_papers(self, category: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv API."""
        # Keep existing implementation but ensure it returns simple dicts
        # Remove any agent-specific code
        
    async def fetch_article_content(self, url: str) -> str:
        """Fetch and extract article content."""
        # Keep existing implementation
        # Add better error handling and timeouts
```

### 2.2 Update `src/main.py`

Replace agent usage with the new service:

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from contextlib import asynccontextmanager
import uuid

from .api_models import DigestRequest, DigestResponse, TaskStatusResponse
from .digest_service import DigestService
from .state import TaskStateManager
from .models import TaskStatus, DigestStatus
from .security import verify_api_key

# Simplified lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services
    app.state.digest_service = DigestService()
    app.state.state_manager = TaskStateManager()
    yield
    # Cleanup
    await app.state.digest_service.fetcher.client.aclose()

app = FastAPI(
    title="Paperboy API",
    description="AI-powered academic paper recommendation system",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/generate-digest", response_model=DigestResponse, dependencies=[Depends(verify_api_key)])
async def generate_digest(
    request: DigestRequest,
    background_tasks: BackgroundTasks
) -> DigestResponse:
    """Generate a personalized research digest."""
    task_id = str(uuid.uuid4())
    
    # Create initial task
    await app.state.state_manager.create_task(
        task_id,
        DigestStatus(status=TaskStatus.PENDING, message="Task created")
    )
    
    # Queue background task
    background_tasks.add_task(
        app.state.digest_service.generate_digest,
        task_id,
        request.dict(),
        request.callback_url
    )
    
    return DigestResponse(
        task_id=task_id,
        status="processing",
        message="Digest generation started"
    )

@app.get("/digest-status/{task_id}", response_model=TaskStatusResponse, dependencies=[Depends(verify_api_key)])
async def get_digest_status(task_id: str) -> TaskStatusResponse:
    """Check the status of a digest generation task."""
    status = await app.state.state_manager.get_task(task_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatusResponse(
        task_id=task_id,
        status=status.status.value,
        message=status.message,
        result=status.result,
        articles=status.articles
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}
```

### 2.3 Update `src/models.py`

Ensure models support the simplified approach:

```python
# Add any missing fields to ArticleAnalysis
class ArticleAnalysis(BaseModel):
    title: str
    url: str
    score: int
    reasoning: str
    key_findings: List[str]
    relevance_to_user: str
    technical_details: str
    potential_applications: str
    critical_notes: Optional[str] = None
    follow_up_suggestions: Optional[str] = None
```

## Step 3: Remove Obsolete Files

Delete these files as they're no longer needed:

```bash
rm src/agent.py
rm src/agent_prompts.py
rm src/agent_tools.py
rm src/agent_tools_lightweight.py
rm src/agent_cloudrun.py
rm src/scraper_wrapper.py  # If not used elsewhere
```

## Step 4: Update Dependencies

### 4.1 Update `requirements.txt`

Remove pydantic-ai and add tenacity for retries:

```txt
# Remove:
# pydantic-ai

# Add:
tenacity>=8.2.0

# Keep existing:
fastapi>=0.104.0
openai>=1.0.0
httpx>=0.25.0
beautifulsoup4>=4.12.0
uvicorn>=0.24.0
pydantic>=2.0.0
logfire
```

### 4.2 Update Docker files

Update all Dockerfile variants to use the new requirements.

## Step 5: Update Tests

### 5.1 Create new test file `tests/test_simplified.py`

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.llm_client import LLMClient
from src.digest_service import DigestService
from src.models import RankedArticle, ArticleAnalysis

@pytest.mark.asyncio
async def test_llm_client_rank_articles():
    """Test LLM client ranking."""
    client = LLMClient()
    
    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps([
        {"title": "Test Paper", "score": 85, "reasoning": "Highly relevant", "url": "http://example.com"}
    ])
    
    with patch.object(client.client.chat.completions, 'create', return_value=mock_response):
        articles = [{"title": "Test Paper", "url": "http://example.com"}]
        user_info = {"research_interests": ["AI"]}
        
        ranked = await client.rank_articles(articles, user_info, 1)
        
        assert len(ranked) == 1
        assert ranked[0].score == 85

@pytest.mark.asyncio
async def test_digest_service_pipeline():
    """Test complete digest pipeline."""
    service = DigestService()
    
    # Mock dependencies
    with patch.object(service.fetcher, 'fetch_arxiv_papers', return_value=[
        {"title": "Paper 1", "url": "http://example.com/1"}
    ]):
        with patch.object(service.llm_client, 'rank_articles', return_value=[
            RankedArticle(title="Paper 1", score=90, reasoning="Good", url="http://example.com/1")
        ]):
            with patch.object(service.fetcher, 'fetch_article_content', return_value="Content"):
                with patch.object(service.llm_client, 'analyze_article', return_value=ArticleAnalysis(
                    title="Paper 1",
                    url="http://example.com/1",
                    score=90,
                    reasoning="Good",
                    key_findings=["Finding 1"],
                    relevance_to_user="Very relevant",
                    technical_details="Details",
                    potential_applications="Applications"
                )):
                    # Run pipeline
                    await service.generate_digest("test-id", {"categories": ["cs.AI"]})
                    
                    # Check task completed
                    status = await service.state_manager.get_task("test-id")
                    assert status.status == TaskStatus.COMPLETED
```

## Step 6: Performance Optimizations

### 6.1 Add caching to LLM client

```python
# In llm_client.py, add simple in-memory cache
from functools import lru_cache
import hashlib

class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self._cache = {}  # Simple cache
    
    def _get_cache_key(self, system: str, user: str) -> str:
        """Generate cache key from prompts."""
        content = f"{system}|{user}"
        return hashlib.md5(content.encode()).hexdigest()
```

### 6.2 Optimize arXiv fetching

- Fetch categories in parallel
- Add connection pooling
- Implement request deduplication

## Step 7: Testing & Validation

1. **Run all tests**: `pytest -v`
2. **Test locally**: 
   ```bash
   uvicorn src.main:app --reload
   curl -X POST http://localhost:8000/generate-digest \
     -H "X-API-Key: your-key" \
     -H "Content-Type: application/json" \
     -d '{"name": "Test User", "categories": ["cs.AI"]}'
   ```
3. **Check Docker build**: `docker-compose -f docker-compose.lightweight.yaml up --build`
4. **Verify memory usage is lower**
5. **Confirm response times are faster**

## Step 8: Documentation Updates

### 8.1 Update CLAUDE.md

- Remove references to agent system
- Update architecture description
- Add new file descriptions

### 8.2 Update README.md

- Simplify architecture diagram
- Update API examples
- Note performance improvements

## Final Checklist

- [ ] All agent-related code removed
- [ ] New pipeline working end-to-end
- [ ] Tests passing
- [ ] Docker builds successful
- [ ] API endpoints unchanged (backward compatible)
- [ ] Performance improved (measure response times)
- [ ] Memory usage reduced
- [ ] Error handling improved
- [ ] Documentation updated
- [ ] Code is simpler and more maintainable

## Rollback Plan

If issues arise:
1. Keep the old agent files in an `archived/` directory initially
2. Test thoroughly in staging before production
3. Monitor error rates and performance metrics
4. Have the previous Docker image tagged for quick rollback

## Expected Improvements

- **Code reduction**: ~40% fewer lines of code
- **Performance**: 20-30% faster response times
- **Memory**: Lower memory footprint
- **Debugging**: Clearer stack traces
- **Maintenance**: Easier to understand and modify