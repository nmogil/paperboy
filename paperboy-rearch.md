# Implementation Instructions for Paperboy Architecture Refactor

## Overview

We're refactoring Paperboy to split the current single `/generate-digest` endpoint into two separate endpoints:

1. **`/fetch-sources`** - Runs once per day to fetch and store all papers and news articles
2. **`/generate-digest`** - Uses pre-fetched sources to create personalized digests for each user

This will significantly reduce LLM costs by fetching content once and reusing it for all users.

## Database Changes (Already Completed)

- Added `daily_sources` table to store fetched content by date
- Added `fetch_tasks` table to track fetch operations
- Updated `digest_tasks` table with `source_date` and `digest_type` columns
- Removed deprecated `paper_recommendations` table

## Implementation Steps

### Step 1: Create the Fetch Service (`src/fetch_service.py`)

Create a new file `src/fetch_service.py` with the following structure:

```python
"""
Service to fetch and store daily sources (papers and news) in Supabase.
Runs once per day to populate the daily_sources table.
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import httpx
import logfire

from .models import TaskStatus
from .fetcher_lightweight import ArxivFetcher, fetch_arxiv_cs_submissions
from .news_fetcher import NewsAPIFetcher
from .config import settings


class FetchSourcesService:
    """Service to fetch and store daily sources."""

    def __init__(self, supabase_client):
        self.supabase_client = supabase_client
        self.arxiv_fetcher = ArxivFetcher()
        self.news_fetcher = None

        # Initialize news fetcher only if configured
        if settings.news_enabled and settings.newsapi_key:
            self.news_fetcher = NewsAPIFetcher()

    async def fetch_sources_for_date(
        self,
        task_id: str,
        source_date: str,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to fetch all sources for a given date.
        This fetches both ArXiv papers and news articles, then stores them in Supabase.
        """
        # Implementation details:
        # 1. Check if sources already exist for this date
        # 2. Fetch ArXiv papers
        # 3. Fetch news articles with generic queries
        # 4. Store both in daily_sources table
        # 5. Update task status throughout
        # 6. Send callback if URL provided
```

Key implementation points for `fetch_sources_for_date`:

- Use `fetch_arxiv_cs_submissions(source_date)` to get papers
- For news, use generic tech queries that cover common topics:
  ```python
  generic_queries = [
      "artificial intelligence breakthrough",
      "machine learning news",
      "AI technology latest",
      "tech industry updates",
      "software engineering AI",
      "data science advances",
      "robotics innovation",
      "natural language processing"
  ]
  ```
- Store normalized data structure in `daily_sources` table
- Handle both sync (no callback) and async (with callback) modes

### Step 2: Modify the Digest Service (`src/digest_service_enhanced.py`)

Update the `EnhancedDigestService.generate_digest` method:

1. **Remove all fetching logic** - No more calling ArXiv or NewsAPI directly
2. **Add source loading** - Load pre-fetched sources from `daily_sources` table
3. **Update the workflow**:

```python
async def generate_digest(self, task_id: str, user_info: Dict[str, Any],
                         callback_url: str = None, target_date: str = None,
                         top_n_articles: int = None, digest_sources: Optional[Dict[str, bool]] = None,
                         top_n_news: int = None) -> None:
    """Generate digest using pre-fetched sources from daily_sources table."""
    try:
        # Step 1: Load pre-fetched sources
        source_date = target_date or datetime.now().strftime("%Y-%m-%d")
        sources = await self._load_daily_sources(source_date)

        if not sources:
            await self._complete_task(
                task_id,
                f"No sources found for {source_date}. Please run /fetch-sources first.",
                callback_url
            )
            return

        # Step 2: Extract papers and news based on digest_sources preferences
        all_content = []
        if digest_sources.get("arxiv", True):
            all_content.extend(sources.get('arxiv_papers', []))
        if digest_sources.get("news_api", False):
            all_content.extend(sources.get('news_articles', []))

        # Step 3: Continue with existing ranking and analysis logic
        # (No changes needed for ranking, analysis, HTML generation)

        # Step 4: Update digest_tasks with source_date
        # When creating/updating task, include source_date field
```

Add the helper method:

```python
async def _load_daily_sources(self, source_date: str) -> Optional[Dict[str, Any]]:
    """Load pre-fetched sources from daily_sources table."""
    try:
        response = self.supabase_client.table('daily_sources')\
            .select("*")\
            .eq('source_date', source_date)\
            .eq('fetch_status', 'completed')\
            .execute()

        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logfire.error(f"Failed to load daily sources: {e}")
        return None
```

### Step 3: Update API Endpoints (`src/main.py`)

Add the new `/fetch-sources` endpoint:

```python
from .fetch_service import FetchSourcesService

# In lifespan function, add:
app.state.fetch_service = FetchSourcesService(app.state.supabase_client)

@app.post("/fetch-sources")
async def fetch_sources(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    api_key: str = Depends(validate_api_key)
):
    """
    Fetch and store daily sources (papers and news).
    Can run synchronously (wait for result) or asynchronously (with callback).
    """
    source_date = request.get('source_date', datetime.now().strftime("%Y-%m-%d"))
    callback_url = request.get('callback_url')

    # Validate date format
    try:
        datetime.strptime(source_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Create task
    task_id = str(uuid.uuid4())
    task_data = {
        'task_id': task_id,
        'source_date': source_date,
        'status': 'pending',
        'callback_url': callback_url,
        'created_at': datetime.now().isoformat()
    }
    app.state.supabase_client.table('fetch_tasks').insert(task_data).execute()

    if callback_url:
        # Async processing with callback
        background_tasks.add_task(
            app.state.fetch_service.fetch_sources_for_date,
            task_id,
            source_date,
            callback_url
        )

        return {
            "task_id": task_id,
            "status": "processing",
            "message": f"Fetching sources for {source_date}",
            "callback_url": callback_url
        }
    else:
        # Synchronous processing - wait for result
        try:
            result = await app.state.fetch_service.fetch_sources_for_date(
                task_id,
                source_date,
                None
            )
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/fetch-status/{task_id}")
async def get_fetch_status(
    task_id: str,
    api_key: str = Depends(validate_api_key)
):
    """Check status of fetch-sources task."""
    response = app.state.supabase_client.table('fetch_tasks')\
        .select("*")\
        .eq('task_id', task_id)\
        .execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Task not found")

    return response.data[0]
```

### Step 4: Remove Fetching Logic from Existing Services

1. **In `digest_service_enhanced.py`**:

   - Remove `_fetch_papers_with_breaker` method
   - Remove `_fetch_news_with_breaker` method
   - Remove `_generate_queries_with_breaker` method
   - Remove direct calls to `ArxivFetcher` and `NewsAPIFetcher`

2. **Keep the following methods** as they're still needed:
   - `_rank_content` and `_rank_content_with_breaker`
   - `_analyze_papers` and `_analyze_papers_with_breaker`
   - `_generate_html` and related HTML generation methods
   - All circuit breaker logic for LLM calls

### Step 5: Update Request/Response Models (`src/api_models.py`)

Add new models for the fetch endpoint:

```python
class FetchSourcesRequest(BaseModel):
    """Request model for fetching daily sources."""
    source_date: Optional[str] = Field(
        default=None,
        description="Date to fetch sources for (YYYY-MM-DD). Defaults to today."
    )
    callback_url: Optional[HttpUrl] = Field(
        default=None,
        description="Optional webhook URL for async processing"
    )

class FetchSourcesResponse(BaseModel):
    """Response model for fetch sources request."""
    task_id: str
    status: str
    message: str
    callback_url: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

class FetchStatusResponse(BaseModel):
    """Response model for checking fetch task status."""
    task_id: str
    source_date: str
    status: str
    callback_url: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
```

### Step 6: Update State Manager (`src/state_supabase.py`)

Add methods to handle fetch tasks:

```python
class TaskStateManager:
    # ... existing code ...

    async def create_fetch_task(self, task_id: str, source_date: str, callback_url: Optional[str] = None) -> None:
        """Create a new fetch task."""
        data = {
            'task_id': task_id,
            'source_date': source_date,
            'status': 'pending',
            'callback_url': callback_url,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=7)).isoformat()
        }

        self.client.table('fetch_tasks').insert(data).execute()
        logfire.info(f"Fetch task {task_id} created for {source_date}")

    async def update_fetch_task(self, task_id: str, status: str, result: Optional[Dict] = None, error: Optional[str] = None) -> None:
        """Update fetch task status."""
        data = {
            'status': status,
            'updated_at': datetime.now().isoformat()
        }

        if result:
            data['result'] = result
        if error:
            data['error'] = error

        self.client.table('fetch_tasks').update(data).eq('task_id', task_id).execute()
```

### Step 7: Testing Strategy

1. **Test `/fetch-sources` endpoint**:

   ```bash
   # Test synchronous mode
   curl -X POST http://localhost:8000/fetch-sources \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"source_date": "2025-01-06"}'

   # Test async mode with callback
   curl -X POST http://localhost:8000/fetch-sources \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{"source_date": "2025-01-06", "callback_url": "https://webhook.site/your-url"}'
   ```

2. **Test `/generate-digest` with pre-fetched sources**:
   ```bash
   # First ensure sources are fetched for the date
   # Then generate digest
   curl -X POST http://localhost:8000/generate-digest \
     -H "X-API-Key: your_api_key" \
     -H "Content-Type: application/json" \
     -d '{
       "user_info": {
         "name": "Test User",
         "title": "AI Researcher",
         "goals": "Machine learning research"
       },
       "target_date": "2025-01-06"
     }'
   ```

### Step 8: Cloud Run Deployment Considerations

1. **Set up Cloud Scheduler** to trigger `/fetch-sources` daily:

   ```yaml
   schedule: "0 6 * * *" # Run at 6 AM daily
   httpTarget:
     uri: https://your-service.run.app/fetch-sources
     httpMethod: POST
     headers:
       X-API-Key: ${API_KEY}
     body: { "source_date": null } # Will use current date
   ```

2. **Environment variables** remain the same

### Step 9: Migration Path

1. Deploy the new code with both endpoints
2. Run `/fetch-sources` to populate historical data if needed
3. Update any scheduled jobs to use the new endpoints
4. Monitor for a few days to ensure stability
5. Remove old fetching code in a future release

### Key Benefits of This Implementation

1. **Cost Savings**: Fetch once, use many times
2. **Performance**: Digest generation is much faster (no waiting for external APIs)
3. **Reliability**: If fetching fails, previous day's sources can be used
4. **Scalability**: Can handle many more users without hitting API limits
5. **Flexibility**: Can pre-fetch sources for future dates
6. **Monitoring**: Clear separation of concerns makes debugging easier
