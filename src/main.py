from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from contextlib import asynccontextmanager
import uuid
import asyncio
import logfire
import logging
from typing import Dict, Any, Optional

from .api_models import GenerateDigestRequest, GenerateDigestResponse, DigestStatusResponse
from .digest_service import DigestService
from .state import TaskStateManager
from .models import TaskStatus, DigestStatus
from .security import validate_api_key
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Logfire
LOGFIRE_TOKEN = settings.logfire_token
if not LOGFIRE_TOKEN:
    logger.error("LOGFIRE_TOKEN is missing! Logs will NOT be sent to Logfire.")
else:
    try:
        logfire.configure(send_to_logfire="always")
        logger.info(f"Logfire initialized successfully - Lightweight mode: {settings.use_lightweight}")
    except Exception as e:
        logger.exception(f"Failed to initialize logfire: {e}")

# Simplified lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services
    app.state.state_manager = TaskStateManager()
    app.state.digest_service = DigestService()
    # Ensure digest service uses the same state manager
    app.state.digest_service.state_manager = app.state.state_manager

    # Start cleanup task
    asyncio.create_task(cleanup_old_tasks(app))

    yield

    # Cleanup
    await app.state.digest_service.fetcher.close()

async def cleanup_old_tasks(app):
    """Background task to clean up old tasks."""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        try:
            await app.state.state_manager.cleanup_old_tasks(24)
        except Exception as e:
            logfire.error(f"Cleanup task failed: {e}")

app = FastAPI(
    title="Paperboy API",
    description="AI-powered academic paper recommendation system",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/generate-digest", response_model=GenerateDigestResponse, dependencies=[Depends(validate_api_key)])
async def generate_digest(
    request: GenerateDigestRequest,
    background_tasks: BackgroundTasks
) -> GenerateDigestResponse:
    """Generate a personalized research digest."""
    task_id = str(uuid.uuid4())

    # Create initial task
    await app.state.state_manager.create_task(
        task_id,
        DigestStatus(status=TaskStatus.PENDING, message="Task created")
    )

    # Convert request to dict format expected by service
    user_info = {
        "name": request.user_info.name,
        "research_interests": [request.user_info.goals],  # Convert goals to research_interests
        "categories": getattr(request, 'categories', ['cs.AI', 'cs.LG']),
        "affiliation": getattr(request.user_info, 'title', None),  # Map title to affiliation
        "recent_focus": getattr(request.user_info, 'goals', None)
    }

    # Queue background task
    background_tasks.add_task(
        app.state.digest_service.generate_digest,
        task_id,
        user_info,
        str(request.callback_url) if request.callback_url else None,
        request.target_date,
        request.top_n_articles
    )

    return GenerateDigestResponse(
        task_id=task_id,
        status="processing",
        message="Digest generation started"
    )

@app.get("/digest-status/{task_id}", response_model=DigestStatusResponse, dependencies=[Depends(validate_api_key)])
async def get_digest_status(task_id: str) -> DigestStatusResponse:
    """Check the status of a digest generation task."""
    status = await app.state.state_manager.get_task(task_id)

    if not status:
        raise HTTPException(status_code=404, detail="Task not found")

    # Convert articles to dict for API response
    articles_dict = None
    if status.articles:
        articles_dict = [article.model_dump() for article in status.articles]

    return DigestStatusResponse(
        task_id=task_id,
        status=status.status.value,
        message=status.message,
        result=status.result,
        articles=articles_dict
    )

@app.get("/digest-status/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/health")
async def health_check_alt():
    """Alternative health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

# Additional endpoints for compatibility
@app.post("/generate_digest", response_model=GenerateDigestResponse, dependencies=[Depends(validate_api_key)])
async def generate_digest_alt(
    request: GenerateDigestRequest,
    background_tasks: BackgroundTasks
) -> GenerateDigestResponse:
    """Alternative endpoint for backward compatibility."""
    return await generate_digest(request, background_tasks)

@app.get("/digest_status/{task_id}", response_model=DigestStatusResponse, dependencies=[Depends(validate_api_key)])
async def get_digest_status_alt(task_id: str) -> DigestStatusResponse:
    """Alternative endpoint for backward compatibility."""
    return await get_digest_status(task_id)