"""
Enhanced main.py with Supabase state management, circuit breakers, and graceful shutdown.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
import asyncio
import logfire
import logging
import os
from typing import Dict, Any, Optional
from supabase import create_client, Client

from .api_models import GenerateDigestRequest, GenerateDigestResponse, DigestStatusResponse
from .digest_service_enhanced import EnhancedDigestService as DigestService
from .models import TaskStatus, DigestStatus
from .security import validate_api_key
from .config import settings

# Import new components
from .state_supabase import TaskStateManager
from .cache_supabase import HybridCache
from .circuit_breaker import ServiceCircuitBreakers
from .graceful_shutdown import GracefulShutdown, RequestTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Logfire
LOGFIRE_TOKEN = settings.logfire_token
if not LOGFIRE_TOKEN:
    logger.error("LOGFIRE_TOKEN is missing! Logs will NOT be sent to Logfire.")
else:
    try:
        logfire.configure(send_to_logfire="always")
        logger.info(f"Logfire initialized successfully - Lightweight mode: {settings.use_lightweight}")
    except Exception as e:
        logger.exception(f"Failed to initialize logfire: {e}")

# Global instances for connection pooling
SUPABASE_CLIENT: Optional[Client] = None
SHUTDOWN_HANDLER: Optional[GracefulShutdown] = None


def get_supabase_client() -> Client:
    """Get or create Supabase client singleton."""
    global SUPABASE_CLIENT
    if SUPABASE_CLIENT is None:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        SUPABASE_CLIENT = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized")
    
    return SUPABASE_CLIENT





@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan with graceful shutdown and connection management."""
    # Initialize shutdown handler
    global SHUTDOWN_HANDLER
    SHUTDOWN_HANDLER = GracefulShutdown(timeout=int(os.getenv('SHUTDOWN_TIMEOUT', '30')))
    SHUTDOWN_HANDLER.setup_signal_handlers()
    
    # Initialize Supabase state manager
    app.state.state_manager = TaskStateManager()
    
    # Initialize Supabase client
    app.state.supabase_client = get_supabase_client()
    
    # Initialize circuit breakers with Supabase
    app.state.circuit_breakers = ServiceCircuitBreakers(app.state.supabase_client)
    
    # Initialize hybrid cache
    app.state.cache = HybridCache(
        app.state.supabase_client,
        memory_ttl=300,  # 5 min memory cache
        persistent_ttl=int(os.getenv('CACHE_TTL', '3600'))  # 1 hour persistent
    )
    
    # Initialize digest service with enhanced components
    app.state.digest_service = DigestService()
    app.state.digest_service.state_manager = app.state.state_manager
    app.state.digest_service.circuit_breakers = app.state.circuit_breakers
    app.state.digest_service.cache = app.state.cache
    
    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_old_tasks(app))
    cache_cleanup_task = asyncio.create_task(cleanup_cache(app))
    
    # Register shutdown tasks
    async def close_connections():
        logger.info("Closing connections...")
        if hasattr(app.state.digest_service, 'fetcher'):
            await app.state.digest_service.fetcher.close()
        if hasattr(app.state.digest_service, 'news_fetcher'):
            await app.state.digest_service.news_fetcher.close()
    
    SHUTDOWN_HANDLER.register_shutdown_task(close_connections)
    
    yield
    
    # Graceful shutdown
    logger.info("Starting graceful shutdown...")
    await SHUTDOWN_HANDLER.wait_for_shutdown()
    
    # Cancel background tasks
    cleanup_task.cancel()
    cache_cleanup_task.cancel()
    
    try:
        await cleanup_task
        await cache_cleanup_task
    except asyncio.CancelledError:
        pass
    
    logger.info("Shutdown complete")


async def cleanup_old_tasks(app):
    """Background task to clean up old tasks."""
    while not SHUTDOWN_HANDLER.is_shutting_down():
        try:
            await asyncio.sleep(3600)  # Run hourly
            
            if SHUTDOWN_HANDLER.is_shutting_down():
                break
            
            count = await app.state.state_manager.cleanup_old_tasks(24)
            logfire.info(f"Cleaned up {count} old tasks")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logfire.error(f"Cleanup task failed: {e}")


async def cleanup_cache(app):
    """Background task to clean up expired cache entries."""
    while not SHUTDOWN_HANDLER.is_shutting_down():
        try:
            await asyncio.sleep(1800)  # Run every 30 minutes
            
            if SHUTDOWN_HANDLER.is_shutting_down():
                break
            
            if app.state.cache:
                await app.state.cache.clear_expired()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logfire.error(f"Cache cleanup failed: {e}")


# Create FastAPI app
app = FastAPI(
    title="Paperboy API",
    description="AI-powered academic paper recommendation system with enhanced reliability",
    version="2.1.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request tracking middleware for graceful shutdown
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for graceful shutdown."""
    if SHUTDOWN_HANDLER:
        tracker = RequestTracker(SHUTDOWN_HANDLER)
        return await tracker(request, call_next)
    return await call_next(request)


@app.post("/generate-digest", response_model=GenerateDigestResponse, dependencies=[Depends(validate_api_key)])
async def generate_digest(
    request: GenerateDigestRequest,
    background_tasks: BackgroundTasks
) -> GenerateDigestResponse:
    """Generate a personalized newsletter digest with enhanced reliability."""
    # Check if we're shutting down
    if SHUTDOWN_HANDLER and SHUTDOWN_HANDLER.is_shutting_down():
        raise HTTPException(status_code=503, detail="Service is shutting down")
    
    task_id = str(uuid.uuid4())
    
    user_info = {
        "name": request.user_info.name,
        "title": request.user_info.title,
        "goals": request.user_info.goals,
        "news_interest": getattr(request.user_info, 'news_interest', None),
        "research_interests": [request.user_info.goals],
        "categories": getattr(request, 'categories', ['cs.AI', 'cs.LG']),
        "affiliation": getattr(request.user_info, 'title', None),
        "recent_focus": getattr(request.user_info, 'goals', None)
    }
    
    # Create task with user info stored
    await app.state.state_manager.create_task(
        task_id,
        DigestStatus(status=TaskStatus.PENDING, message="Task created"),
        user_info=user_info
    )
    
    # Add task to background with request tracking
    if SHUTDOWN_HANDLER:
        async with SHUTDOWN_HANDLER.track_request(f"digest-{task_id}"):
            background_tasks.add_task(
                app.state.digest_service.generate_digest,
                task_id,
                user_info,
                str(request.callback_url) if request.callback_url else None,
                request.target_date,
                request.top_n_articles,
                request.digest_sources,
                request.top_n_news
            )
    else:
        background_tasks.add_task(
            app.state.digest_service.generate_digest,
            task_id,
            user_info,
            str(request.callback_url) if request.callback_url else None,
            request.target_date,
            request.top_n_articles,
            request.digest_sources,
            request.top_n_news
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


@app.get("/preview-new-format/{task_id}", response_class=HTMLResponse)
async def preview_new_format(task_id: str, api_key: str = Depends(validate_api_key)):
    """Preview the newsletter format for a completed task."""
    try:
        status = await app.state.state_manager.get_task(task_id)

        if not status or status.status != TaskStatus.COMPLETED:
            raise HTTPException(status_code=404, detail="Task not found or not completed")

        # Try to get user info from task or use defaults
        user_info = {
            "name": "Test User",
            "title": "Researcher",
            "goals": "AI Research"
        }

        # If we have a way to get user info from task, use it
        if hasattr(status, 'user_info') and status.user_info:
            user_info.update(status.user_info)

        # Re-generate HTML
        if status.articles:
            html = app.state.digest_service._generate_html(
                status.articles,
                user_info
            )
            return HTMLResponse(content=html)

        raise HTTPException(status_code=404, detail="No articles found in task")
    except HTTPException:
        raise
    except Exception as e:
        logfire.error(f"Error generating preview: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating preview: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.1.0"}


@app.get("/metrics")
async def get_metrics(api_key: str = Depends(validate_api_key)):
    """Get application metrics and circuit breaker status."""
    metrics = {
        "version": "2.1.0",
        "has_supabase": app.state.supabase_client is not None,
        "has_cache": app.state.cache is not None,
        "circuit_breakers": await app.state.circuit_breakers.get_all_status() if app.state.circuit_breakers else {},
    }
    
    # Add recent tasks if state manager supports it
    if hasattr(app.state.state_manager, 'get_recent_tasks'):
        try:
            metrics["recent_tasks"] = await app.state.state_manager.get_recent_tasks(5)
        except Exception as e:
            logfire.error(f"Failed to get recent tasks: {e}")
    
    return metrics


@app.get("/digest-status/health")
async def health_check_alt():
    """Alternative health check endpoint."""
    return {"status": "healthy", "version": "2.1.0"}


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