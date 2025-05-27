import uuid
import os # Add this import

from typing import Dict, Any, Union, Optional, List
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import asyncio
import logging
import httpx
import logfire
from pydantic import HttpUrl
from functools import wraps

# Import config first to check lightweight mode
from .config import settings

# Conditional imports based on lightweight mode
if not settings.use_lightweight:
    # Set Playwright debug environment variable only for full version
    os.environ["DEBUG"] = "pw:api,pw:browser"
    from crawl4ai import AsyncWebCrawler, BrowserConfig

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Logfire with proper error handling
LOGFIRE_TOKEN = os.getenv('LOGFIRE_TOKEN')
if not LOGFIRE_TOKEN:
    logger.error("LOGFIRE_TOKEN is missing! Logs will NOT be sent to Logfire.")
else:
    try:
        logfire.configure(
            send_to_logfire="always"  # Always send logs, don't buffer
        )
        # Skip HTTPX instrumentation for now as it requires additional dependencies
        # logfire.instrument_httpx(capture_all=True)
        Agent.instrument_all()
        logger.info(f"Logfire initialized successfully [Docker] - Lightweight mode: {settings.use_lightweight}")
    except Exception as e:
        logger.exception(f"Failed to initialize logfire: {e}")

from .api_models import GenerateDigestRequest, GenerateDigestResponse, DigestStatusResponse
from .agent import rank_articles, generate_html_email
from .agent_prompts import SYSTEM_PROMPT
from .security import validate_api_key

# Conditional imports for fetcher and tools
if settings.use_lightweight:
    from .fetcher_lightweight import fetch_arxiv_cs_submissions
    from .agent_tools_lightweight import scrape_article, analyze_article
    logger.info("Using lightweight modules (no Playwright)")
else:
    from .fetcher import fetch_arxiv_cs_submissions
    from .agent_tools import scrape_article, analyze_article
    logger.info("Using full modules (with Playwright)")

# Initialize FastAPI app
app = FastAPI(
    title="Paperboy Digest Agent API",
    description="API for generating personalized `ArXiv` paper digests",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task storage
tasks: Dict[str, Dict[str, Any]] = {}

# ---- Diagnostic Router ----
diagnostics = APIRouter()

@diagnostics.get("/logfire-health", response_class=JSONResponse)
async def logfire_health_check():
    """
    Emit a test logfire event and check LOGFIRE_TOKEN surface.
    Returns success if logfire seems healthy, warns otherwise.
    """
    errors = []

    # Check token presence
    logfire_token = os.getenv('LOGFIRE_TOKEN')
    if not logfire_token:
        errors.append("LOGFIRE_TOKEN not detected in environment!")
    else:
        # Send test log using logfire directly
        logfire.info("Testing logfire log from /logfire-health endpoint.")

    # Emit a custom span for diagnostics
    with logfire.span("diagnostics.logfire_health") as span:
        span.set_attribute("env.LOGFIRE_TOKEN_length", len(logfire_token or ""))
        span.set_attribute("logfire_initiated", bool(logfire_token))

    # Check .env file configuration
    env_file = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
    env_file_exists = os.path.exists(env_file)
    if env_file_exists:
        try:
            with open(env_file, "r") as f:
                env_contents = f.read()
                if "LOGFIRE_TOKEN" not in env_contents:
                    errors.append("LOGFIRE_TOKEN not found in .env file")
        except Exception as e:
            errors.append(f"Error reading .env file: {str(e)}")
    else:
        errors.append(".env file not found")

    return {
        "logfire_token_present": bool(logfire_token),
        "token_length": len(logfire_token or "") if logfire_token else 0,
        "env_file_exists": env_file_exists,
        "errors": errors,
        "msg": "Logfire health endpoint tested; check Logfire Dashboard live view for events.",
    }

app.include_router(diagnostics, prefix="/diagnostics")

async def send_callback(task_id: str, status: str, callback_url: Optional[HttpUrl], result: Optional[str] = None):
    """Sends a POST request to the callback URL with the task status and optional result."""
    if callback_url:
        payload = {"task_id": task_id, "status": status}
        if result is not None:
            payload["result"] = result # Add result to payload if provided
        
        try:
            # Create client locally
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.post(str(callback_url), json=payload)
                response.raise_for_status() # Raise an exception for bad status codes
            logger.info(f"Callback sent successfully to {callback_url} for task {task_id}")
        except httpx.RequestError as exc:
            logger.error(f"Error sending callback to {callback_url} for task {task_id}: {exc}")
        except httpx.HTTPStatusError as exc:
             logger.error(f"Error sending callback to {callback_url} for task {task_id}: Status {exc.response.status_code} - {exc.response.text}")

def create_agent() -> Agent:
    """Create and configure a new agent instance."""
    provider = OpenAIProvider(api_key=settings.openai_api_key)
    # Initialize the model separately
    llm_model = OpenAIModel(
        settings.openai_model, # Pass model name string here
        provider=provider
        # system_prompt and max_retries don't belong here
    )
    # Pass llm_model as the FIRST positional argument
    # Pass other args like system_prompt and retries as keyword args
    return Agent(
        llm_model, # Correct: Positional argument
        system_prompt=SYSTEM_PROMPT,
        retries=settings.agent_retries # Use 'retries' not 'max_retries' for Agent
    )

async def process_with_timeout(
    task_id: str,
    user_info: Any,
    target_date: Optional[str],
    top_n: Optional[int],
    callback_url: Optional[HttpUrl] = None
) -> None:
    """Wrapper to add timeout to process_digest_request"""
    try:
        await asyncio.wait_for(
            process_digest_request(task_id, user_info, target_date, top_n, callback_url),
            timeout=settings.task_timeout if hasattr(settings, 'task_timeout') else 300  # 5 minutes default
        )
    except asyncio.TimeoutError:
        logger.error(f"Task {task_id} timed out after {settings.task_timeout if hasattr(settings, 'task_timeout') else 300} seconds")
        tasks[task_id]["status"] = "FAILED"
        tasks[task_id]["error"] = "Request timed out after 5 minutes"
        if callback_url:
            await send_callback(task_id, "FAILED", callback_url, result="Request timed out")

async def process_digest_request(
    task_id: str,
    user_info: Any,
    target_date: Optional[str],
    top_n: Optional[int],
    callback_url: Optional[HttpUrl] = None
) -> None:
    """Background task to process the digest generation request."""
    current_status = ""
    final_result: Optional[str] = None
    try:
        tasks[task_id]["status"] = "PROCESSING"
        current_status = "PROCESSING"
        # Commented out to reduce noise - only send COMPLETED callbacks
        # await send_callback(task_id, current_status, callback_url) # No result for PROCESSING
        
        agent = create_agent()
        
        # Execute the main workflow
        if settings.use_lightweight:
            # Lightweight mode - use httpx client
            async with httpx.AsyncClient(timeout=httpx.Timeout(settings.http_timeout)) as client:
                # Fetch articles
                raw_articles = await fetch_arxiv_cs_submissions(target_date, client=client)
                
                if not raw_articles:
                    logger.warning(f"No raw articles fetched for {target_date}. Ending task.")
                    tasks[task_id]["status"] = "FAILED"
                    error_message = "Failed to fetch articles or none found."
                    tasks[task_id]["error"] = error_message
                    current_status = "FAILED"
                    await send_callback(task_id, current_status, callback_url, result=error_message)
                    return

                # Rank articles
                ranked_articles = await rank_articles(user_info, raw_articles, top_n or settings.top_n_articles)
                
                if not ranked_articles:
                    logger.warning("No articles were ranked. Ending task.")
                    tasks[task_id]["status"] = "FAILED"
                    error_message = "Ranking process returned no articles."
                    tasks[task_id]["error"] = error_message
                    current_status = "FAILED"
                    await send_callback(task_id, current_status, callback_url, result=error_message)
                    return

                # Scrape articles using httpx client
                scraped_article_data: Dict[str, str] = {}
                scrape_tasks = []
                for article in ranked_articles:
                    scrape_url = str(article.abstract_url)
                    if scrape_url:
                        async def scrape_task(url_to_scrape):
                            try:
                                content = await scrape_article(client, url_to_scrape)
                                return url_to_scrape, content
                            except Exception as scrape_exc:
                                logger.error(f"Error during scrape_article call for {url_to_scrape}: {scrape_exc}", exc_info=True)
                                return url_to_scrape, f"Scraping failed: {scrape_exc}"
                        scrape_tasks.append(scrape_task(scrape_url))
                    else:
                        logger.warning(f"Skipping scrape for article '{article.title}' due to missing abstract_url.")
                
                scrape_results = await asyncio.gather(*scrape_tasks)
                for url, content_or_error in scrape_results:
                    scraped_article_data[url] = content_or_error
        else:
            # Full mode - use crawler
            playwright_launch_args = [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--disable-software-rasterizer',
                '--disable-background-networking',
                '--disable-default-apps',
                '--disable-extensions',
                '--disable-sync',
                '--disable-translate',
                '--metrics-recording-only',
                '--mute-audio',
                '--no-first-run',
                '--safebrowsing-disable-auto-update',
                '--disable-dbus',
                '--no-zygote'
            ]
            browser_config = BrowserConfig(
                extra_args=playwright_launch_args
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Fetch articles
                raw_articles = await fetch_arxiv_cs_submissions(target_date, crawler=crawler)
                
                if not raw_articles:
                    logger.warning(f"No raw articles fetched for {target_date}. Ending task.")
                    tasks[task_id]["status"] = "FAILED"
                    error_message = "Failed to fetch articles or none found."
                    tasks[task_id]["error"] = error_message
                    current_status = "FAILED"
                    await send_callback(task_id, current_status, callback_url, result=error_message)
                    return

                # Rank articles
                ranked_articles = await rank_articles(user_info, raw_articles, top_n or settings.top_n_articles)
                
                if not ranked_articles:
                    logger.warning("No articles were ranked. Ending task.")
                    tasks[task_id]["status"] = "FAILED"
                    error_message = "Ranking process returned no articles."
                    tasks[task_id]["error"] = error_message
                    current_status = "FAILED"
                    await send_callback(task_id, current_status, callback_url, result=error_message)
                    return

                # Scrape articles using crawler
                scraped_article_data: Dict[str, str] = {}
                scrape_tasks = []
                for article in ranked_articles:
                    scrape_url = str(article.abstract_url)
                    if scrape_url:
                        async def scrape_task(url_to_scrape):
                            try:
                                content = await scrape_article(crawler, url_to_scrape)
                                return url_to_scrape, content
                            except Exception as scrape_exc:
                                logger.error(f"Error during scrape_article call for {url_to_scrape}: {scrape_exc}", exc_info=True)
                                return url_to_scrape, f"Scraping failed: {scrape_exc}"
                        scrape_tasks.append(scrape_task(scrape_url))
                    else:
                        logger.warning(f"Skipping scrape for article '{article.title}' due to missing abstract_url.")
                
                scrape_results = await asyncio.gather(*scrape_tasks)
                for url, content_or_error in scrape_results:
                    scraped_article_data[url] = content_or_error
        
        # Analyze articles (common for both modes) - Process in batches to prevent overload
        analyses = []
        batch_size = 3  # Process 3 articles at a time
        
        # Prepare analysis tasks
        analysis_jobs = []
        for article in ranked_articles:
            url_key = str(article.abstract_url)
            scraped_content = scraped_article_data.get(url_key)
            
            if scraped_content and not scraped_content.startswith("Scraping failed:"):
                analysis_jobs.append((article, scraped_content))
            elif scraped_content:
                logger.warning(f"Skipping analysis for '{article.title}' due to scraping error: {scraped_content}")
            else:
                logger.warning(f"Skipping analysis for '{article.title}' because no scraped content was found (key: {url_key})")
        
        # Process in batches
        for i in range(0, len(analysis_jobs), batch_size):
            batch = analysis_jobs[i:i + batch_size]
            batch_tasks = []
            
            for article, content in batch:
                async def analysis_task(current_article, content):
                    try:
                        return await analyze_article(
                            agent,
                            content,
                            user_info,
                            current_article.model_dump()
                        )
                    except Exception as analysis_exc:
                        logger.error(f"Error during analyze_article call for {current_article.title}: {analysis_exc}", exc_info=True)
                        return None
                
                batch_tasks.append(analysis_task(article, content))
            
            # Wait for batch to complete before starting next batch
            batch_results = await asyncio.gather(*batch_tasks)
            analyses.extend([result for result in batch_results if result is not None])
            
            # Log progress
            logger.info(f"Completed analysis batch {i//batch_size + 1}/{(len(analysis_jobs) + batch_size - 1)//batch_size}")
        
        # Generate HTML
        html_content = generate_html_email(user_info, ranked_articles, analyses)
        final_result = html_content

        # Update task status
        tasks[task_id]["status"] = "COMPLETED"
        tasks[task_id]["result"] = final_result
        current_status = "COMPLETED"
        await send_callback(task_id, current_status, callback_url, result=final_result)
            
    except Exception as e:
        logger.exception(f"Unhandled error in process_digest_request for task {task_id}") # Log full traceback
        error_message = str(e)
        tasks[task_id]["status"] = "FAILED"
        tasks[task_id]["error"] = error_message
        # Send final FAILED status update if not already sent
        if current_status != "FAILED":
             await send_callback(task_id, "FAILED", callback_url, result=error_message)

@app.post("/generate-digest", response_model=GenerateDigestResponse)
async def start_digest_generation(
    request: GenerateDigestRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    api_key: str = Depends(validate_api_key)
) -> GenerateDigestResponse:
    """Start the digest generation process."""
    task_id = str(uuid.uuid4())
    callback_url_str = str(request.callback_url) if request.callback_url else None
    tasks[task_id] = {
        "status": "PENDING",
        "result": None,
        "error": None,
        "callback_url": callback_url_str
    }

    # Send initial PENDING status update (no result)
    # Commented out to reduce noise - only send COMPLETED callbacks
    # await send_callback(task_id, "PENDING", request.callback_url)

    background_tasks.add_task(
        process_with_timeout,
        task_id,
        request.user_info,
        request.target_date,
        request.top_n_articles,
        request.callback_url
    )
    
    status_url = str(http_request.url_for('get_digest_status', task_id=task_id))
    
    return GenerateDigestResponse(
        task_id=task_id,
        status="PENDING",
        message="Digest generation started.",
        status_url=status_url
    )

@app.get("/digest-status/{task_id}", response_model=DigestStatusResponse)
async def get_digest_status(
    task_id: str,
    api_key: str = Depends(validate_api_key)
) -> DigestStatusResponse:
    """Get the status of a digest generation task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return DigestStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error"),
        callback_url=task.get("callback_url")
    ) 