import uuid
from typing import Dict, Any, Union, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import asyncio
import logging

from .api_models import GenerateDigestRequest, GenerateDigestResponse, DigestStatusResponse
from .agent import rank_articles, generate_html_email
from .fetcher import fetch_arxiv_cs_submissions
from .config import settings
from .agent_tools import scrape_article, analyze_article
from crawl4ai import AsyncWebCrawler
from .agent_prompts import SYSTEM_PROMPT

# Initialize FastAPI app
app = FastAPI(
    title="ArXiv Digest Agent API",
    description="API for generating personalized ArXiv paper digests",
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

logger = logging.getLogger(__name__)

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

async def process_digest_request(
    task_id: str,
    user_info: Any,
    target_date: Optional[str],
    top_n: Optional[int]
) -> None:
    """Background task to process the digest generation request."""
    try:
        tasks[task_id]["status"] = "PROCESSING"
        
        # Use async context manager for the crawler
        async with AsyncWebCrawler() as crawler:
            agent = create_agent()
            
            # Fetch articles using the managed crawler
            raw_articles = await fetch_arxiv_cs_submissions(target_date, crawler=crawler)
            
            if not raw_articles:
                logger.warning(f"No raw articles fetched for {target_date}. Ending task.")
                tasks[task_id]["status"] = "FAILED"
                tasks[task_id]["error"] = "Failed to fetch articles or none found."
                return

            # Rank articles
            ranked_articles = await rank_articles(user_info, raw_articles, top_n or settings.top_n_articles)
            
            if not ranked_articles:
                logger.warning("No articles were ranked. Ending task.")
                tasks[task_id]["status"] = "FAILED"
                tasks[task_id]["error"] = "Ranking process returned no articles."
                return

            # Scrape articles using the managed crawler
            scraped_article_data: Dict[str, str] = {} # Map abstract_url to content/error
            scrape_tasks = []
            for article in ranked_articles:
                 # Ensure we use the correct URL (HTML or Abstract) for scraping
                 # Based on agent_tools.py, scrape_article seems to handle html/abs logic
                 # abstract_url is more reliable as key
                 scrape_url = str(article.abstract_url)
                 if scrape_url:
                     async def scrape_task(url_to_scrape):
                         try:
                             content = await scrape_article(crawler, url_to_scrape)
                             return url_to_scrape, content
                         except Exception as scrape_exc:
                             logger.error(f"Error during scrape_article call for {url_to_scrape}: {scrape_exc}", exc_info=True)
                             return url_to_scrape, f"Scraping failed: {scrape_exc}" # Return error message
                     scrape_tasks.append(scrape_task(scrape_url))
                 else:
                      logger.warning(f"Skipping scrape for article '{article.title}' due to missing abstract_url.")
            
            scrape_results = await asyncio.gather(*scrape_tasks)
            for url, content_or_error in scrape_results:
                 scraped_article_data[url] = content_or_error
            
            # Analyze articles
            analyses = []
            analysis_tasks = []
            for article in ranked_articles:
                 url_key = str(article.abstract_url)
                 scraped_content = scraped_article_data.get(url_key)
                 
                 if scraped_content and not scraped_content.startswith("Scraping failed:"):
                     # Use asyncio.create_task for concurrent analysis
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
                             return None # Indicate analysis failure for this article
                     analysis_tasks.append(analysis_task(article, scraped_content))
                 elif scraped_content: # It's an error message
                     logger.warning(f"Skipping analysis for '{article.title}' due to scraping error: {scraped_content}")
                 else:
                     logger.warning(f"Skipping analysis for '{article.title}' because no scraped content was found (key: {url_key})")

            # Wait for all analysis tasks to complete
            analysis_results = await asyncio.gather(*analysis_tasks)
            analyses = [result for result in analysis_results if result is not None]
            
            # Generate HTML
            html_content = generate_html_email(user_info, ranked_articles, analyses)
            
            # Update task status
            tasks[task_id]["status"] = "COMPLETED"
            tasks[task_id]["result"] = html_content
            
    except Exception as e:
        logger.exception(f"Unhandled error in process_digest_request for task {task_id}") # Log full traceback
        tasks[task_id]["status"] = "FAILED"
        tasks[task_id]["error"] = str(e)

@app.post("/generate-digest", response_model=GenerateDigestResponse)
async def start_digest_generation(
    request: GenerateDigestRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
) -> GenerateDigestResponse:
    """Start the digest generation process."""
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "PENDING",
        "result": None,
        "error": None
    }
    
    background_tasks.add_task(
        process_digest_request,
        task_id,
        request.user_info,
        request.target_date,
        request.top_n_articles
    )
    
    status_url = str(http_request.url_for('get_digest_status', task_id=task_id))
    
    return GenerateDigestResponse(
        task_id=task_id,
        status="PENDING",
        message="Digest generation started.",
        status_url=status_url
    )

@app.get("/digest-status/{task_id}", response_model=DigestStatusResponse)
async def get_digest_status(task_id: str) -> DigestStatusResponse:
    """Get the status of a digest generation task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return DigestStatusResponse(
        task_id=task_id,
        status=task["status"],
        result=task.get("result"),
        error=task.get("error")
    ) 