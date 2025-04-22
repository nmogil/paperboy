# settings.py

import os
import logging
import logfire
from pydantic_ai import Agent

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
            project_name="paperboy-api",
            send_to_logfire="always",  # Always send logs, don't buffer
            debug=True  # Enable debug mode for more info
        )
        logfire.instrument_httpx(capture_all=True)
        Agent.instrument_all()
        logfire.get_logger().info("Logfire initialized successfully [Docker].")
    except Exception as e:
        logger.exception(f"Failed to initialize logfire: {e}")

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class AgentSettings(BaseSettings):
    """Settings for the article ranking agent"""
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini-2025-04-14", env="OPENAI_MODEL")
    max_articles: int = Field(default=50, env="MAX_ARTICLES")
    max_concurrent_scrapes: int = Field(default=5, env="MAX_CONCURRENT_SCRAPES")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    arxiv_file: str = Field(default="arxiv_cs_submissions_2025-04-01.json")
    top_n_articles: int = Field(default=5)
    logfire_token: Optional[str] = Field(None, env="LOGFIRE_TOKEN")  # Add explicit logfire token field
    
    class Config:
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        env_file_encoding = "utf-8" 