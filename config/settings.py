# settings.py

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class AgentSettings(BaseSettings):
    """Settings for the article ranking agent"""
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini-2025-04-14", env="OPENAI_MODEL")
    max_articles: int = Field(default=50, env="MAX_ARTICLES")
    max_concurrent_scrapes: int = Field(default=5, env="MAX_CONCURRENT_SCRAPES")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    arxiv_file: str = Field(default="arxiv_cs_submissions_2025-04-01.json")
    top_n_articles: int = Field(default=5)
    
    class Config:
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        env_file_encoding = "utf-8" 