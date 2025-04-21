# src/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):
    """
    Centralized configuration for the Arxiv agent.
    All values are loaded from environment variables (or config/.env) with type safety.
    Missing mandatory fields will raise validation errors at startup.
    """
    openai_api_key: str = Field(..., validation_alias='OPENAI_API_KEY')
    openai_model: str = Field(default='gpt-4o', validation_alias='OPENAI_MODEL')
    log_level: str = Field(default='INFO', validation_alias='LOG_LEVEL')
    crawler_timeout: int = Field(default=25000, validation_alias='CRAWLER_TIMEOUT')
    agent_retries: int = Field(default=2, validation_alias='AGENT_RETRIES')
    arxiv_file: str = Field("arxiv_papers.json", validation_alias="ARXIV_FILE")
    top_n_articles: int = Field(5, validation_alias="TOP_N_ARTICLES")

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), '..', 'config', '.env'),  # Adjust as appropriate if structure changes
        env_file_encoding='utf-8',
        extra='ignore'
    )

# Single, importable settings object
settings = Settings() 