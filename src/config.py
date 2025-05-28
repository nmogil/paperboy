from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
from typing import Optional

class Settings(BaseSettings):
    """
    Centralized configuration for the Arxiv agent.
    All values are loaded from environment variables (or config/.env) with type safety.
    Missing mandatory fields will raise validation errors at startup.
    """
    openai_api_key: str = Field(..., validation_alias='OPENAI_API_KEY')
    openai_model: str = Field(default='gpt-4o', validation_alias='OPENAI_MODEL')
    top_n_articles: int = Field(default=5, validation_alias='TOP_N_ARTICLES')
    ranking_input_max_articles: int = Field(default=20, validation_alias='RANKING_INPUT_MAX_ARTICLES', description="Maximum number of raw articles to send to the LLM for the ranking step.")
    
    use_lightweight: bool = Field(default=True, validation_alias='USE_LIGHTWEIGHT', description="Use lightweight version without Playwright")
    
    logfire_token: Optional[str] = Field(default=None, validation_alias='LOGFIRE_TOKEN', description="Logfire token for monitoring")

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), '..', 'config', '.env'),
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings() 