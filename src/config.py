# src/config.py

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
    log_level: str = Field(default='INFO', validation_alias='LOG_LEVEL')
    crawler_timeout: int = Field(default=25000, validation_alias='CRAWLER_TIMEOUT')
    agent_retries: int = Field(default=2, validation_alias='AGENT_RETRIES')
    arxiv_file: str = Field(default='arxiv_papers.json', validation_alias='ARXIV_FILE')
    top_n_articles: int = Field(default=5, validation_alias='TOP_N_ARTICLES')
    analysis_content_max_chars: int = Field(default=8000, validation_alias='ANALYSIS_CONTENT_MAX_CHARS', description="Maximum characters of article content to send for analysis.")
    ranking_input_max_articles: int = Field(default=20, validation_alias='RANKING_INPUT_MAX_ARTICLES', description="Maximum number of raw articles to send to the LLM for the ranking step.")
    target_date: Optional[str] = Field(default=None, validation_alias='TARGET_DATE', description="Target date for fetching arXiv articles (YYYY-MM-DD). Defaults to today if not set.")
    
    # Output file paths
    output_dir: str = Field(default='output', validation_alias='OUTPUT_DIR')
    ranking_output_file: str = Field(default='output/ranking_results.json', validation_alias='RANKING_OUTPUT_FILE')
    analysis_output_file: str = Field(default='output/analysis_results.json', validation_alias='ANALYSIS_OUTPUT_FILE')

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), '..', 'config', '.env'),  # Adjust as appropriate if structure changes
        env_file_encoding='utf-8',
        extra='ignore'
    )

# Single, importable settings object
settings = Settings() 