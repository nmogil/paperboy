from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
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
    top_n_news: int = Field(default=5, validation_alias='TOP_N_NEWS', description="Number of news articles to include in digest")
    ranking_input_max_articles: int = Field(default=20, validation_alias='RANKING_INPUT_MAX_ARTICLES', description="Maximum number of raw articles to send to the LLM for the ranking step.")
    
    use_lightweight: bool = Field(default=True, validation_alias='USE_LIGHTWEIGHT', description="Use lightweight version without Playwright")
    
    logfire_token: Optional[str] = Field(default=None, validation_alias='LOGFIRE_TOKEN', description="Logfire token for monitoring")
    
    # News API Configuration
    newsapi_key: Optional[str] = Field(None, validation_alias='NEWSAPI_KEY')
    tavily_api_key: Optional[str] = Field(None, validation_alias='TAVILY_API_KEY')
    
    # News fetching parameters
    news_enabled: bool = Field(default=True, validation_alias='NEWS_ENABLED')
    news_max_articles: int = Field(default=50, validation_alias='NEWS_MAX_ARTICLES')
    news_max_extract: int = Field(default=10, validation_alias='NEWS_MAX_EXTRACT')  # Limit extraction
    news_language: str = Field(default='en', validation_alias='NEWS_LANGUAGE')
    news_sort_by: str = Field(default='relevancy', validation_alias='NEWS_SORT_BY')
    news_search_in: str = Field(default='title,description', validation_alias='NEWS_SEARCH_IN')
    news_sources: Optional[str] = Field(None, validation_alias='NEWS_SOURCES')  # Comma-separated
    news_exclude_domains: Optional[str] = Field(None, validation_alias='NEWS_EXCLUDE_DOMAINS')
    
    # Content extraction
    extract_max_concurrent: int = Field(default=3, validation_alias='EXTRACT_MAX_CONCURRENT')
    extract_timeout: int = Field(default=10, validation_alias='EXTRACT_TIMEOUT')
    
    # Rate limiting and delays
    ranking_delay: float = Field(default=0.7, validation_alias='RANKING_DELAY', description="Delay between ranking API calls in seconds")
    summary_delay: float = Field(default=0.7, validation_alias='SUMMARY_DELAY', description="Delay between summary API calls in seconds")
    summary_max_concurrent: int = Field(default=3, validation_alias='SUMMARY_MAX_CONCURRENT', description="Max concurrent summary generation")
    
    # Supabase Integration (essential for Cloud Run)
    supabase_url: Optional[str] = Field(None, validation_alias='SUPABASE_URL')
    supabase_key: Optional[str] = Field(None, validation_alias='SUPABASE_KEY')
    use_supabase: bool = Field(default=True, validation_alias='USE_SUPABASE', description="Use Supabase for distributed state management (recommended for Cloud Run)")
    
    # Caching
    news_cache_ttl: int = Field(default=3600, validation_alias='NEWS_CACHE_TTL')  # 1 hour
    
    # Monitoring configuration
    news_metrics_enabled: bool = Field(default=True, validation_alias='NEWS_METRICS_ENABLED')
    
    @field_validator('newsapi_key', 'tavily_api_key')
    def validate_api_keys(cls, v, info):
        # Only validate if news is enabled and we're checking the keys
        if info.data.get('news_enabled', True) and not v:
            field_name = info.field_name
            # Only raise error if news is enabled but keys are missing
            import os
            if os.getenv('NEWS_ENABLED', 'true').lower() == 'true':
                print(f"Warning: {field_name} is not set but news_enabled=True")
        return v

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), '..', 'config', '.env'),
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings() 