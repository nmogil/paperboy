from typing import Optional, List, Dict, Any
from pydantic import BaseModel, HttpUrl, Field
from .models import UserContext

class GenerateDigestRequest(BaseModel):
    """Request model for generating a digest."""
    user_info: UserContext
    target_date: Optional[str] = None
    top_n_articles: Optional[int] = None
    top_n_news: Optional[int] = None
    callback_url: Optional[HttpUrl] = None
    categories: List[str] = Field(default=["cs.AI", "cs.LG"])
    digest_sources: Optional[Dict[str, bool]] = Field(
        default=None,
        description="Control which sources to include in digest generation. Defaults to all enabled sources if not specified."
    )
    source_date: Optional[str] = Field(
        default=None,
        description="Date of pre-fetched sources to use for digest generation (YYYY-MM-DD). If not provided, uses latest available sources."
    )

class GenerateDigestResponse(BaseModel):
    """Response model for the digest generation request."""
    task_id: str
    status: str
    message: str
    status_url: Optional[str] = None

class DigestStatusResponse(BaseModel):
    """Response model for checking digest generation status."""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: str  # Status message
    result: Optional[str] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    articles: Optional[List[Dict[str, Any]]] = None

class FetchSourcesRequest(BaseModel):
    """Request model for fetching daily sources."""
    source_date: str = Field(
        description="Date to fetch sources for in YYYY-MM-DD format"
    )
    callback_url: Optional[HttpUrl] = Field(
        default=None,
        description="Optional callback URL for async completion notification"
    )

class FetchSourcesResponse(BaseModel):
    """Response model for fetch sources request."""
    task_id: str
    status: str
    message: str
    source_date: str
    status_url: Optional[str] = None

class FetchStatusResponse(BaseModel):
    """Response model for checking fetch task status."""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: str
    source_date: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None 