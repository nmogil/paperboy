from typing import Optional, List, Dict, Any
from pydantic import BaseModel, HttpUrl, Field
from .models import UserContext

class GenerateDigestRequest(BaseModel):
    """Request model for generating a digest."""
    user_info: UserContext
    target_date: Optional[str] = None
    top_n_articles: Optional[int] = None
    callback_url: Optional[HttpUrl] = None
    categories: List[str] = Field(default=["cs.AI", "cs.LG"])

class GenerateDigestResponse(BaseModel):
    """Response model for the digest generation request."""
    task_id: str
    status: str
    message: str
    status_url: Optional[str] = None  # Made optional since we don't return it in new implementation

class DigestStatusResponse(BaseModel):
    """Response model for checking digest generation status."""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: str  # Status message
    result: Optional[str] = None  # HTML content when completed
    error: Optional[str] = None  # Error message if failed (deprecated, use message)
    callback_url: Optional[str] = None  # Optional callback_url
    articles: Optional[List[Dict[str, Any]]] = None  # Analyzed articles when completed 