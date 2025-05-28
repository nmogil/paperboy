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