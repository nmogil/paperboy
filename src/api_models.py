from typing import Optional
from pydantic import BaseModel, HttpUrl
from .models import UserContext

class GenerateDigestRequest(BaseModel):
    """Request model for generating a digest."""
    user_info: UserContext
    target_date: Optional[str] = None
    top_n_articles: Optional[int] = None
    callback_url: Optional[HttpUrl] = None

class GenerateDigestResponse(BaseModel):
    """Response model for the digest generation request."""
    task_id: str
    status: str
    message: str
    status_url: str

class DigestStatusResponse(BaseModel):
    """Response model for checking digest generation status."""
    task_id: str
    status: str  # "PENDING", "PROCESSING", "COMPLETED", "FAILED"
    result: Optional[str] = None  # HTML content when completed
    error: Optional[str] = None  # Error message if failed
    callback_url: Optional[str] = None  # Add optional callback_url (as string, since it's retrieved from storage) 