from __future__ import annotations as _annotations

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator, TypeAdapter
from typing import Any, List, Dict, Optional
from enum import Enum
from datetime import datetime


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ContentType(str, Enum):
    PAPER = "paper"
    NEWS = "news"

class DigestStatus(BaseModel):
    status: TaskStatus
    message: str
    result: Optional[str] = None
    articles: Optional[List['ArticleAnalysis']] = None


class RankedArticle(BaseModel):
    """Pydantic model for a single ranked article."""
    title: str = Field(..., description="The exact title of the article.")
    authors: List[str] = Field(..., min_items=1, description="List of author names for the article.")
    subject: str = Field(..., description="The primary subject category (e.g., cs.AI, physics.hep-th) or 'news' for news articles.")
    score_reason: str = Field(..., description="A brief explanation for the assigned relevance score based on the user's profile.")
    relevance_score: int = Field(..., ge=0, le=100, description="Score from 0 to 100 indicating relevance to the user profile. Higher means more relevant.")
    abstract_url: HttpUrl = Field(..., description="The URL link to the article's abstract page on ArXiv or news URL.")
    html_url: Optional[HttpUrl] = Field(None, description="Optional URL link to an HTML version of the article, if available.")
    pdf_url: Optional[HttpUrl] = Field(None, description="The URL link to the article's PDF version on ArXiv. Not applicable for news.")
    
    # News-specific fields
    type: ContentType = Field(default=ContentType.PAPER, description="Content type: paper or news")
    source: Optional[str] = Field(None, description="News source name")
    published_at: Optional[str] = Field(None, description="Publication timestamp")
    url_to_image: Optional[str] = Field(None, description="Article image URL")
    full_content: Optional[str] = Field(None, description="Full extracted content")
    content_preview: Optional[str] = Field(None, description="Content preview")
    extraction_success: Optional[bool] = Field(None, description="Whether content extraction succeeded")
    relevance_score_normalized: Optional[float] = Field(None, ge=0, le=1, description="Normalized relevance score")

    @field_validator("authors", mode="before")
    @classmethod
    def ensure_authors_list(cls, v):
        """Ensure authors is always a List[str], even if malformed input."""
        if isinstance(v, list):
            authors = [str(a) for a in v if a]
            return authors if authors else ["Unknown"]
        if isinstance(v, str):
            return [v] if v else ["Unknown"]
        return ["Unknown"]

    @field_validator("abstract_url", "html_url", "pdf_url", mode="before")
    @classmethod
    def normalize_url(cls, v):
        # Pydantic's HttpUrl will handle validation, just ensure it's not empty
        return v if v else None

    @field_validator("title", mode="before")
    @classmethod
    def ensure_title(cls, v):
        return str(v) if v else "Untitled Article"
    
    @field_validator('published_at')
    @classmethod
    def parse_published_date(cls, v):
        if v and isinstance(v, str):
            try:
                # Parse ISO format
                return datetime.fromisoformat(v.replace('Z', '+00:00')).isoformat()
            except:
                return v
        return v

    @property
    def arxiv_id(self) -> str | None:
        import re
        if not self.abstract_url:
            return None
        url_str = str(self.abstract_url)
        for pattern in [
            r'/abs/([^/?&#\s]+)',
            r'/pdf/([^/?&#\s]+?)(?:\.pdf)?',
            r'/html/([^/?&#\s]+)',
            r'arxiv.org[:/]([^/?&#\s]+)'
        ]:
            m = re.search(pattern, url_str)
            if m:
                return m.group(1)
        return None

class ArticleAnalysis(BaseModel):
    """Analysis result for a single article."""
    title: str = Field(..., description="The exact title of the analyzed article.")
    authors: List[str] = Field(..., description="List of author names for the analyzed article.")
    subject: str = Field(..., description="The primary subject category of the analyzed article.")
    abstract_url: HttpUrl = Field(..., description="The URL link to the article's abstract page.")
    html_url: Optional[HttpUrl] = Field(None, description="Optional URL link to an HTML version, if available.")
    pdf_url: Optional[HttpUrl] = Field(None, description="The URL link to the article's PDF version. Not applicable for news.")
    relevance_score: int = Field(..., ge=0, le=100, description="The previously assigned relevance score (0-100).")
    score_reason: str = Field(..., description="The reason for the relevance score.")
    
    # Content type field to distinguish papers from news
    type: ContentType = Field(default=ContentType.PAPER, description="Content type: paper or news")
    source: Optional[str] = Field(None, description="News source name")
    published_at: Optional[str] = Field(None, description="Publication timestamp")
    url_to_image: Optional[str] = Field(None, description="Article image URL")

    # Analysis specific fields
    summary: str = Field(..., description="A concise summary of the article's key findings and contributions.")
    importance: str = Field(..., description="Explanation of the article's importance or significance in its field and to the user's interests.")
    recommended_action: str = Field(..., description="Suggested next step for the user regarding this article (e.g., 'Read abstract', 'Skim PDF', 'Deep dive', 'Share with team', 'Ignore').")
    key_findings: List[str] = Field(..., description="List of 3-5 main contributions and findings from the paper.")
    relevance_to_user: str = Field(..., description="How this paper connects to the user's work and research interests.")
    technical_details: str = Field(..., description="Important methods, techniques, or results from the paper.")
    potential_applications: str = Field(..., description="How the user might apply this research in their own work.")
    critical_notes: Optional[str] = Field(None, description="Limitations, concerns, or critical observations about the paper.")
    follow_up_suggestions: Optional[str] = Field(None, description="Related papers or next steps for the user to explore.")

    @property
    def arxiv_id(self) -> str | None:
        import re
        if not self.abstract_url:
            return None
        url_str = str(self.abstract_url)
        m = re.search(r'/abs/([^/?&#\s]+)', url_str)
        return m.group(1) if m else None

class UserContext(BaseModel):
    """User profile information (previously in agent.py)."""
    name: str
    title: str
    goals: str
    news_interest: Optional[str] = Field(None, description="Specific topic of interest for news articles")


class AgentStateModel(BaseModel):
    """Model for agent state persistence (structure based on Archon's suggestion)"""
    last_processed_articles: Dict[str, RankedArticle] = Field(default_factory=dict)
    user_preferences: Dict[str, UserContext] = Field(default_factory=dict)
    session_data: Dict[str, Any] = Field(default_factory=dict)

 

class ScrapedArticle(BaseModel):
    article: RankedArticle = Field(..., description="The ranked article metadata.")
    scraped_content: Optional[str] = Field(None, description="The scraped HTML or text content of the article.")
    scrape_error: Optional[str] = Field(None, description="Error message if scraping failed.") 