from __future__ import annotations as _annotations

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator, TypeAdapter
from typing import Any, List, Dict, Optional

# =============================
#     DATA MODELS (Moved from agent.py)
# =============================

class RankedArticle(BaseModel):
    """Pydantic model for a single ranked article."""
    title: str = Field(..., description="The exact title of the ArXiv article.")
    authors: List[str] = Field(..., min_items=1, description="List of author names for the article.")
    subject: str = Field(..., description="The primary subject category (e.g., cs.AI, physics.hep-th).")
    score_reason: str = Field(..., description="A brief explanation for the assigned relevance score based on the user's profile.")
    relevance_score: int = Field(..., ge=0, le=100, description="Score from 0 to 100 indicating relevance to the user profile. Higher means more relevant.")
    abstract_url: HttpUrl = Field(..., description="The URL link to the article's abstract page on ArXiv.") # Changed to HttpUrl for validation
    html_url: Optional[HttpUrl] = Field(None, description="Optional URL link to an HTML version of the article, if available.") # Changed to HttpUrl, optional
    pdf_url: HttpUrl = Field(..., description="The URL link to the article's PDF version on ArXiv.") # Changed to HttpUrl

    @field_validator("authors", mode="before")
    @classmethod
    def ensure_authors_list(cls, v):
        """Ensure authors is always a List[str], even if malformed input."""
        if isinstance(v, list):
            # Ensure all elements are strings, provide default if empty
            authors = [str(a) for a in v if a]
            return authors if authors else ["Unknown"]
        if isinstance(v, str):
            return [v] if v else ["Unknown"]
        return ["Unknown"]

    @field_validator("abstract_url", "html_url", "pdf_url", mode="before")
    @classmethod
    def normalize_url(cls, v):
        # Pydantic's HttpUrl will handle validation, just ensure it's not empty
        return v if v else None # Return None if empty to allow Optional[HttpUrl]

    @field_validator("title", mode="before")
    @classmethod
    def ensure_title(cls, v):
        return str(v) if v else "Untitled Article"

    # Add arxiv_id extraction logic if needed centrally, maybe as a property or method
    @property
    def arxiv_id(self) -> str | None:
        import re
        if not self.abstract_url:
            return None
        url_str = str(self.abstract_url) # Convert HttpUrl to string
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
    # Inherit fields from RankedArticle or redefine if structure differs significantly
    title: str = Field(..., description="The exact title of the analyzed ArXiv article.")
    authors: List[str] = Field(..., description="List of author names for the analyzed article.")
    subject: str = Field(..., description="The primary subject category of the analyzed article.")
    abstract_url: HttpUrl = Field(..., description="The URL link to the article's abstract page.")
    html_url: Optional[HttpUrl] = Field(None, description="Optional URL link to an HTML version, if available.")
    pdf_url: HttpUrl = Field(..., description="The URL link to the article's PDF version.")
    relevance_score: int = Field(..., ge=0, le=100, description="The previously assigned relevance score (0-100).")
    score_reason: str = Field(..., description="The reason for the relevance score.")

    # Analysis specific fields
    summary: str = Field(..., description="A concise summary of the article's key findings and contributions.")
    importance: str = Field(..., description="Explanation of the article's importance or significance in its field and to the user's interests.")
    recommended_action: str = Field(..., description="Suggested next step for the user regarding this article (e.g., 'Read abstract', 'Skim PDF', 'Deep dive', 'Share with team', 'Ignore').")

    # Optional: Add arxiv_id property if needed here too, or inherit/compose
    @property
    def arxiv_id(self) -> str | None:
        import re
        if not self.abstract_url:
            return None
        url_str = str(self.abstract_url)
        # Simplified pattern assuming abstract_url is canonical
        m = re.search(r'/abs/([^/?&#\s]+)', url_str)
        return m.group(1) if m else None

class UserContext(BaseModel):
    """User profile information (previously in agent.py)."""
    name: str
    title: str # Or role/position
    goals: str # Research interests or objectives

# =============================
#     AGENT STATE MODEL (from Archon suggestion)
# =============================

class AgentStateModel(BaseModel):
    """Model for agent state persistence (structure based on Archon's suggestion)"""
    last_processed_articles: Dict[str, RankedArticle] = Field(default_factory=dict) # Use arxiv_id as key?
    user_preferences: Dict[str, UserContext] = Field(default_factory=dict) # Use user identifier as key?
    session_data: Dict[str, Any] = Field(default_factory=dict) # Keep Any for flexible session data

    # Consider adding methods for saving/loading if state logic resides here
    # Or keep that logic separate in AgentState class in state.py 

# New model for carrying scraped content alongside ranked data
class ScrapedArticle(BaseModel):
    article: RankedArticle = Field(..., description="The ranked article metadata.")
    scraped_content: Optional[str] = Field(None, description="The scraped HTML or text content of the article.")
    scrape_error: Optional[str] = Field(None, description="Error message if scraping failed.") 