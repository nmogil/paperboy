from __future__ import annotations as _annotations

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator, TypeAdapter
from typing import Any, List, Dict, Optional

# =============================
#     DATA MODELS (Moved from agent.py)
# =============================

class RankedArticle(BaseModel):
    """Pydantic model for a single ranked article."""
    title: str
    authors: List[str] = Field(min_items=1)
    subject: str
    score_reason: str
    relevance_score: int = Field(ge=0, le=100)
    abstract_url: HttpUrl # Changed to HttpUrl for validation
    html_url: Optional[HttpUrl] = None # Changed to HttpUrl, optional
    pdf_url: HttpUrl # Changed to HttpUrl

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
    title: str
    authors: List[str]
    subject: str
    abstract_url: HttpUrl
    html_url: Optional[HttpUrl] = None
    pdf_url: HttpUrl
    relevance_score: int = Field(ge=0, le=100)
    score_reason: str

    # Analysis specific fields
    summary: str
    importance: str
    recommended_action: str

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