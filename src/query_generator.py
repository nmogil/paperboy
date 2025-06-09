from typing import List, Dict, Any, Optional
import logfire
from .metrics import NewsMetrics

class QueryGenerator:
    """Generates personalized and timely news queries."""
    
    def __init__(self):
        # Simplified - no LLM client or caching needed
        pass
    
    @NewsMetrics.track_query_generation
    async def generate_queries(
        self,
        user_info: Dict[str, Any],
        target_date: Optional[str] = None,
        max_queries: int = 10
    ) -> List[str]:
        """Generate news search queries - FORCED to always return AI query."""
        
        # FORCED: Always return ["AI"] regardless of user input
        logfire.info("Forced AI query for news search", extra={
            "query": "AI", 
            "user": user_info.get('name'),
            "ignored_news_interest": user_info.get('news_interest'),
            "ignored_target_date": target_date
        })
        
        return ["AI"]