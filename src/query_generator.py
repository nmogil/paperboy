from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime
import logfire
from .llm_client import LLMClient
from .metrics import NewsMetrics

class QueryGenerator:
    """Generates personalized and timely news queries."""
    
    def __init__(self):
        self.llm_client = LLMClient()
        self._query_cache = {}
    
    @NewsMetrics.track_query_generation
    async def generate_queries(
        self,
        user_info: Dict[str, Any],
        target_date: Optional[str] = None,
        max_queries: int = 10
    ) -> List[str]:
        """Generate news search queries with temporal awareness."""
        
        # Check cache
        cache_key = f"{user_info.get('name')}:{user_info.get('title')}:{target_date}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Extract direct queries
        direct_queries = self._extract_direct_queries(user_info)
        
        # Add temporal queries if recent
        temporal_queries = self._generate_temporal_queries(user_info, target_date)
        
        # Generate smart queries using LLM
        remaining = max_queries - len(direct_queries) - len(temporal_queries)
        llm_queries = []
        if remaining > 0:
            llm_queries = await self._generate_llm_queries(user_info, target_date, remaining)
        
        # Combine and deduplicate
        all_queries = list(dict.fromkeys(direct_queries + temporal_queries + llm_queries))
        
        result = all_queries[:max_queries]
        self._query_cache[cache_key] = result
        
        logfire.info("Generated news queries", extra={"queries": result, "user": user_info.get('name')})
        return result
    
    def _extract_direct_queries(self, user_info: Dict[str, Any]) -> List[str]:
        """Extract obvious queries from user info."""
        queries = []
        
        # PRIORITY: Use news_interest if provided (this is the main fix)
        news_interest = user_info.get('news_interest', '').strip()
        if news_interest:
            # Add the news interest as the primary query
            queries.append(news_interest)
            
            # Add variations of the news interest
            if len(news_interest.split()) == 1:  # Single word
                queries.append(f'{news_interest} news')
                queries.append(f'{news_interest} latest')
            else:  # Multi-word phrase
                queries.append(f'"{news_interest}"')  # Exact phrase search
                queries.append(f'{news_interest} developments')
            
            logfire.info("Using news_interest as primary query", extra={"news_interest": news_interest})
        
        # Company extraction with better regex
        title = user_info.get('title', '')
        company_patterns = [
            r'\bat\s+([A-Z]\w+(?:\s+[A-Z]\w+)*)',  # at Company Name
            r'(?:^|\s)([A-Z]\w+(?:\s+[A-Z]\w+)*)\s+(?:Inc|Corp|LLC|Ltd)',  # Company Inc
            r'([A-Z]\w+)\s+(?:Technologies|Labs|Research)',  # Tech companies
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, title)
            if match:
                company = match.group(1).strip()
                queries.append(company)
                
                # Add company + domain combinations
                if 'AI' in user_info.get('goals', '').upper() or news_interest.lower() == 'ai':
                    queries.append(f'"{company}" AI artificial intelligence')
                if 'voice' in title.lower():
                    queries.append(f'"{company}" voice technology')
                break
        
        # Role-based queries with better coverage (only if no news_interest)
        if not news_interest:
            role_keywords = {
                'product manager': ['product management AI tools', 'PM artificial intelligence'],
                'engineer': ['software engineering AI', 'AI coding tools'],
                'researcher': ['AI research breakthroughs', 'machine learning advances'],
                'voice': ['voice AI technology', 'speech recognition advances'],
                'data scientist': ['data science AI', 'ML algorithms news'],
                'ml engineer': ['MLOps news', 'machine learning engineering']
            }
            
            title_lower = title.lower()
            for role, keywords in role_keywords.items():
                if role in title_lower:
                    queries.extend(keywords)
                    break
        
        # Interest-based queries (only if no news_interest)
        if not news_interest:
            goals = user_info.get('goals', '')
            if goals:
                # Extract key phrases
                interest_patterns = [
                    r'(?:interested in|working on|focused on|researching)\s+([^,\.]+)',
                    r'(?:developing|building|creating)\s+([^,\.]+)',
                ]
                
                for pattern in interest_patterns:
                    matches = re.findall(pattern, goals, re.IGNORECASE)
                    for match in matches:
                        queries.append(match.strip())
        
        return queries[:7]  # Increased limit to allow for news_interest variations
    
    def _generate_temporal_queries(
        self,
        user_info: Dict[str, Any],
        target_date: Optional[str]
    ) -> List[str]:
        """Generate time-sensitive queries."""
        temporal_queries = []
        
        if target_date:
            date_obj = datetime.strptime(target_date, "%Y-%m-%d")
            days_ago = (datetime.now() - date_obj).days
            
            # Recent news needs different queries
            if days_ago <= 7:
                base_interests = self._extract_interests(user_info)
                for interest in base_interests[:2]:  # Top 2 interests
                    temporal_queries.append(f'"{interest}" latest news {date_obj.year}')
                    temporal_queries.append(f'{interest} announcement breakthrough')
        
        return temporal_queries
    
    def _extract_interests(self, user_info: Dict[str, Any]) -> List[str]:
        """Extract core interests from user info."""
        interests = []
        
        # From goals
        goals = user_info.get('goals', '')
        if 'AI' in goals.upper():
            interests.append('artificial intelligence')
        if 'machine learning' in goals.lower():
            interests.append('machine learning')
        if 'voice' in goals.lower():
            interests.append('voice technology')
        if 'llm' in goals.lower() or 'language model' in goals.lower():
            interests.append('large language models')
        
        return interests
    
    async def _generate_llm_queries(
        self,
        user_info: Dict[str, Any],
        target_date: Optional[str],
        num_queries: int
    ) -> List[str]:
        """Use LLM to generate smart queries."""
        
        # Add temporal context to prompt
        temporal_context = ""
        if target_date:
            date_obj = datetime.strptime(target_date, "%Y-%m-%d")
            days_ago = (datetime.now() - date_obj).days
            if days_ago <= 1:
                temporal_context = "Focus on breaking news, announcements, and latest developments."
            elif days_ago <= 7:
                temporal_context = "Focus on recent developments, weekly updates, and new releases."
            else:
                temporal_context = f"Focus on news from around {target_date}."
        
        system_prompt = f"""Generate {num_queries} specific news search queries for this user.
{temporal_context}

Create queries that would find:
1. Breaking news about their company or competitors
2. Industry developments relevant to their role
3. New technologies or tools in their field
4. Research breakthroughs related to their interests
5. Business news affecting their sector

Guidelines:
- Use quotes for exact phrases when needed
- Include company names and specific technologies
- Mix broad and specific queries
- Consider Boolean operators for better results
- Keep queries under 10 words each

Return ONLY a JSON array of query strings."""
        
        user_prompt = f"""
User: {user_info.get('name')}
Title: {user_info.get('title')}
Goals: {user_info.get('goals')}
Date context: {target_date or 'today'}

Generate news search queries that would find relevant, timely news for this person."""
        
        try:
            response = await self.llm_client._call_llm(
                system_prompt,
                user_prompt,
                temperature=0.8,  # Higher for more variety
                max_tokens=500
            )
            
            queries = json.loads(response)
            return queries if isinstance(queries, list) else []
        except Exception as e:
            logfire.error("Failed to generate LLM queries", extra={"error": str(e)})
            # Fallback queries
            return [
                "AI technology news",
                "machine learning breakthrough",
                "tech industry updates"
            ]