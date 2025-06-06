import json
import time
from typing import Any, Dict, List, Optional, Union
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import logfire
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime

from .config import settings
from .models import RankedArticle, ArticleAnalysis

# Structured output schemas for ranking
class ArticleRanking(BaseModel):
    """Schema for complete ranked article in structured output - matches RankedArticle structure."""
    title: str = Field(description="Paper or article title")
    authors: List[str] = Field(description="List of authors")
    subject: str = Field(description="Subject category or 'news'")
    abstract_url: str = Field(description="Article URL")
    relevance_score: int = Field(ge=0, le=100, description="Relevance score 0-100")
    score_reason: str = Field(description="Brief explanation of relevance")
    
    # Optional URLs
    html_url: Optional[str] = Field(default=None, description="HTML URL if available")
    pdf_url: Optional[str] = Field(default=None, description="PDF URL if available")
    
    # Content type and news-specific fields
    type: str = Field(default="paper", description="Content type: 'paper' or 'news'")
    source: Optional[str] = Field(default=None, description="News source name")
    published_at: Optional[str] = Field(default=None, description="Publication timestamp")
    url_to_image: Optional[str] = Field(default=None, description="Article image URL")
    full_content: Optional[str] = Field(default=None, description="Full extracted content")
    content_preview: Optional[str] = Field(default=None, description="Content preview")
    extraction_success: Optional[bool] = Field(default=None, description="Whether content extraction succeeded")
    relevance_score_normalized: Optional[float] = Field(default=None, description="Normalized relevance score")
    
    # Preserve any additional metadata fields
    url: Optional[str] = Field(default=None, description="Alternative URL field for news articles")

class RankingResponse(BaseModel):
    """Schema for ranking response containing list of ranked articles."""
    articles: List[ArticleRanking] = Field(min_items=1, max_items=20, description="List of ranked articles")

class LLMClient:
    """Simple LLM client with retry logic and structured outputs."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """Make LLM call with retry logic and performance monitoring using Responses API."""
        start_time = time.time()
        try:
            # Format input for Responses API - combine system and user prompts
            input_content = f"{system_prompt}\n\n{user_prompt}"
            
            response = await self.client.responses.create(
                model=self.model,
                input=input_content,
                temperature=temperature
            )
            
            # Extract content from Responses API format
            if response.output and len(response.output) > 0:
                output_item = response.output[0]
                if hasattr(output_item, 'content') and len(output_item.content) > 0:
                    content = output_item.content[0].text
                else:
                    logfire.error("Unexpected response format: no content in output")
                    raise ValueError("No content in response output")
            else:
                logfire.error("Unexpected response format: no output")
                raise ValueError("No output in response")

            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            return content.strip()
        except Exception as e:
            logfire.error(f"LLM call failed: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            logfire.info(f"LLM call completed", extra={
                "duration": duration,
                "model": self.model,
                "tokens": max_tokens,
                "temperature": temperature
            })

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _call_llm_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: BaseModel,
        temperature: float = 0.3,
        fallback_to_manual: bool = True
    ) -> Union[BaseModel, List[Dict[str, Any]]]:
        """Make structured LLM call with Pydantic validation and fallback to manual parsing."""
        start_time = time.time()
        
        try:
            # Try structured output with Responses API
            logfire.info(f"Attempting structured output with model: {response_model.__name__}")
            
            response = await self.client.responses.parse(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt,
                text_format=response_model,
                temperature=temperature
            )
            
            # Access the parsed Pydantic object directly
            if hasattr(response, 'output_parsed') and response.output_parsed:
                logfire.info(f"Successfully received structured output: {type(response.output_parsed)}")
                return response.output_parsed
            else:
                raise ValueError("No parsed output in structured response")
                
        except Exception as e:
            logfire.warning(f"Structured output failed: {str(e)}")
            
            if fallback_to_manual:
                logfire.info("Falling back to manual JSON parsing")
                try:
                    # Fallback to manual parsing
                    text_response = await self._call_llm(system_prompt, user_prompt, temperature)
                    
                    # Attempt to parse as JSON and convert to expected format
                    data = json.loads(text_response)
                    
                    # Handle different response formats for fallback
                    if isinstance(data, dict) and 'articles' in data:
                        return data['articles']
                    elif isinstance(data, list):
                        return data
                    else:
                        logfire.error(f"Unexpected fallback response format: {type(data)}")
                        return []
                        
                except Exception as fallback_error:
                    logfire.error(f"Manual parsing fallback also failed: {str(fallback_error)}")
                    raise ValueError(f"Both structured output and manual parsing failed: {str(e)} | {str(fallback_error)}")
            else:
                raise
        finally:
            duration = time.time() - start_time
            logfire.info(f"Structured LLM call completed", extra={
                "duration": duration,
                "model": self.model,
                "response_model": response_model.__name__,
                "temperature": temperature
            })

    async def rank_articles(
        self,
        articles: List[Dict[str, Any]],
        user_info: Dict[str, Any],
        top_n: int
    ) -> List[RankedArticle]:
        """Rank articles based on user profile using structured outputs with complete article data."""
        
        # Extract and structure user interests more comprehensively  
        primary_goals = user_info.get('goals', 'AI Research')
        user_title = user_info.get('title', 'Researcher')
        
        system_prompt = f"""You are an expert research paper analyst. Your task is to rank academic papers based on their relevance to a user's research interests and background.

Below is a list of articles in JSON format with complete metadata. Select the {top_n} most relevant articles based on the user profile.

USER PROFILE ANALYSIS:
- Primary Role: {user_title}
- Primary research/learning goals: {primary_goals}

RANKING STRATEGY for Research Papers:
1. **Technical Relevance**: How directly does this paper relate to "{primary_goals}"?
2. **Methodological Value**: Does this introduce methods/techniques applicable to their work?
3. **Foundational Knowledge**: Does this advance understanding in their field of interest?
4. **Practical Applications**: Can insights from this paper be applied in their role as {user_title}?
5. **Learning Value**: Will this expand their knowledge in meaningful ways?

You must return a structured response with an "articles" field containing exactly {top_n} selected articles. For each article:
- Return ALL the original fields from the input article
- Add relevance_score: Relevance score 0-100 (integer)
- Add score_reason: Brief explanation of relevance that explains WHY this serves their specific goals and role
- Ensure the article data is complete and unchanged except for the ranking fields

CRITICAL INSTRUCTIONS:
1. You MUST select exactly {top_n} articles (not fewer, not more)
2. Preserve ALL original article metadata and fields
3. Even if only 1 article seems highly relevant, find {top_n} articles with varying relevance scores
4. Return them in descending order by relevance score
5. Explain relevance in terms of the user's specific goals: "{primary_goals}"
6. Do NOT modify any existing fields except to add relevance_score and score_reason"""

        user_prompt = f"""User profile:
Name: {user_info.get('name', 'Researcher')}
Title: {user_title}
Primary Goals: {primary_goals}

Articles:
{json.dumps(articles, indent=2)}"""

        try:
            # Try structured output first
            response = await self._call_llm_structured(
                system_prompt, 
                user_prompt, 
                RankingResponse,
                temperature=0.3
            )
            
            if isinstance(response, RankingResponse):
                # Successful structured output - articles are already complete
                logfire.info(f"Structured output success: got {len(response.articles)} complete articles")
                
                ranked_articles = []
                for article in response.articles[:top_n]:
                    try:
                        # Convert ArticleRanking to dict and create RankedArticle
                        article_data = article.model_dump()
                        # Ensure we have the required field mappings for RankedArticle
                        if 'score' in article_data and 'relevance_score' not in article_data:
                            article_data['relevance_score'] = article_data.pop('score')
                        if 'reasoning' in article_data and 'score_reason' not in article_data:
                            article_data['score_reason'] = article_data.pop('reasoning')
                        
                        ranked_articles.append(RankedArticle(**article_data))
                    except Exception as e:
                        item_str = str(article.model_dump())[:200].replace('{', '{{').replace('}', '}}')
                        logfire.error(f"Failed to create RankedArticle from structured output: {str(e)}, data: {item_str}")
                        continue
                
                logfire.info(f"Successfully processed {len(ranked_articles)} complete articles via structured output")
                return ranked_articles
                
            elif isinstance(response, list):
                # Fallback response as list of dicts - should already be complete
                logfire.info(f"Fallback response: got {len(response)} complete articles as list")
                
                ranked_articles = []
                for item in response[:top_n]:
                    try:
                        # Ensure we have the required field mappings for RankedArticle
                        if 'score' in item and 'relevance_score' not in item:
                            item['relevance_score'] = item.pop('score')
                        if 'reasoning' in item and 'score_reason' not in item:
                            item['score_reason'] = item.pop('reasoning')
                        
                        ranked_articles.append(RankedArticle(**item))
                    except Exception as e:
                        item_str = str(item)[:200].replace('{', '{{').replace('}', '}}')
                        logfire.error(f"Failed to create RankedArticle from fallback: {str(e)}, data: {item_str}")
                        continue
                
                return ranked_articles
            else:
                logfire.error(f"Unexpected response type from structured call: {type(response)}")
                return []
                
        except Exception as e:
            logfire.error(f"Structured ranking failed completely: {str(e)}")
            raise ValueError(f"Article ranking failed: {str(e)}")

    async def rank_mixed_content(
        self,
        content: List[Dict[str, Any]],
        user_info: Dict[str, Any],
        top_n: int,
        weight_recency: bool = True
    ) -> List[RankedArticle]:
        """Rank mixed content (papers and news) with type-aware scoring using structured outputs with complete content data."""
        
        # Extract and structure user interests more comprehensively
        primary_goals = user_info.get('goals', 'AI Research')
        news_interest = user_info.get('news_interest', '')
        user_title = user_info.get('title', 'Researcher')
        
        # Build more comprehensive interest context
        interest_context = f"Primary research/learning goals: {primary_goals}"
        if news_interest:
            interest_context += f"\nSpecific news interest: {news_interest}"
        
        system_prompt = f"""You are an expert analyst who ranks both research papers and news articles based on their relevance to a user's research interests and current work.

Below is a list of content items in JSON format with complete metadata, which may include both academic papers and news articles. Select the {top_n} most relevant items based on the user profile.

USER PROFILE ANALYSIS:
- Primary Role: {user_title}
- {interest_context}

RANKING STRATEGY - Balance multiple dimensions of relevance:

1. **Research Papers**: Score based on:
   - Technical relevance to primary goals ({primary_goals})
   - Methodological contributions applicable to user's work
   - Foundational knowledge that advances understanding
   - Practical applications for their role

2. **News Articles**: Score based on:
   - Industry developments affecting their work
   - Company/technology mentions relevant to their interests
   - Emerging trends that impact their field
   - Timely information for professional awareness

3. **Balanced Selection Rules**:
   - PRIORITIZE PRIMARY GOALS: Papers related to "{primary_goals}" should be weighted heavily
   - SUPPLEMENT WITH NEWS: Include relevant industry news that connects to their work
   - {"RECENCY BONUS: Give slight preference to recent developments when equally relevant" if weight_recency else "IGNORE RECENCY: Focus purely on relevance regardless of date"}
   - AIM FOR DIVERSITY: Unless user has very narrow interests, include both research depth AND industry awareness
   - QUALITY OVER TYPE: Choose the most relevant content regardless of paper/news ratio

4. **Special Considerations**:
   - For technical roles (like "Tester", "Engineer", "Developer"): Balance cutting-edge research with practical industry developments
   - For research roles: Emphasize papers but include industry trends that inform research direction
   - When news_interest is specified: Include relevant news but don't let it dominate over primary research goals

You must return a structured response with an "articles" field containing exactly {top_n} selected items. For each item:
- Return ALL the original fields from the input content
- Add relevance_score: Relevance score 0-100 (integer)
- Add score_reason: Brief explanation of relevance that mentions WHY this specific content serves their goals
- Ensure the content data is complete and unchanged except for the ranking fields

CRITICAL INSTRUCTIONS:
1. You MUST select exactly {top_n} items (not fewer, not more)
2. Preserve ALL original content metadata and fields
3. SELECT THE MOST RELEVANT CONTENT regardless of type - don't artificially balance
4. Explain relevance in terms of the user's specific role and goals
5. Consider both immediate applicability and long-term learning value
6. Return them in descending order by relevance score
7. Do NOT modify any existing fields except to add relevance_score and score_reason"""

        user_prompt = f"""User profile:
Name: {user_info.get('name', 'Researcher')}
Title: {user_title}
Primary Goals: {primary_goals}
{f"News Interest: {news_interest}" if news_interest else ""}

Content items:
{json.dumps(content, indent=2)}"""

        try:
            # Try structured output first
            response = await self._call_llm_structured(
                system_prompt, 
                user_prompt, 
                RankingResponse,
                temperature=0.3
            )
            
            if isinstance(response, RankingResponse):
                # Successful structured output - content is already complete
                logfire.info(f"Mixed content structured output success: got {len(response.articles)} complete items")
                
                ranked_articles = []
                for article in response.articles[:top_n]:
                    try:
                        # Convert ArticleRanking to dict and create RankedArticle
                        article_data = article.model_dump()
                        # Ensure we have the required field mappings for RankedArticle
                        if 'score' in article_data and 'relevance_score' not in article_data:
                            article_data['relevance_score'] = article_data.pop('score')
                        if 'reasoning' in article_data and 'score_reason' not in article_data:
                            article_data['score_reason'] = article_data.pop('reasoning')
                        
                        # Ensure proper type field
                        if 'type' not in article_data or not article_data['type']:
                            article_data['type'] = 'news' if article_data.get('subject') == 'news' else 'paper'
                        
                        ranked_articles.append(RankedArticle(**article_data))
                    except Exception as e:
                        item_str = str(article.model_dump())[:200].replace('{', '{{').replace('}', '}}')
                        logfire.error(f"Failed to create RankedArticle from mixed structured output: {str(e)}, data: {item_str}")
                        continue

                logfire.info(f"Successfully ranked {len(ranked_articles)} complete mixed content items via structured output")
                return ranked_articles
                
            elif isinstance(response, list):
                # Fallback response as list of dicts - should already be complete
                logfire.info(f"Mixed content fallback response: got {len(response)} complete items as list")
                
                ranked_articles = []
                for item in response[:top_n]:
                    try:
                        # Ensure we have the required field mappings for RankedArticle
                        if 'score' in item and 'relevance_score' not in item:
                            item['relevance_score'] = item.pop('score')
                        if 'reasoning' in item and 'score_reason' not in item:
                            item['score_reason'] = item.pop('reasoning')
                        
                        # Ensure proper type field
                        if 'type' not in item or not item['type']:
                            item['type'] = 'news' if item.get('subject') == 'news' else 'paper'
                        
                        ranked_articles.append(RankedArticle(**item))
                    except Exception as e:
                        item_str = str(item)[:200].replace('{', '{{').replace('}', '}}')
                        logfire.error(f"Failed to create RankedArticle from mixed fallback: {str(e)}, data: {item_str}")
                        continue

                return ranked_articles
            else:
                logfire.error(f"Unexpected response type from mixed content structured call: {type(response)}")
                return []
                
        except Exception as e:
            logfire.error(f"Mixed content structured ranking failed: {str(e)}")
            # Fallback to regular ranking
            logfire.info("Falling back to regular article ranking for mixed content")
            return await self.rank_articles(content, user_info, top_n)

    async def analyze_article(
        self,
        article_content: str,
        article_metadata: Dict[str, Any],
        user_info: Dict[str, Any]
    ) -> ArticleAnalysis:
        """Analyze a single article with simplified language."""
        system_prompt = """Analyze the following article for the user based on their profile.

Your response MUST use simple, conversational language. Avoid academic jargon. Write as if explaining to a colleague over coffee.

Focus on:
1. What this actually means in practical terms
2. How it directly impacts their daily work
3. What specific action they should take

Your response MUST be **only** a valid JSON object structured exactly as follows:

{
  "summary": "<One clear sentence explaining what this is about. No jargon.>",
  "importance": "<Why they should care, in plain English. Be specific to their role.>",
  "recommended_action": "<Specific next step: 'Add X to your test suite', 'Review your Y process', 'No action needed', etc.>",
  "key_findings": ["<Finding 1 in simple terms>", "<Finding 2>", "<Finding 3>"],
  "relevance_to_user": "<How this connects to their specific work. Use 'you' and 'your'. Be direct.>",
  "technical_details": "<Only the technical bits they need to know, simplified.>",
  "potential_applications": "<Concrete ways they could use this in their work.>",
  "critical_notes": "<Any warnings or limitations they should know about (or null).>",
  "follow_up_suggestions": "<Specific resources or actions if they want to dig deeper (or null).>"
}

Example of good vs bad:
BAD: "This paper presents novel methodologies for optimizing transformer architectures..."
GOOD: "Researchers found a way to make AI respond 40% faster - this could speed up your voice tests."

Do not include any other text outside of this JSON structure."""

        user_prompt = f"""User Name: {user_info.get('name', 'Researcher')}
User Title: {user_info.get('title', 'Researcher')}
User Research Goals: {user_info.get('goals', ', '.join(user_info.get('research_interests', [])))}

Article Title: {article_metadata.get('title', 'N/A')}
Article Authors: {', '.join(article_metadata.get('authors', []))}
Article Subject: {article_metadata.get('subject', 'N/A')}

Article Content (first 8000 chars):
{article_content[:8000]}"""

        response = await self._call_llm(system_prompt, user_prompt, temperature=0.3)

        try:
            data = json.loads(response)
            
            analysis_data = {
                **article_metadata,
                'summary': data.get('summary', ''),
                'importance': data.get('importance', ''),
                'recommended_action': data.get('recommended_action', 'Review when time permits'),
                'key_findings': data.get('key_findings', ['No findings available']),
                'relevance_to_user': data.get('relevance_to_user', ''),
                'technical_details': data.get('technical_details', ''),
                'potential_applications': data.get('potential_applications', ''),
                'critical_notes': data.get('critical_notes'),
                'follow_up_suggestions': data.get('follow_up_suggestions')
            }
            
            analysis_data['title'] = article_metadata.get('title', analysis_data.get('title', 'Unknown'))
            analysis_data['authors'] = article_metadata.get('authors', analysis_data.get('authors', ['Unknown']))
            analysis_data['subject'] = article_metadata.get('subject', analysis_data.get('subject', 'cs.AI'))
            analysis_data['abstract_url'] = article_metadata.get('abstract_url', analysis_data.get('abstract_url', ''))
            analysis_data['html_url'] = article_metadata.get('html_url', analysis_data.get('html_url'))
            analysis_data['pdf_url'] = article_metadata.get('pdf_url', analysis_data.get('pdf_url', ''))
            analysis_data['relevance_score'] = article_metadata.get('relevance_score', analysis_data.get('relevance_score', 0))
            analysis_data['score_reason'] = article_metadata.get('score_reason', analysis_data.get('score_reason', ''))
            
            return ArticleAnalysis(**analysis_data)
        except (json.JSONDecodeError, ValueError) as e:
            logfire.error(f"Failed to parse analysis response: {e}")
            error_data = {
                **article_metadata,
                'summary': f"Analysis failed: {type(e).__name__} - {str(e)}",
                'importance': "Unknown",
                'recommended_action': "Error during analysis",
                'key_findings': ['Analysis failed'],
                'relevance_to_user': 'Unknown',
                'technical_details': 'Not analyzed due to error',
                'potential_applications': 'Not analyzed due to error',
                'critical_notes': None,
                'follow_up_suggestions': None
            }
            return ArticleAnalysis(**error_data)
        except Exception as e:
            logfire.error(f"Error during article analysis for '{article_metadata.get('title')}': {e}", exc_info=True)
            error_data = {
                **article_metadata,
                'summary': f"Analysis failed: {type(e).__name__} - {str(e)}",
                'importance': "Unknown",
                'recommended_action': "Error during analysis",
                'key_findings': ['Analysis failed'],
                'relevance_to_user': 'Unknown',
                'technical_details': 'Not analyzed due to error',
                'potential_applications': 'Not analyzed due to error',
                'critical_notes': None,
                'follow_up_suggestions': None
            }
            return ArticleAnalysis(**error_data)



    def _determine_action(self, analysis_data: Dict[str, Any]) -> str:
        """Determine recommended action based on analysis."""
        relevance = analysis_data.get('relevance_to_user', '')
        if 'highly relevant' in relevance.lower() or 'directly applicable' in relevance.lower():
            return "Deep dive - this paper is highly relevant to your work"
        elif 'somewhat relevant' in relevance.lower():
            return "Skim the methodology section"
        else:
            return "Save for later reference"

    