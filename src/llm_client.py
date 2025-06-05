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
    """Schema for individual ranked article in structured output."""
    title: str = Field(description="Paper or article title")
    score: int = Field(ge=0, le=100, description="Relevance score 0-100")
    reasoning: str = Field(description="Brief explanation of relevance")
    abstract_url: str = Field(description="Article URL")
    authors: List[str] = Field(description="List of authors")
    subject: str = Field(description="Subject category or 'news'")
    html_url: Optional[str] = Field(default=None, description="HTML URL if available")
    pdf_url: Optional[str] = Field(default=None, description="PDF URL if available")
    type: Optional[str] = Field(default=None, description="Type: 'paper' or 'news'")
    source: Optional[str] = Field(default=None, description="News source name for news articles")
    published_at: Optional[str] = Field(default=None, description="Publication timestamp for news")

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
        """Rank articles based on user profile using structured outputs."""
        system_prompt = f"""You are an expert research paper analyst. Your task is to rank academic papers based on their relevance to a user's research interests and background.

Below is a list of articles in JSON format. Select the {top_n} most relevant articles based on the user profile.

You must return a structured response with an "articles" field containing exactly {top_n} selected articles. Each article must include:
- title: Paper title
- score: Relevance score 0-100 (integer)
- reasoning: Brief explanation of relevance  
- abstract_url: The paper's abstract URL
- authors: List of authors
- subject: Subject category
- html_url: HTML URL if available (optional)
- pdf_url: PDF URL (optional)

CRITICAL INSTRUCTIONS:
1. You MUST select exactly {top_n} articles (not fewer, not more)
2. Even if only 1 article seems highly relevant, find {top_n} articles with varying relevance scores
3. Return them in descending order by relevance score"""

        user_prompt = f"""User profile:
Name: {user_info.get('name', 'Researcher')}
Title: {user_info.get('title', 'Researcher')}
Research Interests: {user_info.get('goals', ', '.join(user_info.get('research_interests', [])))}

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
                # Successful structured output
                logfire.info(f"Structured output success: got {len(response.articles)} articles")
                
                # Convert to list of dicts for merging
                ranked_data = [article.model_dump() for article in response.articles]
                merged_articles = self._merge_llm_and_original_articles(ranked_data, articles)
                
                ranked_articles = []
                for item in merged_articles[:top_n]:
                    try:
                        ranked_articles.append(RankedArticle(**item))
                    except Exception as e:
                        item_str = str(item)[:200].replace('{', '{{').replace('}', '}}')
                        logfire.error(f"Failed to create RankedArticle from structured output: {str(e)}, data: {item_str}")
                        continue
                
                logfire.info(f"Successfully processed {len(ranked_articles)} articles via structured output")
                return ranked_articles
                
            elif isinstance(response, list):
                # Fallback response as list of dicts
                logfire.info(f"Fallback response: got {len(response)} articles as list")
                merged_articles = self._merge_llm_and_original_articles(response, articles)
                
                ranked_articles = []
                for item in merged_articles[:top_n]:
                    try:
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
        """Rank mixed content (papers and news) with type-aware scoring using structured outputs."""
        system_prompt = f"""You are an expert analyst who ranks both research papers and news articles based on their relevance to a user's research interests and current work.

Below is a list of content items in JSON format, which may include both academic papers and news articles. Select the {top_n} most relevant items based on the user profile.

When ranking, consider:
1. Research papers: Focus on technical relevance, methodological contributions, and potential applications
2. News articles: Focus on industry relevance, emerging trends, company/technology mentions, and timely information
3. {"Give slightly higher weight to recent news for timely insights" if weight_recency else "Weight all content equally regardless of publication date"}

You must return a structured response with an "articles" field containing exactly {top_n} selected items. Each item must include:
- title: Title
- score: Relevance score 0-100 (integer)
- reasoning: Brief explanation of relevance
- abstract_url: The item's URL
- authors: List of authors
- subject: Subject category or "news"
- html_url: HTML URL if available (optional)
- pdf_url: PDF URL for papers (optional)
- type: "paper" or "news"
- source: News source name for news articles (optional)
- published_at: Publication timestamp for news (optional)

CRITICAL INSTRUCTIONS:
1. You MUST select exactly {top_n} items (not fewer, not more)
2. Mix papers and news based on relevance - don't artificially balance types
3. Consider the user's role and interests when weighing paper vs news relevance
4. For news, focus on business/industry impact and emerging trends
5. Return them in descending order by relevance score"""

        user_prompt = f"""User profile:
Name: {user_info.get('name', 'Researcher')}
Title: {user_info.get('title', 'Researcher')}
Research Interests: {user_info.get('goals', ', '.join(user_info.get('research_interests', [])))}

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
                # Successful structured output
                logfire.info(f"Mixed content structured output success: got {len(response.articles)} items")
                
                # Convert to list of dicts for merging
                ranked_data = [article.model_dump() for article in response.articles]
                merged_content = self._merge_llm_and_original_articles(ranked_data, content)
                
                ranked_articles = []
                for item in merged_content[:top_n]:
                    try:
                        # Ensure proper type field
                        if 'type' not in item:
                            item['type'] = 'news' if item.get('subject') == 'news' else 'paper'
                        
                        ranked_articles.append(RankedArticle(**item))
                    except Exception as e:
                        item_str = str(item)[:200].replace('{', '{{').replace('}', '}}')
                        logfire.error(f"Failed to create RankedArticle from mixed structured output: {str(e)}, data: {item_str}")
                        continue

                logfire.info(f"Successfully ranked {len(ranked_articles)} mixed content items via structured output")
                return ranked_articles
                
            elif isinstance(response, list):
                # Fallback response as list of dicts
                logfire.info(f"Mixed content fallback response: got {len(response)} items as list")
                merged_content = self._merge_llm_and_original_articles(response, content)
                
                ranked_articles = []
                for item in merged_content[:top_n]:
                    try:
                        # Ensure proper type field
                        if 'type' not in item:
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

    def _merge_llm_and_original_articles(self, llm_results: List[Dict], original_articles: List[Dict]) -> List[Dict]:
        """Merge LLM ranking output with original article data."""
        logfire.info(f"Merging {len(llm_results)} LLM results with {len(original_articles)} original articles")
        
        original_articles_map = {str(orig.get('abstract_url')): orig for orig in original_articles if orig.get('abstract_url')}
        
        filled_ranked_articles = []
        processed_urls = set()
        
        for llm_article in llm_results:
            abstract_url = str(llm_article.get('abstract_url', ''))
            
            if abstract_url in processed_urls:
                logfire.warn(f"Skipping duplicate article from LLM output: {llm_article.get('title')}")
                continue
            processed_urls.add(abstract_url)
            
            original_data = original_articles_map.get(abstract_url)
            
            if original_data:
                merged_data = original_data.copy()
                
                merged_data.update({
                    'relevance_score': llm_article.get('score', llm_article.get('relevance_score', 0)),
                    'score_reason': llm_article.get('reasoning', llm_article.get('score_reason', '')),
                })
                
                merged_data.setdefault('title', llm_article.get('title', 'Unknown'))
                merged_data.setdefault('authors', llm_article.get('authors', ['Unknown']))
                merged_data.setdefault('subject', llm_article.get('subject', 'cs.AI'))
                
                filled_ranked_articles.append(merged_data)
            else:
                logfire.warn(f"Could not find original data for ranked article: '{llm_article.get('title')}' ({abstract_url}). Using LLM output directly.")
                
                llm_article.setdefault('relevance_score', llm_article.get('score', 0))
                llm_article.setdefault('score_reason', llm_article.get('reasoning', ''))
                llm_article.setdefault('authors', ['Unknown'])
                llm_article.setdefault('subject', 'cs.AI')
                
                filled_ranked_articles.append(llm_article)
        
        logfire.info(f"Successfully merged {len(filled_ranked_articles)} articles")
        return filled_ranked_articles

    def _determine_action(self, analysis_data: Dict[str, Any]) -> str:
        """Determine recommended action based on analysis."""
        relevance = analysis_data.get('relevance_to_user', '')
        if 'highly relevant' in relevance.lower() or 'directly applicable' in relevance.lower():
            return "Deep dive - this paper is highly relevant to your work"
        elif 'somewhat relevant' in relevance.lower():
            return "Skim the methodology section"
        else:
            return "Save for later reference"

    