# agent_fixed.py

from __future__ import annotations
import os
import asyncio
import json
import re
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from src.agent_prompts import SYSTEM_PROMPT, ARTICLE_ANALYSIS_PROMPT
from src.agent_tools import scrape_article, analyze_article

from config.settings import AgentSettings
from src.state import AgentState

# Load environment variables from config directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", ".env"))

# ========== SETUP LOGGING ==========
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("article_rank_agent")

# Also set crawl4ai logger to DEBUG
logging.getLogger("crawl4ai").setLevel(logging.DEBUG)

# Add OpenAI client logging
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

# ========== MODELS ==========
class RankedArticle(BaseModel):
    """Pydantic model for validating ranked article data"""
    title: str
    authors: List[str] = Field(min_items=1)
    subject: str
    score_reason: str
    relevance_score: int = Field(ge=0, le=100)
    abstract_url: Optional[str] = Field(default="")
    html_url: Optional[str] = Field(default="")
    pdf_url: Optional[str] = Field(default="")

    @field_validator('abstract_url', 'html_url', 'pdf_url')
    @classmethod
    def validate_urls(cls, v: Optional[str]) -> str:
        """Validate URLs, converting None to empty string"""
        if v is None:
            return ""
        return v

class ArticleAnalysis(BaseModel):
    """Pydantic model for article analysis results"""
    title: str
    authors: List[str] = Field(min_items=1)
    subject: str
    summary: str
    importance: str
    recommended_action: str
    abstract_url: str
    html_url: str
    pdf_url: str
    relevance_score: int = Field(ge=0, le=100)
    score_reason: str

class BatchAnalysis(BaseModel):
    """Pydantic model for batch analysis results"""
    analyses: List[ArticleAnalysis]

class UserContext(BaseModel):
    """Pydantic model for user context"""
    name: str
    title: str
    goals: str

# ========== AGENT ==========
class ArticleRankAgent:
    """Agent responsible for ranking and analyzing articles"""
    
    def __init__(self, settings: AgentSettings, state: AgentState):
        self.settings = settings
        self.state = state
        
        # Log API key info (masked)
        api_key = settings.openai_api_key
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if api_key else "None"
        logger.debug(f"Initializing OpenAI model with API key: {masked_key}")
        logger.debug(f"Using model: {settings.openai_model}")
        
        # Initialize OpenAI model
        try:
            self.llm = OpenAIModel(
                settings.openai_model,
                provider=OpenAIProvider(api_key=settings.openai_api_key)
            )
            logger.info("Successfully initialized OpenAI model")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI model: {str(e)}")
            raise
        
        # Initialize agent
        try:
            self.agent = Agent(
                self.llm,
                system_prompt=SYSTEM_PROMPT,
                retries=1
            )
            logger.info("Successfully initialized Agent")
        except Exception as e:
            logger.error(f"Failed to initialize Agent: {str(e)}")
            raise
    
    async def rank_articles(
        self,
        user_info: Dict[str, str],
        articles: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[RankedArticle]:
        """
        Rank articles based on user information.
        
        Args:
            user_info: Dictionary with user information (name, title, goals)
            articles: List of article dictionaries
            top_n: Number of top articles to return
            
        Returns:
            List of ranked articles
        """
        # Store original articles in state for later use
        self.state.update_last_processed({"original_articles": articles})
        
        # Limit articles to prevent token limit issues
        limited_articles = articles[:self.settings.max_articles]
        if len(articles) > self.settings.max_articles:
            logger.warning(f"Only the first {self.settings.max_articles} articles will be considered for ranking")
        
        # Compose user message
        user_message = (
            f"Here is some information about me:\n"
            f"Name: {user_info['name']}\n"
            f"Title: {user_info['title']}\n"
            f"Research interests: {user_info['goals']}\n\n"
            f"Below is a list of recent arXiv articles. For each article, you have its title, authors, subject, and the abstract. "
            f"Please pick the {top_n} most relevant articles for my research interests, and for each, output:\n"
            "- title\n- authors\n- subject\n- a reasoning why you selected it\n- a relevance score from 0-100\n\n"
            f"Here are the articles (as JSON):\n\n{json.dumps(limited_articles, indent=2)}\n\n"
            "Format your response ONLY as valid JSON. Do not include markdown or any extra commentary."
        )
        
        try:
            logger.info(f"Running agent with model: {self.settings.openai_model}")
            logger.debug("Making API call to OpenAI...")
            
            # Log the request details (without sensitive data)
            logger.debug(f"Request details:")
            logger.debug(f"- Model: {self.settings.openai_model}")
            logger.debug(f"- System prompt length: {len(SYSTEM_PROMPT)} chars")
            logger.debug(f"- User message length: {len(user_message)} chars")
            logger.debug(f"- Number of articles: {len(limited_articles)}")
            
            response = await self.agent.run(user_message)
            logger.info("Successfully received response from OpenAI")
            
            # Parse and validate the response
            articles_ranked = self._parse_response(response)
            logger.info(f"Successfully parsed {len(articles_ranked)} ranked articles")
            
            return articles_ranked
            
        except Exception as e:
            logger.error(f"Error ranking articles: {str(e)}", exc_info=True)
            if hasattr(e, 'response'):
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return []
    
    async def analyze_ranked_articles(
        self,
        user_info: Dict[str, str],
        ranked_articles: List[RankedArticle],
        top_n: int = 5
    ) -> List[ArticleAnalysis]:
        """
        Analyze ranked articles in detail.
        
        Args:
            user_info: Dictionary with user information
            ranked_articles: List of ranked articles
            top_n: Number of top articles to analyze
            
        Returns:
            List of article analyses
        """
        try:
            # Limit to top N articles
            articles_to_analyze = ranked_articles[:top_n]
            
            # Create user context object
            user_ctx = UserContext(
                name=user_info["name"],
                title=user_info["title"],
                goals=user_info["goals"]
            )
            
            # Get original articles from state
            state_data = self.state.get_last_processed()
            original_articles = state_data.get("original_articles", [])
            
            if not original_articles:
                logger.error("No original articles found in state")
                return []
            
            # Analyze each article
            analyses = []
            for article in articles_to_analyze:
                try:
                    # Create article metadata
                    metadata = {
                        "title": article.title,
                        "authors": article.authors,
                        "subject": article.subject,
                        "relevance_score": article.relevance_score,
                        "score_reason": article.score_reason
                    }
                    
                    # Get article content from original articles
                    article_data = next(
                        (a for a in original_articles if a["title"] == article.title),
                        None
                    )
                    
                    if not article_data:
                        logger.error(f"Could not find article data for: {article.title}")
                        continue
                    
                    # Get article content - either from body field or by scraping
                    article_content = ""
                    if "body" in article_data and article_data["body"]:
                        article_content = article_data["body"]
                    else:
                        # Scrape content from HTML URL
                        logger.info(f"Scraping content for article: {article.title}")
                        article_content = await scrape_article(article.html_url)
                        
                        if not article_content:
                            logger.error(f"Failed to scrape content for article: {article.title}")
                            continue
                    
                    # Analyze content
                    analysis_result = await analyze_article(
                        self.agent,  # Pass the agent as context
                        article_content,  # The article text
                        user_ctx,  # User context
                        metadata  # Article metadata
                    )
                    
                    # Create analysis object
                    analysis = ArticleAnalysis(
                        title=article.title,
                        authors=article.authors,
                        subject=article.subject,
                        summary=analysis_result.get("summary", ""),
                        importance=analysis_result.get("importance", ""),
                        recommended_action=analysis_result.get("recommended_action", ""),
                        abstract_url=article.abstract_url,
                        html_url=article.html_url,
                        pdf_url=article.pdf_url,
                        relevance_score=article.relevance_score,
                        score_reason=article.score_reason
                    )
                    analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Error analyzing article {article.title}: {e}")
                    continue
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error analyzing articles: {e}")
            return []
    
    def _parse_response(self, response: Any) -> List[RankedArticle]:
        """
        Parse and validate the LLM response.
        
        Args:
            response: The response from the LLM
            
        Returns:
            List of validated RankedArticle objects
        """
        try:
            # Handle AgentRunResult
            if hasattr(response, 'data'):
                data = json.loads(response.data)
            # If response is a string, try to parse it as JSON
            elif isinstance(response, str):
                # First try to extract JSON if it's wrapped in text/markdown
                json_str = self._extract_json(response)
                data = json.loads(json_str)
            else:
                data = response
            
            # Handle dict with 'papers' key
            if isinstance(data, dict) and 'papers' in data:
                data = data['papers']
            
            # Ensure data is a list
            if not isinstance(data, list):
                logger.error(f"Expected a list of articles, got {type(data)}")
                return []
            
            # Log the raw data for debugging
            logger.debug(f"Raw response data: {json.dumps(data, indent=2)}")
            
            # Normalize and validate each article with Pydantic
            normalized_articles = []
            for article in data:
                try:
                    normalized = self._normalize_article_data(article)
                    normalized_articles.append(normalized)
                except Exception as e:
                    logger.error(f"Error normalizing article: {e}")
                    continue
            
            # Validate all articles
            validated_articles = []
            for article in normalized_articles:
                try:
                    validated = RankedArticle.model_validate(article)
                    validated_articles.append(validated)
                except ValidationError as e:
                    logger.error(f"Validation error for article {article.get('title', 'Unknown')}: {e}")
                    continue
            
            return validated_articles
            
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return []
    
    def _extract_json(self, output: str) -> str:
        """
        Extract JSON from LLM output, handling various formats and wrapping.
        
        Args:
            output: The raw output from the LLM
            
        Returns:
            Extracted JSON string
        """
        # Try to find a JSON array
        json_match = re.search(r'\[.*\]', output, flags=re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # Try to find a dict with 'papers' key
        dict_match = re.search(r'\{.*"papers"\s*:\s*\[.*\]\s*\}', output, flags=re.DOTALL)
        if dict_match:
            return dict_match.group(0)
        
        # If no match found, return the original output
        return output
    
    def _normalize_article_data(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize article data to handle synonym keys and ensure consistent structure.
        
        Args:
            article: Raw article dictionary
            
        Returns:
            Normalized article dictionary
        """
        normalized = article.copy()
        
        # Handle subject/subjects synonym
        if "subjects" in normalized and "subject" not in normalized:
            normalized["subject"] = normalized["subjects"]
        elif "subject" not in normalized:
            normalized["subject"] = "Not specified"
        
        # Ensure authors is a list
        if "authors" not in normalized and "author" in normalized:
            if isinstance(normalized["author"], str):
                normalized["authors"] = [normalized["author"]]
            elif isinstance(normalized["author"], list):
                normalized["authors"] = normalized["author"]
            else:
                normalized["authors"] = ["Unknown"]
        elif "authors" not in normalized:
            normalized["authors"] = ["Unknown"]
        elif not isinstance(normalized["authors"], list):
            try:
                if isinstance(normalized["authors"], str):
                    normalized["authors"] = [normalized["authors"]]
                else:
                    normalized["authors"] = list(normalized["authors"])
            except:
                normalized["authors"] = ["Unknown"]
        
        # Ensure at least one author
        if not normalized["authors"]:
            normalized["authors"] = ["Unknown"]
        
        # Ensure relevance_score is an integer
        if "relevance_score" in normalized:
            try:
                if isinstance(normalized["relevance_score"], str):
                    normalized["relevance_score"] = int(float(normalized["relevance_score"]))
                elif isinstance(normalized["relevance_score"], float):
                    normalized["relevance_score"] = int(normalized["relevance_score"])
            except (ValueError, TypeError):
                normalized["relevance_score"] = 0
        else:
            normalized["relevance_score"] = 0
        
        # Clamp relevance score to valid range
        normalized["relevance_score"] = max(0, min(100, normalized["relevance_score"]))
        
        # Ensure title is a string
        if not normalized.get("title"):
            normalized["title"] = "Untitled Article"
        elif not isinstance(normalized["title"], str):
            try:
                normalized["title"] = str(normalized["title"])
            except:
                normalized["title"] = "Untitled Article"
        
        # Try to extract arxiv_id and construct URLs
        arxiv_id = None
        
        # Function to safely extract arxiv_id from URL
        def extract_arxiv_id(url: Any) -> Optional[str]:
            if not isinstance(url, str):
                return None
            patterns = [
                r'/abs/([^/\s]+)',
                r'/pdf/([^/\s]+?)(?:\.pdf)?$',
                r'/html/([^/\s]+)',
                r'arxiv.org[:/]([^/\s]+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)
            return None
        
        # Try to get arxiv_id from various sources
        sources = [
            ("arxiv_id", normalized.get("arxiv_id")),
            ("abstract_url", normalized.get("abstract_url")),
            ("html_url", normalized.get("html_url")),
            ("pdf_url", normalized.get("pdf_url"))
        ]
        
        for source_name, source_value in sources:
            if source_value:
                if source_name == "arxiv_id":
                    arxiv_id = str(source_value)
                    break
                extracted = extract_arxiv_id(source_value)
                if extracted:
                    arxiv_id = extracted
                    break
        
        # Clean arxiv_id if found
        if arxiv_id:
            # Remove any 'arxiv:' prefix
            arxiv_id = re.sub(r'^arxiv:', '', arxiv_id, flags=re.IGNORECASE)
            # Remove any file extension
            arxiv_id = re.sub(r'\.pdf$', '', arxiv_id, flags=re.IGNORECASE)
            
            # Construct URLs
            normalized["abstract_url"] = f"https://arxiv.org/abs/{arxiv_id}"
            normalized["html_url"] = f"https://arxiv.org/html/{arxiv_id}"
            normalized["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        else:
            # If no valid arxiv_id found, set empty URLs
            normalized["abstract_url"] = ""
            normalized["html_url"] = ""
            normalized["pdf_url"] = ""
        
        # Ensure all required fields are present with valid values
        required_fields = {
            "title": "Untitled Article",
            "authors": ["Unknown"],
            "subject": "Not specified",
            "score_reason": "No reason provided",
            "relevance_score": 0,
            "abstract_url": "",
            "html_url": "",
            "pdf_url": ""
        }
        
        for field, default_value in required_fields.items():
            if field not in normalized or normalized[field] is None:
                normalized[field] = default_value
                logger.warning(f"Missing required field '{field}' in article data, using default: {default_value}")
        
        return normalized

# ========= MAIN LOGIC =========
async def main():
    """Run the article ranking agent with a sample user."""
    logger.info("Initializing article ranking agent...")

    # Initialize settings and state
    settings = AgentSettings()
    state = AgentState()
    
    # Initialize agent
    agent = ArticleRankAgent(settings, state)

    # User info (hardcoded for demonstration, could load from user input)
    user_info = {
        "name": "Dr. Jane Smith",
        "title": "Computer Science Professor",
        "goals": "I'm researching new approaches to natural language processing and looking for papers on transformer architectures and their applications."
    }

    # Load articles from file or use demo data
    arxiv_file = os.getenv("ARXIV_FILE", "arxiv_cs_submissions_2025-04-01.json")
    if os.path.exists(arxiv_file):
        with open(arxiv_file, "r") as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {arxiv_file}.")
    else:
        logger.warning(f"Warning: {arxiv_file} not found. Using sample data.")
        articles = [
            {
                "arxiv_id": "2403.12345",
                "title": "Advanced Transformer Architectures for NLP",
                "subjects": "cs.CL",
                "authors": ["John Doe", "Jane Smith"],
                "abstract_url": "https://arxiv.org/abs/2403.12345",
                "pdf_url": "https://arxiv.org/pdf/2403.12345.pdf",
                "html_url": "https://arxiv.org/html/2403.12345"
            },
            {
                "arxiv_id": "2403.12346",
                "title": "Reinforcement Learning for Robotics",
                "subjects": "cs.RO",
                "authors": ["Alice Johnson", "Bob Wilson"],
                "abstract_url": "https://arxiv.org/abs/2403.12346",
                "pdf_url": "https://arxiv.org/pdf/2403.12346.pdf",
                "html_url": "https://arxiv.org/html/2403.12346"
            },
            {
                "arxiv_id": "2403.12347",
                "title": "Quantum Computing Algorithms",
                "subjects": "quant-ph",
                "authors": ["Charlie Brown", "Diana Ross"],
                "abstract_url": "https://arxiv.org/abs/2403.12347",
                "pdf_url": "https://arxiv.org/pdf/2403.12347.pdf",
                "html_url": "https://arxiv.org/html/2403.12347"
            }
        ]
    
    # Get number of top articles from environment or use default
    top_n = int(os.getenv("TOP_N_ARTICLES", "5"))
    
    # Rank articles
    articles_ranked = await agent.rank_articles(user_info, articles, top_n=top_n)
    
    if not articles_ranked:
        logger.error("Failed to rank articles")
        return
    
    # After ranking articles, analyze them
    analyses = await agent.analyze_ranked_articles(user_info, articles_ranked)
    
    # Print results
    print("\n" + "=" * 80)
    print(f"TOP {len(articles_ranked)} RELEVANT PAPERS FOR {user_info['name']}")
    print("=" * 80)
    
    for i, article in enumerate(articles_ranked, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Authors: {', '.join(article.authors)}")
        print(f"   Subject: {article.subject}")
        print(f"   Relevance Score: {article.relevance_score}/100")
        print(f"   Reasoning: {article.score_reason}")
        print(f"   URLs:")
        print(f"     - Abstract: {article.abstract_url}")
        print(f"     - HTML: {article.html_url}")
        print(f"     - PDF: {article.pdf_url}")
        print("-" * 80)

    # Print analyses
    if analyses:
        print("\n" + "=" * 80)
        print("ARTICLE ANALYSES")
        print("=" * 80)
        
        for i, analysis in enumerate(analyses, 1):
            print(f"\n{i}. {analysis.title}")
            print(f"   Authors: {', '.join(analysis.authors)}")
            print(f"   Subject: {analysis.subject}")
            print(f"   Relevance Score: {analysis.relevance_score}/100")
            print("\n   SUMMARY:")
            print(f"   {analysis.summary}")
            print("\n   IMPORTANCE:")
            print(f"   {analysis.importance}")
            print("\n   RECOMMENDED ACTION:")
            print(f"   {analysis.recommended_action}")
            print("-" * 80)
    else:
        print("\n" + "=" * 80)
        print("NO ARTICLE ANALYSES AVAILABLE")
        print("=" * 80)
        print("This may be due to scraping issues with the arXiv website.")
        print("You can still access the articles directly using the URLs provided above.")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 