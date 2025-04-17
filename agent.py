# agent.py

from __future__ import annotations
import os
import asyncio
import json
import re
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from agent_prompts import SYSTEM_PROMPT, ARTICLE_ANALYSIS_PROMPT
from agent_tools import scrape_article, analyze_article

# ========== SETUP LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("article_rank_agent")

# ========== ENV SETUP ==========
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")

# Allow model selection via environment variable
model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")
logger.info(f"Model name from environment: {model_name}")

llm = OpenAIModel(
    model_name,
    provider=OpenAIProvider(api_key=openai_api_key)
)

# ========== MODELS ==========
class RankedArticle(BaseModel):
    """Pydantic model for validating ranked article data"""
    title: str
    authors: List[str] = Field(min_items=1)
    subject: str
    score_reason: str
    relevance_score: int = Field(ge=0, le=100)
    abstract_url: str
    html_url: str
    pdf_url: str

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
article_rank_agent = Agent(
    llm,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)

# ========== UTILITY FUNCTIONS ==========
def extract_json(output: str) -> str:
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

def normalize_article_data(article: Dict[str, Any]) -> Dict[str, Any]:
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
    
    # Ensure authors is a list
    if "authors" not in normalized and "author" in normalized:
        if isinstance(normalized["author"], str):
            normalized["authors"] = [normalized["author"]]
        else:
            normalized["authors"] = normalized["author"]
    
    # Ensure relevance_score is an integer
    if "relevance_score" in normalized and isinstance(normalized["relevance_score"], str):
        try:
            normalized["relevance_score"] = int(normalized["relevance_score"])
        except ValueError:
            normalized["relevance_score"] = 0
    
    return normalized

def parse_response(response: Any) -> List[RankedArticle]:
    """
    Parse and validate the LLM response.
    
    Args:
        response: The response from the LLM (string, dict, or AgentRunResult)
        
    Returns:
        List of validated RankedArticle objects
    """
    # Handle AgentRunResult
    if hasattr(response, 'data'):
        try:
            data = json.loads(response.data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from AgentRunResult: {e}")
            logger.debug(f"Raw response data: {response.data}")
            return []
    # If response is a string, try to parse it as JSON
    elif isinstance(response, str):
        try:
            # First try to extract JSON if it's wrapped in text/markdown
            json_str = extract_json(response)
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw response: {response}")
            return []
    else:
        data = response
    
    # Handle dict with 'papers' key
    if isinstance(data, dict) and 'papers' in data:
        data = data['papers']
    
    # Ensure data is a list
    if not isinstance(data, list):
        logger.error(f"Expected a list of articles, got {type(data)}")
        return []
    
    # Normalize and validate each article with Pydantic
    try:
        normalized_articles = [normalize_article_data(article) for article in data]
        return [RankedArticle.model_validate(article) for article in normalized_articles]
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return []

def limit_articles(articles: List[Dict[str, Any]], max_articles: int = 50) -> List[Dict[str, Any]]:
    """
    Limit the number of articles to prevent token limit issues.
    
    Args:
        articles: List of article dictionaries
        max_articles: Maximum number of articles to include
        
    Returns:
        Limited list of articles
    """
    if len(articles) <= max_articles:
        return articles
    
    logger.warning(f"Limiting articles from {len(articles)} to {max_articles} to prevent token limit issues")
    return articles[:max_articles]

async def rank_articles(
    user_info: Dict[str, str],
    articles: List[Dict[str, Any]],
    top_n: int = 5,
    max_articles: int = 50
) -> List[RankedArticle]:
    """
    Rank articles based on user information.
    
    Args:
        user_info: Dictionary with user information (name, title, goals)
        articles: List of article dictionaries
        top_n: Number of top articles to return
        max_articles: Maximum number of articles to include in the prompt
        
    Returns:
        List of ranked articles
    """
    # Limit articles to prevent token limit issues
    limited_articles = limit_articles(articles, max_articles)
    if len(articles) > max_articles:
        logger.warning(f"Only the first {max_articles} articles will be considered for ranking")
    
    # Compose user message (the key step!)
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
    
    # Log a preview of the prompt for debugging
    prompt_preview = user_message[:500] + "..." if len(user_message) > 500 else user_message
    logger.debug(f"Prompt preview: {prompt_preview}")
    
    try:
        logger.info(f"Running agent with model: {model_name}")
        response = await article_rank_agent.run(user_message)
        
        # Parse and validate the response
        articles_ranked = parse_response(response)
        
        if not articles_ranked:
            logger.error("No valid articles returned from the agent")
            return []
        
        return articles_ranked
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        return []

async def analyze_ranked_articles(
    user_info: Dict[str, str],
    ranked_articles: List[RankedArticle],
    top_n: int = 5
) -> List[ArticleAnalysis]:
    """
    Analyze ranked articles for a user.

    Args:
        user_info: User information (name, title, goals)
        ranked_articles: List of ranked articles from the ranking agent
        top_n: Number of top articles to analyze

    Returns:
        List of article analyses
    """
    # Limit to top N articles
    articles_to_analyze = ranked_articles[:top_n]

    # Create user context
    user_context = UserContext(
        name=user_info["name"],
        title=user_info["title"],
        goals=user_info["goals"]
    )

    # Initialize results
    analyses = []
    failed_articles = []

    # Process each article
    for article in articles_to_analyze:
        try:
            # Scrape article content
            article_content = await scrape_article(article.html_url)

            if not article_content:
                logger.warning(f"Failed to scrape content for article: {article.title}")
                failed_articles.append(article.title)
                continue

            # Analyze article
            analysis_result = await analyze_article(
                ctx=article_rank_agent,
                article_text=article_content,
                user_context=user_context,
                article_metadata=article
            )

            # Create analysis object
            analysis = ArticleAnalysis(
                title=article.title,
                authors=article.authors,
                subject=article.subject,
                summary=analysis_result["summary"],
                importance=analysis_result["importance"],
                recommended_action=analysis_result["recommended_action"],
                abstract_url=article.abstract_url,
                html_url=article.html_url,
                pdf_url=article.pdf_url,
                relevance_score=article.relevance_score,
                score_reason=article.score_reason
            )

            analyses.append(analysis)

        except Exception as e:
            logger.error(f"Error analyzing article {article.title}: {e}")
            failed_articles.append(article.title)
            # Continue with next article

    # Log summary of failed articles
    if failed_articles:
        logger.warning(f"Failed to analyze {len(failed_articles)} articles: {', '.join(failed_articles)}")
    else:
        logger.info(f"Successfully analyzed all {len(articles_to_analyze)} articles")

    return analyses

# ========= MAIN LOGIC =========
async def main():
    """Run the article ranking agent with a sample user."""
    logger.info("Initializing article ranking agent...")

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
    articles_ranked = await rank_articles(user_info, articles, top_n=top_n)
    
    if not articles_ranked:
        logger.error("Failed to rank articles")
        return
    
    # After ranking articles, analyze them
    analyses = await analyze_ranked_articles(user_info, articles_ranked)
    
    # Print results
    print(f"\nTop {len(articles_ranked)} Relevant Papers:")
    print("=" * 80)
    for i, article in enumerate(articles_ranked, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Authors: {', '.join(article.authors)}")
        print(f"   Subject: {article.subject}")
        print(f"   Relevance Score: {article.relevance_score}/100")
        print(f"   Reasoning: {article.score_reason}")
        print(f"   URL: {article.abstract_url}")
        print(f"   HTML URL: {article.html_url}")
        print(f"   PDF URL: {article.pdf_url}")
        print("-" * 80)

    # Print analyses
    if analyses:
        print("\nArticle Analyses:")
        print("=" * 80)
        for analysis in analyses:
            print(f"\nArticle: {analysis.title}")
            print(f"Summary: {analysis.summary}")
            print(f"Importance: {analysis.importance}")
            print(f"Recommended Action: {analysis.recommended_action}")
            print("-" * 80)
    else:
        print("\nNo article analyses available. This may be due to scraping issues with the arXiv website.")
        print("You can still access the articles directly using the URLs provided above.")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())