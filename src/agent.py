# agent_fixed.py

from __future__ import annotations as _annotations

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, List, Dict
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from .agent_prompts import SYSTEM_PROMPT, ARTICLE_ANALYSIS_PROMPT
from .agent_tools import scrape_article, analyze_article
from crawl4ai import AsyncWebCrawler # Import the crawler

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    # Fallback to project root if not found in config
    load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("pydantic_arxiv_agent")

# ==============================
#         DATA MODELS
# ==============================

class RankedArticle(BaseModel):
    """Pydantic model for a single ranked article."""
    title: str
    authors: List[str] = Field(min_items=1)
    subject: str
    score_reason: str
    relevance_score: int = Field(ge=0, le=100)
    abstract_url: str
    html_url: str
    pdf_url: str

    @field_validator("authors", mode="before")
    @classmethod
    def ensure_authors_list(cls, v):
        """Ensure authors is always a List[str], even if malformed input."""
        if isinstance(v, list):
            return v or ["Unknown"]
        if isinstance(v, str):
            return [v]
        return ["Unknown"]

    @field_validator("abstract_url", "html_url", "pdf_url", mode="before")
    @classmethod
    def normalize_url(cls, v):
        return v or ""

    @field_validator("title", mode="before")
    @classmethod
    def ensure_title(cls, v):
        return v or "Untitled Article"

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, values):
        # Handle synonyms and minimal key consistency
        if "subjects" in values and "subject" not in values:
            values["subject"] = values.pop("subjects")
        if "author" in values and "authors" not in values:
            # Try to wrap single author
            authors = values.pop("author")
            if isinstance(authors, str):
                values["authors"] = [authors]
            else:
                values["authors"] = authors or ["Unknown"]
        if "score_reason" not in values:
            values["score_reason"] = "No reason provided"
        if "relevance_score" not in values:
            values["relevance_score"] = 0
        # Clamp relevance_score if string/float
        score = values.get("relevance_score", 0)
        if isinstance(score, str) or isinstance(score, float):
            try:
                score = int(float(score))
            except Exception:
                score = 0
        values["relevance_score"] = max(0, min(100, int(score)))
        # URLs from arXiv ID if missing
        arxiv_id = None
        for k in ("arxiv_id", "abstract_url", "html_url", "pdf_url"):
            if k in values and values[k]:
                arxiv_id = _extract_arxiv_id(values[k])
                if arxiv_id:
                    break
        if arxiv_id:
            values["abstract_url"] = f"https://arxiv.org/abs/{arxiv_id}"
            values["html_url"] = f"https://arxiv.org/html/{arxiv_id}"
            values["pdf_url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        else:
            for f in ("abstract_url", "html_url", "pdf_url"):
                values[f] = values.get(f) or ""
        return values

def _extract_arxiv_id(url: Any) -> str | None:
    import re
    if not isinstance(url, str):
        return None
    for pattern in [
        r'/abs/([^/?&#\s]+)',
        r'/pdf/([^/?&#\s]+?)(?:\.pdf)?',
        r'/html/([^/?&#\s]+)',
        r'arxiv.org[:/]([^/?&#\s]+)'
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None

class ArticleAnalysis(BaseModel):
    """Analysis result for a single article."""
    title: str
    authors: List[str]
    subject: str
    summary: str
    importance: str
    recommended_action: str
    abstract_url: str
    html_url: str
    pdf_url: str
    relevance_score: int = Field(ge=0, le=100)
    score_reason: str

class UserContext(BaseModel):
    """User/researcher profile."""
    name: str
    title: str
    goals: str

# ============================
#      DEPS + AGENT SETUP
# ============================

@dataclass
class Deps:
    agent: Agent      # For prompt-based LLM injection in tools (if needed)

# --- Pydantic AI Agent Definition ---

arxiv_agent = Agent(
    model=OpenAIModel(
        os.getenv("OPENAI_MODEL", "gpt-4o"),
        provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    ),
    system_prompt=SYSTEM_PROMPT,
    deps_type=Deps,
    retries=2,
    response_model=List[RankedArticle],   # Guides LLM to strictly follow schema! (Best practice)
)

# Note: Tools are imported and registered in agent_tools.py using @arxiv_agent.tool

# ============================
#     MAIN AGENT LOGIC
# ============================

async def rank_articles(
    user_info: dict,
    articles: list[dict],
    top_n: int = 5
) -> List[RankedArticle]:
    """
    Runs the LLM to rank articles for the given user profile.

    Args:
        user_info: {"name": ..., "title": ..., "goals": ...}
        articles: List of article dicts (raw)
        top_n: How many top articles to select

    Returns:
        List of validated RankedArticle objects
    """
    user = UserContext(**user_info)
    input_articles = articles[:20]  # Use a sane LLM upper bound or fetch from env/config

    user_prompt = (
        f"User profile:\n"
        f"Name: {user.name}\n"
        f"Title: {user.title}\n"
        f"Research Interests: {user.goals}\n\n"
        f"Now, from the articles JSON below, select the {top_n} most relevant. Output only as valid, pure JSON (see schema)."
        f"\n\nArticles:\n{input_articles}"
    )

    try:
        resp = await arxiv_agent.run(user_prompt)
        # With response_model=List[RankedArticle], output is auto-validated
        if not hasattr(resp, 'output') or not isinstance(resp.output, str):
            logger.error(f"Agent response did not contain the expected string output. Response: {resp}")
            raise TypeError(f"Agent response output is not a string: {type(resp.output if hasattr(resp, 'output') else resp)}")
        else:
            json_string = resp.output
            logger.debug(f"Received JSON string from agent: {json_string[:200]}...")
            
            # Manually parse the JSON string and validate with Pydantic model
            try:
                data = json.loads(json_string)
                if not isinstance(data, list):
                    raise TypeError(f"Parsed JSON is not a list: {type(data)}")
                
                validated_articles = []
                for item in data:
                    try:
                        validated_article = RankedArticle.model_validate(item)
                        validated_articles.append(validated_article)
                    except ValidationError as val_err:
                        logger.warning(f"Validation failed for article item: {item}. Error: {val_err}")
                        # Decide whether to skip or handle partially valid data
                        continue # Skip invalid items for now
                
                ranked: List[RankedArticle] = validated_articles
                
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON string from agent response: {json_err}")
                logger.debug(f"Invalid JSON string: {json_string}")
                raise
            except TypeError as type_err:
                 logger.error(f"Error processing parsed JSON: {type_err}")
                 raise
            
        logger.info(f"Returned {len(ranked)} ranked articles.")
        return ranked
    except ValidationError as e:
        logger.error(f"LLM output failed schema validation:\n{e}")
        # Optionally extract partial data here if needed
        raise
    except Exception as e:
        logger.error(f"Error ranking articles: {e}")
        raise

async def analyze_articles(
    user_info: dict,
    articles: List[RankedArticle],
    top_n: int = 5
) -> List[ArticleAnalysis]:
    """
    Get LLM-generated analyses for the most relevant articles.

    Args:
        user_info: Dict of user profile details
        articles: Output of rank_articles
        top_n: Number of articles to analyze

    Returns:
        List of ArticleAnalysis objects
    """
    # Prepare dependencies for tool usage
    user = UserContext(**user_info)
    analyses = []
    
    # Create crawler instance once, outside the loop
    async with AsyncWebCrawler(verbose=False) as crawler:
        logger.info(f"Analyzing top {min(top_n, len(articles))} articles...")
        for art in articles[:top_n]:
            try:
                article_metadata = art.model_dump()
                
                # Pass the crawler instance to scrape_article
                article_content = await scrape_article(crawler, art.html_url)
                
                if not article_content:
                    logger.warning(f"Skipping analysis for '{art.title}' (scraping failed or no content)")
                    continue
                
                # NOTE: analyze_article currently returns dummy data
                analysis_dict = await analyze_article(
                    arxiv_agent, # Pass the agent instance
                    article_content=article_content,
                    user_context=user,
                    article_metadata=article_metadata
                )
                analyses.append(ArticleAnalysis(
                    **article_metadata,
                    summary=analysis_dict.get("summary", ""),
                    importance=analysis_dict.get("importance", ""),
                    recommended_action=analysis_dict.get("recommended_action", "")
                ))
            except Exception as e:
                # Log the exception details for the specific article
                logger.error(f"Failed to analyze article '{art.title}': {e}", exc_info=True)
                continue # Continue to the next article
                
    logger.info(f"Completed analysis for {len(analyses)} articles.")
    return analyses

# ============================
#       HTML GENERATION
# ============================

def generate_html_email(
    user_info: Dict,
    ranked_articles: List[RankedArticle],
    analyses: List[ArticleAnalysis]
) -> str:
    """Formats the ranked articles and analyses into an HTML string for email."""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<title>ArXiv Digest for {user_info.get('name', 'Researcher')}</title>
<style>
  body {{ font-family: sans-serif; line-height: 1.6; }}
  h1, h2, h3 {{ color: #333; }}
  ul {{ list-style-type: none; padding: 0; }}
  li {{ margin-bottom: 20px; border: 1px solid #eee; padding: 15px; border-radius: 5px; }}
  .score {{ font-weight: bold; }}
  .links a {{ margin-right: 10px; text-decoration: none; color: #007bff; }}
  .analysis {{ margin-top: 10px; background-color: #f9f9f9; padding: 10px; border-radius: 4px; }}
</style>
</head>
<body>

<h1>ArXiv Digest for {user_info.get('name', 'Researcher')}</h1>
<p>Based on your interests: {user_info.get('goals', 'N/A')}</p>

<h2>Top Ranked Articles</h2>
<ul>
"""
    
    # Create a mapping from abstract_url to analysis for easy lookup
    analysis_map = {a.abstract_url: a for a in analyses}

    for rank_art in ranked_articles:
        html_content += f"""
  <li>
    <h3>{rank_art.title}</h3>
    <p>Authors: {', '.join(rank_art.authors)}</p>
    <p>Subject: {rank_art.subject}</p>
    <p><span class="score">Relevance Score: {rank_art.relevance_score}/100</span> - {rank_art.score_reason}</p>
    <p class="links">
      <a href="{rank_art.abstract_url}" target="_blank">Abstract</a> | 
      <a href="{rank_art.html_url}" target="_blank">HTML</a> | 
      <a href="{rank_art.pdf_url}" target="_blank">PDF</a>
    </p>
"""
        # Add analysis if available
        analysis = analysis_map.get(rank_art.abstract_url)
        if analysis:
            html_content += f"""
    <div class="analysis">
      <h4>Analysis:</h4>
      <p><strong>Summary:</strong> {analysis.summary}</p>
      <p><strong>Importance:</strong> {analysis.importance}</p>
      <p><strong>Recommended Action:</strong> {analysis.recommended_action}</p>
    </div>
"""
        html_content += "  </li>\n"

    html_content += """
</ul>

</body>
</html>
"""
    return html_content

# ============================
#          MAIN
# ============================

async def main():
    """Standalone test: rank and analyze articles with demo user/article set."""
    sample_user = {
        "name": "Dr. Jane Smith",
        "title": "Professor",
        "goals": "Transformer architectures and NLP applications"
    }
    
    # Load articles from file or use demo data
    default_arxiv_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'arxiv_cs_submissions_2025-04-01.json')
    arxiv_file_path = os.getenv("ARXIV_FILE", default_arxiv_file)
    
    articles_to_process = []
    if os.path.exists(arxiv_file_path):
        try:
            with open(arxiv_file_path, "r") as f:
                articles_to_process = json.load(f)
            logger.info(f"Loaded {len(articles_to_process)} articles from {arxiv_file_path}.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {arxiv_file_path}: {e}")
            logger.warning("Falling back to sample data due to JSON error.")
        except Exception as e:
             logger.error(f"Error reading file {arxiv_file_path}: {e}")
             logger.warning("Falling back to sample data due to file reading error.")
    else:
        logger.warning(f"Warning: {arxiv_file_path} not found. Using sample data.")

    # Use sample data as fallback if file loading failed or file not found
    if not articles_to_process:
         articles_to_process = [
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
            }
        ]

    # Get number of top articles from environment or use default
    top_n = int(os.getenv("TOP_N_ARTICLES", "5"))

    # Use the loaded (or sample) articles
    ranked = await rank_articles(sample_user, articles_to_process, top_n=top_n)
    analyses = await analyze_articles(sample_user, ranked, top_n=top_n)
    
    # Generate HTML content
    html_output = generate_html_email(sample_user, ranked, analyses)
    
    # Define output path in the project root
    output_file_path = os.path.join(os.path.dirname(__file__), '..', 'email_output.html')
    
    # Save the HTML to a file
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(html_output)
        logger.info(f"Successfully generated HTML email report at: {output_file_path}")
    except IOError as e:
        logger.error(f"Failed to write HTML email report to {output_file_path}: {e}")

    # Keep original print statements for console feedback during testing if needed
    # print("\n--- Ranked Articles ---")
    # for art in ranked:
    #     print(f"{art.title} (Score: {art.relevance_score}) – {art.score_reason}")
    # print("\n--- Article Analyses ---")    
    # for a in analyses:
    #     print(f"\n{a.title}\nSummary: {a.summary}\nImportance: {a.importance}\nAction: {a.recommended_action}")

if __name__ == "__main__":
    asyncio.run(main()) 