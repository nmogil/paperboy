# agent_fixed.py

from __future__ import annotations as _annotations

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, List, Dict
import json

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator, TypeAdapter
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import models from the new central location
from .models import RankedArticle, ArticleAnalysis, UserContext
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
#         DATA MODELS (Removed - Now in src/models.py)
# ==============================
# Definitions for RankedArticle, _extract_arxiv_id, ArticleAnalysis, UserContext
# are removed from here.

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
    response_model=List[RankedArticle],   # Uses imported RankedArticle
)

# Note: Tools are imported and registered in agent_tools.py using @arxiv_agent.tool

# ============================
#     MAIN AGENT LOGIC
# ============================

async def rank_articles(
    user_info: UserContext,
    articles: List[Dict[str, Any]],
    top_n: int = 5
) -> List[RankedArticle]:
    """
    Runs the LLM to rank articles for the given user profile.
    Includes post-processing to ensure all available fields are populated.

    Args:
        user_info: UserContext object containing user profile.
        articles: List of raw article dictionaries.
        top_n: How many top articles to select

    Returns:
        List of validated RankedArticle objects with fields filled from original data.
    """
    input_articles = articles[:20] # Limit input size for LLM context

    # --- 1. Improved Prompt --- 
    # Explicitly instruct LLM to include all fields from input if available
    user_prompt = (
        f"User profile:\n"
        f"Name: {user_info.name}\n"
        f"Title: {user_info.title}\n"
        f"Research Interests: {user_info.goals}\n\n"
        f"Below is a list of articles in JSON format. Select the {top_n} most relevant articles based on the user profile."
        f"Your response MUST be a valid JSON list containing ONLY the selected articles, strictly adhering to the RankedArticle schema provided."
        f"IMPORTANT: For each selected article, you MUST include **all** fields specified in the RankedArticle schema if they are present in the original input data for that article. "
        f"Do NOT omit fields like 'html_url', 'pdf_url', 'subject', 'authors', etc., if they exist in the input entry for the article. Copy the exact values."
        f"Schema Reference (RankedArticle): title (str), authors (List[str]), subject (str), score_reason (str), relevance_score (int 0-100), abstract_url (str URL), html_url (str URL, optional), pdf_url (str URL)"
        f"\n\nArticles:\n{json.dumps(input_articles, indent=2)}" # Pass as JSON string for clarity
    )

    try:
        # --- 2. LLM Call --- 
        res = await arxiv_agent.run(
            user_prompt,
            response_model=List[RankedArticle],
        )

        # --- 3. Initial Parsing & Validation (Handled by Pydantic AI) ---
        if isinstance(res.output, list) and all(isinstance(a, RankedArticle) for a in res.output):
            ranked_articles_from_llm = res.output
        else:
            # Fallback parsing (as before)
            logger.warning(f"Agent output was not List[RankedArticle] as expected (Type: {type(res.output)}). Attempting fallback parsing.")
            if isinstance(res.output, str):
                 try:
                     RankedArticleListAdapter = TypeAdapter(List[RankedArticle])
                     ranked_articles_from_llm = RankedArticleListAdapter.validate_json(res.output)
                 except ValidationError as val_err:
                     logger.error(f"Fallback JSON validation failed: {val_err}", exc_info=True) # Added exc_info
                     raise
                 except Exception as json_err:
                     logger.error(f"Failed to parse fallback JSON string: {json_err}", exc_info=True) # Added exc_info
                     raise
            else:
                 logger.error(f"Unexpected output type from agent: {type(res.output)}")
                 raise TypeError(f"Cannot process agent output type: {type(res.output)}")

        # --- 4. Post-processing: Fill Missing Fields --- 
        # Create a lookup map from the original articles using abstract_url as key
        original_articles_map = {str(orig.get('abstract_url')): orig for orig in articles if orig.get('abstract_url')}
        
        filled_ranked_articles = []
        processed_urls = set() # To handle potential duplicates from LLM

        for llm_article in ranked_articles_from_llm:
            # Use abstract_url (as string) to find the original data
            original_data = original_articles_map.get(str(llm_article.abstract_url))
            
            if str(llm_article.abstract_url) in processed_urls:
                 logger.warning(f"Skipping duplicate article from LLM output: {llm_article.title}")
                 continue
            processed_urls.add(str(llm_article.abstract_url))

            if original_data:
                # Create a new dictionary merging original data with LLM output
                # Prioritize LLM fields where they exist and are valid (handled by initial Pydantic validation)
                # But fill from original_data if LLM omitted an optional field
                merged_data = original_data.copy() # Start with original data
                merged_data.update(llm_article.model_dump(exclude_unset=True)) # Overlay non-None fields from LLM output
                
                try:
                    # Re-validate the merged data to ensure consistency
                    final_article = RankedArticle(**merged_data)
                    filled_ranked_articles.append(final_article)
                except ValidationError as e:
                    logger.warning(f"Validation failed after merging LLM output with original data for '{llm_article.title}'. Error: {e}. Skipping article.")
            else:
                # If original data not found (e.g., LLM hallucinated an article?), keep LLM version if valid
                logger.warning(f"Could not find original data for ranked article: '{llm_article.title}' ({llm_article.abstract_url}). Using LLM output directly.")
                filled_ranked_articles.append(llm_article) # Keep the article from LLM

        logger.info(f"Agent initially returned {len(ranked_articles_from_llm)} articles. After post-processing: {len(filled_ranked_articles)} articles.")
        return filled_ranked_articles # Return the list with filled/verified data

    except ValidationError as e:
        logger.error(f"LLM output failed schema validation: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in rank_articles: {e}", exc_info=True)
        raise

async def analyze_articles(
    user_info: UserContext, # Changed from dict to UserContext
    articles: List[RankedArticle],
    top_n: int = 5
) -> List[ArticleAnalysis]:
    """
    Get LLM-generated analyses for the most relevant articles.

    Args:
        user_info: UserContext object.
        articles: List of RankedArticle objects.
        top_n: Number of articles to analyze

    Returns:
        List of ArticleAnalysis objects.
    """
    # user = UserContext(**user_info) # No longer needed
    analyses = []
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        logger.info(f"Analyzing top {min(top_n, len(articles))} articles...")
        for art in articles[:top_n]:
            try:
                article_metadata = art.model_dump() # Still need dict for analyze_article tool input
                # Ensure html_url is a string for scrape_article
                html_url_str = str(art.html_url) if art.html_url else None
                
                if not html_url_str:
                    logger.warning(f"Skipping analysis for '{art.title}' (no HTML URL)")
                    continue
                    
                article_content = await scrape_article(crawler, html_url_str)
                
                if not article_content:
                    logger.warning(f"Skipping analysis for '{art.title}' (scraping failed or no content)")
                    continue
                
                # Call analyze_article (which now returns ArticleAnalysis)
                analysis_result = await analyze_article(
                    arxiv_agent, 
                    article_content=article_content,
                    user_context=user_info, # Pass the UserContext object directly
                    article_metadata=article_metadata # Pass the dict derived from RankedArticle
                )
                analyses.append(analysis_result)
            except Exception as e:
                logger.error(f"Failed to analyze article '{art.title}': {e}", exc_info=True)
                continue 
                
    logger.info(f"Completed analysis for {len(analyses)} articles.")
    return analyses

# ============================
#       HTML GENERATION
# ============================

def generate_html_email(
    user_info: UserContext, # Changed from Dict to UserContext
    ranked_articles: List[RankedArticle],
    analyses: List[ArticleAnalysis]
) -> str:
    """Formats the ranked articles and analyses into an HTML string for email."""
    
    # user = UserContext(**user_info) # No longer needed

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<title>ArXiv Digest for {user_info.name}</title> 
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

<h1>ArXiv Digest for {user_info.name}</h1> 
<p>Based on your interests: {user_info.goals}</p> 

<h2>Top Ranked Articles</h2>
<ul>
"""
    
    # Create a mapping from abstract_url to analysis for easy lookup
    analysis_map = {str(a.abstract_url): a for a in analyses} # Use str(url) for key

    for rank_art in ranked_articles:
        html_content += f"""
  <li>
    <h3>{rank_art.title}</h3>
    <p>Authors: {', '.join(rank_art.authors)}</p>
    <p>Subject: {rank_art.subject}</p>
    <p><span class="score">Relevance Score: {rank_art.relevance_score}/100</span> - {rank_art.score_reason}</p>
    <p class="links">
      <a href="{str(rank_art.abstract_url)}" target="_blank">Abstract</a> | 
      <a href="{str(rank_art.html_url) if rank_art.html_url else '#'}" target="_blank">HTML</a> | 
      <a href="{str(rank_art.pdf_url)}" target="_blank">PDF</a>
    </p>
"""
        # Add analysis if available
        analysis = analysis_map.get(str(rank_art.abstract_url)) # Use str(url) for lookup
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
#        TESTING AREA
# ============================
async def main():
    """Example usage demonstrating Pydantic model usage and file loading"""
    # Create UserContext instance
    try:
        user = UserContext(
            name="Dr. Evelyn Reed",
            title="AI Researcher",
            goals="Cutting-edge advancements in Large Language Models and their applications in scientific discovery."
        )
    except ValidationError as e:
        logger.error(f"Failed to create UserContext: {e}")
        return
    
    # Define path for article data file
    default_arxiv_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'arxiv_cs_submissions_2025-04-01.json')
    arxiv_file_path = os.getenv("ARXIV_FILE", default_arxiv_file)
    logger.info(f"Attempting to load articles from: {arxiv_file_path}")

    # Attempt to load articles from the JSON file
    articles_to_process = []
    try:
        with open(arxiv_file_path, "r", encoding="utf-8") as f:
            articles_to_process = json.load(f)
        if not isinstance(articles_to_process, list):
            logger.error(f"Loaded data from {arxiv_file_path} is not a list. Type: {type(articles_to_process)}")
            articles_to_process = [] # Reset to empty list
        else:
             logger.info(f"Successfully loaded {len(articles_to_process)} articles from {arxiv_file_path}.")
    except FileNotFoundError:
        logger.warning(f"Article data file not found: {arxiv_file_path}. Using fallback sample data.")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from {arxiv_file_path}: {e}. Using fallback sample data.")
    except Exception as e:
        logger.error(f"An unexpected error occurred loading {arxiv_file_path}: {e}. Using fallback sample data.")

    # Fallback to sample data if loading failed or produced no articles
    if not articles_to_process:
        logger.warning("Using hardcoded sample article data.")
        articles_to_process = [
             {
                 "title": "Self-Consuming Generative Models Go Mad",
                 "authors": ["R. Rombach", "A. Blattmann"],
                 "subject": "cs.LG",
                 "score_reason": "Directly related to LLM behavior",
                 "relevance_score": 95,
                 "abstract_url": "https://arxiv.org/abs/2307.01850",
                 "html_url": "https://arxiv.org/html/2307.01850",
                 "pdf_url": "https://arxiv.org/pdf/2307.01850.pdf"
             },
             {
                 "title": "Attention Is All You Need",
                 "authors": ["A. Vaswani", "et al."],
                 "subject": "cs.CL",
                 "score_reason": "Fundamental paper for modern LLMs",
                 "relevance_score": "90",
                 "abstract_url": "https://arxiv.org/abs/1706.03762",
                 "html_url": None,
                 "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"
             },
            # Add more samples if needed
        ]
        
    if not articles_to_process:
        logger.error("No articles available to process (neither from file nor fallback). Aborting.")
        return

    # Determine how many articles to rank
    top_n = int(os.getenv("TOP_N_ARTICLES", "5"))

    try:
        logger.info("--- Ranking Articles ---")
        # Pass UserContext object and the loaded (or sample) raw article dicts
        ranked_articles = await rank_articles(user, articles_to_process, top_n=top_n)
        logger.info(f"Top {min(top_n, len(articles_to_process))} articles ranked (received {len(ranked_articles)}): ") # Adjusted log
        for article in ranked_articles:
            logger.info(f"  - {article.title} (Score: {article.relevance_score}) - ID: {article.arxiv_id}")

        if not ranked_articles:
            logger.warning("No articles were ranked successfully. Skipping analysis and email generation.")
            return

        logger.info("--- Analyzing Top Articles ---")
        # Pass UserContext object and List[RankedArticle]
        # Analyze only the top_n ranked articles, matching the number requested for ranking
        analyses = await analyze_articles(user, ranked_articles, top_n=top_n) 
        logger.info(f"Analyses generated for {len(analyses)} articles:")
        for analysis in analyses:
            logger.info(f"  - Analysis for: {analysis.title}")
            logger.info(f"    Summary: {analysis.summary[:50]}...")
            logger.info(f"    Action: {analysis.recommended_action}")

        logger.info("--- Generating HTML Email ---")
        # Pass UserContext object, List[RankedArticle], List[ArticleAnalysis]
        html_output = generate_html_email(user, ranked_articles, analyses)
        
        output_path = "arxiv_digest_output.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_output)
        logger.info(f"HTML email content saved to {output_path}")

    except ValidationError as e:
        logger.error(f"Pydantic validation error during execution: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main()) 