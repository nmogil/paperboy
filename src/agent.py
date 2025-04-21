# agent_fixed.py

from __future__ import annotations as _annotations

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
import json

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator, TypeAdapter
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import models from the new central location
from .models import RankedArticle, ArticleAnalysis, UserContext, ScrapedArticle
from .agent_prompts import SYSTEM_PROMPT, ARTICLE_ANALYSIS_PROMPT
from .agent_tools import scrape_article, analyze_article
from crawl4ai import AsyncWebCrawler # Import the crawler

# Import settings from .config
from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO), # Use settings
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
        settings.openai_model, # Use settings
        provider=OpenAIProvider(api_key=settings.openai_api_key) # Use settings
    ),
    system_prompt=SYSTEM_PROMPT,
    deps_type=Deps,
    retries=settings.agent_retries, # Use settings
    response_model=List[RankedArticle],   # Uses imported RankedArticle
)

# Note: Tools are imported and registered in agent_tools.py using @arxiv_agent.tool

# ============================
#     HELPER FUNCTIONS
# ============================

def _merge_llm_and_original_articles(
    llm_articles: List[RankedArticle],
    original_articles_map: Dict[str, Dict[str, Any]]
) -> List[RankedArticle]:
    """Merges LLM output with original article data for completeness."""
    filled_ranked_articles = []
    processed_urls = set() # To handle potential duplicates from LLM

    for llm_article in llm_articles:
        # Use abstract_url (as string) to find the original data
        original_data = original_articles_map.get(str(llm_article.abstract_url))

        if str(llm_article.abstract_url) in processed_urls:
            logger.warning(f"Skipping duplicate article from LLM output: {llm_article.title}")
            continue
        processed_urls.add(str(llm_article.abstract_url))

        if original_data:
            # Create a new dictionary merging original data with LLM output
            # Prioritize LLM fields where they exist and are valid
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
            # If original data not found, keep LLM version if valid
            logger.warning(f"Could not find original data for ranked article: '{llm_article.title}' ({llm_article.abstract_url}). Using LLM output directly.")
            filled_ranked_articles.append(llm_article) # Keep the article from LLM

    return filled_ranked_articles


async def _scrape_ranked_articles(
    ranked_articles: List[RankedArticle],
    crawler: AsyncWebCrawler
) -> List[ScrapedArticle]:
    """Concurrently scrapes the HTML content for a list of ranked articles."""
    tasks = []
    for article in ranked_articles:
        if article.html_url:
            tasks.append(scrape_article(crawler, str(article.html_url)))
        else:
            # Append a placeholder for articles without HTML URL
            # Create a future that immediately returns None
            future = asyncio.get_event_loop().create_future()
            future.set_result((None, "No HTML URL provided")) # Tuple (content, error)
            tasks.append(future)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    scraped_data = []
    for i, result in enumerate(results):
        article = ranked_articles[i]
        content = None
        error_msg = None
        if isinstance(result, Exception):
            error_msg = f"Scraping failed: {result}"
            logger.warning(f"Error scraping {article.html_url} for '{article.title}': {result}")
        elif isinstance(result, tuple): # Handle the (None, "No HTML URL") case
             content, error_msg = result
        else:
            content = result # Successful scrape
            if not content:
                error_msg = "Scraping returned no content"
                logger.warning(f"Scraping returned no content for {article.html_url} ('{article.title}')")


        scraped_data.append(
            ScrapedArticle(
                article=article,
                scraped_content=content,
                scrape_error=error_msg
            )
        )

    return scraped_data


# ============================
#     MAIN AGENT LOGIC
# ============================

async def rank_articles(
    user_info: UserContext,
    articles: List[Dict[str, Any]],
    top_n: int = settings.top_n_articles
) -> List[RankedArticle]:
    """
    Runs the LLM to rank articles for the given user profile.
    Includes post-processing to ensure all available fields are populated.

    Args:
        user_info: UserContext object containing user profile.
        articles: List of raw article dictionaries.
        top_n: How many top articles to select (defaults to value in settings)

    Returns:
        List of validated RankedArticle objects with fields filled from original data.
    """
    input_articles = articles[:settings.ranking_input_max_articles] # Limit input size using setting

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
            # Fallback parsing
            logger.warning(f"Agent output was not List[RankedArticle] as expected (Type: {type(res.output)}). Attempting fallback parsing.")
            if isinstance(res.output, str):
                 try:
                     RankedArticleListAdapter = TypeAdapter(List[RankedArticle])
                     ranked_articles_from_llm = RankedArticleListAdapter.validate_json(res.output)
                 except ValidationError as val_err:
                     logger.error(f"Fallback JSON validation failed: {val_err}", exc_info=True)
                     raise
                 except Exception as json_err:
                     logger.error(f"Failed to parse fallback JSON string: {json_err}", exc_info=True)
                     raise
            else:
                 logger.error(f"Unexpected output type from agent: {type(res.output)}")
                 raise TypeError(f"Cannot process agent output type: {type(res.output)}")

        # --- 4. Post-processing: Use Helper Function --- 
        # Create a lookup map from the original articles using abstract_url as key
        original_articles_map = {str(orig.get('abstract_url')): orig for orig in articles if orig.get('abstract_url')}

        # Call the extracted helper function
        filled_ranked_articles = _merge_llm_and_original_articles(
            ranked_articles_from_llm,
            original_articles_map
        )

        logger.info(f"Agent initially returned {len(ranked_articles_from_llm)} articles. After post-processing: {len(filled_ranked_articles)} articles.")
        return filled_ranked_articles # Return the list with filled/verified data

    except ValidationError as e:
        # Use ModelRetry as suggested in refactoring plan Step 6
        logger.error(f"LLM output failed schema validation, attempting retry: {e}", exc_info=True)
        raise ModelRetry(f"Schema validation failed, retrying: {e}") from e
    except Exception as e:
        logger.error(f"Error in rank_articles: {e}", exc_info=True)
        raise

async def analyze_articles(
    user_info: UserContext,
    scraped_articles: List[ScrapedArticle], # Changed input type
    top_n: int = settings.top_n_articles
) -> List[ArticleAnalysis]: # Return type remains the same
    """
    Get LLM-generated analyses for the most relevant articles using pre-scraped content.

    Args:
        user_info: UserContext object.
        scraped_articles: List of ScrapedArticle objects containing metadata and content/error.
        top_n: Number of articles to analyze (defaults to value in settings)

    Returns:
        List of ArticleAnalysis objects.
    """
    analyses = []

    # Crawler is no longer needed here
    logger.info(f"Analyzing top {min(top_n, len(scraped_articles))} articles...")
    articles_to_analyze = scraped_articles[:top_n]

    for item in articles_to_analyze:
        art = item.article # Get the RankedArticle part
        article_content = item.scraped_content
        scrape_error = item.scrape_error

        if scrape_error:
            logger.warning(f"Skipping analysis for '{art.title}' due to scraping error: {scrape_error}")
            continue
        if not article_content:
            # This case might be redundant if scrape_error covers it, but good for safety
            logger.warning(f"Skipping analysis for '{art.title}' (no content available after scraping attempt)")
            continue

        try:
            # Call analyze_article tool
            analysis_result = await analyze_article(
                arxiv_agent,
                article_content=article_content,
                user_context=user_info,
                article_metadata=art.model_dump() # Pass metadata dict
            )
            analyses.append(analysis_result)
        except Exception as e:
            logger.error(f"Failed to analyze article '{art.title}': {e}", exc_info=True)
            # Decide if you want to stop analysis or just skip this article
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
#        MAIN EXECUTION
# ============================

async def main():
    logger.info("Starting ArXiv Agent...")
    # --- User Profile ---
    user_info = UserContext(
        name="Dr. Evelyn Reed",
        title="Computational Linguist",
        goals="Stay updated on Large Language Models, specifically Transformer architectures, model optimization techniques, and ethical considerations in AI."
    )

    # --- 1. Read ArXiv data --- 
    try:
        with open(settings.arxiv_file, 'r') as f:
            raw_articles = json.load(f)
        logger.info(f"Loaded {len(raw_articles)} articles from {settings.arxiv_file}")
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {settings.arxiv_file}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {settings.arxiv_file}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred reading the input file: {e}", exc_info=True)
        return

    # Create crawler instance once
    async with AsyncWebCrawler(verbose=False) as crawler:
        try:
            # --- 2. Rank Articles --- 
            logger.info("Ranking articles...")
            ranked_articles = await rank_articles(user_info, raw_articles)
            logger.info(f"Ranking complete. Got {len(ranked_articles)} relevant articles.")
            if not ranked_articles:
                logger.info("No relevant articles found after ranking. Exiting.")
                return

            # --- 3. Scrape Ranked Articles --- 
            logger.info("Scraping content for ranked articles...")
            # Pass the crawler instance to the new scraping function
            scraped_article_data = await _scrape_ranked_articles(ranked_articles, crawler)
            successful_scrapes = sum(1 for item in scraped_article_data if item.scraped_content is not None)
            logger.info(f"Scraping complete. Successfully scraped content for {successful_scrapes} out of {len(ranked_articles)} articles.")

            # --- 4. Analyze Articles --- 
            logger.info("Analyzing scraped articles...")
            # Pass the scraped data to analyze_articles
            analyses = await analyze_articles(user_info, scraped_article_data)
            logger.info(f"Analysis complete. Generated {len(analyses)} analyses.")

            # --- 5. Generate HTML Email --- 
            logger.info("Generating HTML digest...")
            html_output = generate_html_email(user_info, ranked_articles, analyses)

            # --- 6. Save Output --- 
            output_filename = "arxiv_digest.html"
            with open(output_filename, "w") as f:
                f.write(html_output)
            logger.info(f"HTML digest saved to {output_filename}")

        except ModelRetry as e:
            logger.error(f"Agent failed after retries during ranking: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An error occurred during the main agent execution: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main()) 