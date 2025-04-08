from __future__ import annotations as _annotations

import httpx
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field # Assuming BaseModel is needed for Pydantic models
# Assuming RunContext and FunctionModel might be needed if this were a tool,
# but it's presented as a direct function call. Let's assume ctx is globally available
# or passed implicitly in the Archon environment where this function runs.
# from pydantic_ai import FunctionModel, RunContext
import json # Added for potential prompt formatting

logger = logging.getLogger(__name__)

# --- Input Data Structures ---
class UserInfo(BaseModel):
     """User profile information."""
     first_name: str
     job_tittle: str # Note: Typo in original spec, keeping as 'tittle' for consistency
     company: str
     goal: str
 

class ArticleData(BaseModel):
    arxiv_id: str
    title: str
    abstract_url: str
    pdf_url: str
    html_url: Optional[str] = None # Added based on example output
    authors: List[str]
    subjects: List[str]
    primary_subject: str
    comments: Optional[str] = None # Made optional as it might not always exist

class RankedArticle(BaseModel):
    rank: int
    article: ArticleData
    reason_for_ranking: str

# --- AI Output Structure ---

class ArticleInsights(BaseModel):
    ai_summary: str = Field(..., description="Concise, objective summary of the article's content.")
    ai_key_take_aways: List[str] = Field(..., description="Bullet points highlighting the main findings or contributions.")
    personalized_summary: str = Field(..., description="Summary specifically tailored to the user's job title and goal.")
    why_it_matters_to_user: str = Field(..., description="Explanation of the article's significance for the user.")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Numerical score (0.0 to 1.0) indicating relevance.")
    tags: List[str] = Field(..., description="Relevant keywords or tags for the article.")
    length_minutes: Optional[int] = Field(None, description="Estimated reading time in minutes (null if not determinable).")

# --- Enriched Article Structure (Input to Newsletter Function) ---

class EnrichedArticle(ArticleData, ArticleInsights):
    """Combined article data with AI-generated insights."""
    # Inherits fields from ArticleData and ArticleInsights
    # Add any fields present in the example input but not in the base models
    rank: Optional[int] = None # Assuming rank might be passed along
    reason_for_ranking: Optional[str] = None # Assuming reason might be passed along
    # Note: If 'rank' and 'reason_for_ranking' are *always* present, remove Optional

# --- Main Functions ---

# Assuming 'ctx' is available in the execution context provided by Archon
# If not, this function signature might need adjustment based on how Archon invokes functions/steps.
async def research_and_summarize_articles(ctx: Any, user_info: Dict[str, Any], ranked_articles_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Researches and summarizes a list of ranked arXiv articles based on user profile.

    Args:
        ctx: The Archon execution context (provides call_model, etc.).
        user_info: Dictionary containing user profile information.
        ranked_articles_input: List of dictionaries, each representing a ranked article.

    Returns:
        A list of enriched article objects, including AI-generated summaries and insights.
        If processing fails for an article, an 'error' key is added to its dict.
    """
    enriched_articles_output = []
    # Validate input user info
    try:
        user = UserInfo(**user_info)
        logger.info(f"Validated user info for: {user.first_name}")
    except Exception as e:
        logger.error(f"Input validation error for user_info: {str(e)}. Input received: {user_info}")
        # Cannot proceed without valid user info
        raise ValueError(f"Invalid user_info provided: {str(e)}")

    # Validate input articles using Pydantic
    try:
        ranked_articles = [RankedArticle(**item) for item in ranked_articles_input]
        logger.info(f"Successfully validated {len(ranked_articles)} input articles.")
    except Exception as e:
        logger.error(f"Input validation error for ranked_articles: {str(e)}. Input received: {ranked_articles_input}")
        # Return empty list or raise a specific error if input is fundamentally wrong
        return [] # Or raise ValueError("Invalid ranked_articles_input provided")

    # Using a timeout for external requests
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        for ranked_article in ranked_articles:
            article = ranked_article.article
            abstract_text: Optional[str] = None
            # Convert Pydantic model to dict early for easier merging later
            article_dict = article.model_dump(exclude_none=True)
            # Add rank and reason back for potential use in enrichment or later steps
            article_dict['rank'] = ranked_article.rank
            article_dict['reason_for_ranking'] = ranked_article.reason_for_ranking


            # Attempt to fetch the abstract if URL exists
            if article.abstract_url:
                try:
                    logger.info(f"Attempting to fetch abstract for '{article.title}' from: {article.abstract_url}")
                    # Mimic browser request slightly
                    headers = {'User-Agent': 'Mozilla/5.0 (compatible; PaperboyArchonBot/1.0)'}
                    response = await client.get(article.abstract_url, headers=headers)
                    response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
                    # TODO: Check content-type. If HTML, consider parsing libraries (e.g., BeautifulSoup)
                    # For now, assume text or use raw HTML content.
                    abstract_text = response.text
                    logger.info(f"Successfully fetched abstract for '{article.title}' ({len(abstract_text)} chars).")
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error {e.response.status_code} fetching abstract for '{article.title}' from {article.abstract_url}: {e!r}")
                except httpx.RequestError as e:
                     logger.error(f"Request error fetching abstract for '{article.title}' from {article.abstract_url}: {e!r}")
                except Exception as e:
                    # Catch any other unexpected errors during fetch
                    logger.error(f"Unexpected error fetching abstract for '{article.title}' from {article.abstract_url}: {e!r}")
            else:
                logger.warning(f"No abstract URL provided for article: '{article.title}' ({article.arxiv_id})")

            # Construct the context for the AI model prompt
            prompt_context = {
                "article_title": article.title,
                "article_authors": article.authors,
                "article_subjects": article.subjects,
                "article_primary_subject": article.primary_subject,
                "article_comments": article.comments,
                "reason_for_ranking": ranked_article.reason_for_ranking,
                # Provide fetched abstract or clear indication of absence
                "abstract_text": abstract_text if abstract_text else "Abstract could not be fetched or was not available.",
                "user_info": user.model_dump() # Pass validated user info model as dict
            }

            # Define the desired output structure for the AI
            output_format_schema = ArticleInsights.model_json_schema()

            # Call the AI model to generate insights
            try:
                logger.info(f"Calling AI model 'generate_article_insights' for article: '{article.title}'")
                # Replace "generate_article_insights" with the actual model/function name configured in Archon
                # This name should map to the prompt defined in agent_prompts.py
                ai_response_raw = await ctx.call_model(
                    "generate_article_insights", # This should map to a prompt in agent_prompts.py
                    prompt_context,
                    output_format=output_format_schema # Request structured JSON output
                )

                # Validate and parse the AI response using the Pydantic model
                # Assuming the response object has a 'data' attribute containing the JSON
                if hasattr(ai_response_raw, 'data') and ai_response_raw.data:
                    # Ensure data is dict before validation if needed
                    response_data = ai_response_raw.data
                    if isinstance(response_data, str):
                        try:
                            response_data = json.loads(response_data)
                        except json.JSONDecodeError:
                             logger.error(f"AI response data for '{article.title}' is a string but not valid JSON: {response_data}")
                             enriched_articles_output.append({**article_dict, "error": "AI response data was not valid JSON."})
                             continue # Skip to next article

                    ai_insights = ArticleInsights.model_validate(response_data)
                    ai_insights_dict = ai_insights.model_dump(exclude_none=True)
                    logger.info(f"Successfully generated and validated insights for article: '{article.title}'")

                    # Enrich article data: merge original article dict with AI insights dict
                    enriched_article_data = {**article_dict, **ai_insights_dict}
                    enriched_articles_output.append(enriched_article_data)
                else:
                     logger.error(f"AI response for '{article.title}' was empty or malformed. Response: {ai_response_raw}")
                     enriched_articles_output.append({**article_dict, "error": "AI response was empty or malformed."})

            except Exception as e:
                # Catch errors during AI call or response validation/parsing
                logger.exception(f"AI call or processing error for article '{article.title}': {str(e)}")
                enriched_articles_output.append({
                    **article_dict,
                    "error": f"Failed to generate insights: {str(e)}"
                })
                # Continue with the next article
                continue

    logger.info(f"Finished processing articles. Returning {len(enriched_articles_output)} enriched articles.")
    return enriched_articles_output


async def generate_markdown_newsletter(ctx: Any, user_info: Dict[str, Any], enriched_articles: List[Dict[str, Any]]) -> str:
    """
    Generates a personalized newsletter in Markdown format from user info and enriched articles.

    Args:
        ctx: The Archon execution context (provides call_model, etc.).
        user_info: Dictionary containing user profile information.
        enriched_articles: List of dictionaries, each representing an enriched article.

    Returns:
        A string containing the generated Markdown newsletter.
        Returns an error message string if generation fails.
    """
    # Validate inputs
    try:
        user = UserInfo(**user_info)
        articles = [EnrichedArticle(**item) for item in enriched_articles]
        logger.info(f"Newsletter generation: Validated user '{user.first_name}' and {len(articles)} articles.")
    except Exception as e:
        logger.error(f"Input validation error for newsletter generation: {str(e)}")
        return f"# Error\n\nFailed to validate input data for newsletter generation: {str(e)}"

    # Construct the prompt for the AI model
    prompt = f"""
You are a helpful assistant acting as a Copy Writer. Your task is to generate a personalized newsletter in Markdown format based on the provided user information and a list of enriched arXiv research articles.

**User Information:**
- First Name: {user.first_name}
- Job Title: {user.job_tittle}
- Company: {user.company}
- Goal: {user.goal}

**Instructions:**
1.  Start with a personalized greeting (e.g., "Hi {user.first_name},").
2.  Write a brief introduction explaining that this is a curated list of recent arXiv papers relevant to their role as a {user.job_tittle} at {user.company} and their goal to "{user.goal}".
3.  Iterate through the provided articles and format each one clearly using Markdown. For each article:
    *   Use the article title as a Level 2 Markdown heading (##). Link the title to the `abstract_url`.
    *   List the authors, separated by commas.
    *   Include the "Personalized Summary" (`personalized_summary`).
    *   Include "Why it Matters to You" (`why_it_matters_to_user`).
    *   List the "Key Takeaways" (`ai_key_take_aways`) as a bulleted list.
    *   Provide links to the Abstract (`abstract_url`) and PDF (`pdf_url`). Format as `[Abstract](URL)` and `[PDF](URL)`.
    *   Optionally mention the relevance score (`relevance_score`) or estimated reading time (`length_minutes`) if available.
    *   Use Markdown formatting (like bold text, bullet points) to enhance readability.
4.  Add a brief concluding remark.
5.  Ensure the *entire* output is valid Markdown text. Do not include any preamble or explanation outside the Markdown content itself.

**Enriched Articles Data:**
```json
{json.dumps([article.model_dump(exclude_none=True) for article in articles], indent=2)}
```

Now, generate the Markdown newsletter based on these instructions and data.
"""

    try:
        logger.info(f"Calling AI model 'generate_markdown_newsletter' for user: {user.first_name}")
        # Assuming 'generate_markdown_newsletter' maps to a suitable text generation model/prompt in Archon
        # We expect plain text (Markdown) back, so no output_format schema is specified.
        ai_response = await ctx.call_model(
            "generate_markdown_newsletter", # This name should map to the prompt/model in Archon
            {"prompt": prompt} # Pass the constructed prompt
            # output_format=None # Explicitly stating we want raw text output
        )

        # Assuming the response object has a 'text' or similar attribute for the raw output
        if hasattr(ai_response, 'text') and ai_response.text:
            markdown_output = ai_response.text
            logger.info(f"Successfully generated Markdown newsletter for user: {user.first_name}")
            return markdown_output.strip() # Return the cleaned Markdown string
        elif isinstance(ai_response, str): # Handle cases where the model directly returns a string
             markdown_output = ai_response
             logger.info(f"Successfully generated Markdown newsletter (raw string) for user: {user.first_name}")
             return markdown_output.strip()
        else:
            logger.error(f"AI response for newsletter generation was empty or in unexpected format. Response: {ai_response}")
            return f"# Error\n\nAI response for newsletter generation was empty or in an unexpected format."

    except Exception as e:
        logger.exception(f"AI call error during newsletter generation for user '{user.first_name}': {str(e)}")
        return f"# Error\n\nFailed to generate newsletter due to an AI call error: {str(e)}"