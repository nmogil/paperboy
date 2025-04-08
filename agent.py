from __future__ import annotations as _annotations

import httpx
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.llm.openai import OpenAI # Added for OpenAI client
import asyncio # Added for async execution
import os # Added for environment variables
import dotenv # Added for loading .env file

# Import prompts and formatting helpers
import agent_prompts
from agent_prompts import format_articles_for_prompt, format_user_info_for_prompt

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Input Data Structures ---
class UserInfo(BaseModel):
    """User profile information."""
    first_name: str
    job_title: str # Corrected typo from job_tittle
    company: str
    goal: str

class ArticleData(BaseModel):
    arxiv_id: str
    title: str
    abstract_url: str
    pdf_url: str
    authors: List[str]
    subjects: List[str]
    primary_subject: str
    comments: Optional[str] = None

class RankedArticle(BaseModel):
    rank: int
    article: ArticleData
    reason_for_ranking: str

class ArticleInsights(BaseModel):
    ai_summary: str = Field(..., description="Concise summary of the article.")
    ai_key_take_aways: List[str] = Field(..., description="Key findings of the article.")
    personalized_summary: str = Field(..., description="Summary tailored to user.")
    why_it_matters_to_user: str = Field(..., description="Explanation of significance to user.")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    tags: List[str] = Field(..., description="Keywords for the article.")
    length_minutes: Optional[int] = Field(None, description="Estimated reading time.")

class EnrichedArticle(ArticleData, ArticleInsights):
    """Combined article data with AI-generated insights."""
    rank: int
    reason_for_ranking: str

# --- Parsing Function ---
def parse_html(html_content: str) -> List[Dict[str, Any]]:
    """Parses the HTML content of an arXiv catchup page to extract article data."""
    soup = BeautifulSoup(html_content, 'html.parser')
    articles_data = []
    base_url = "https://arxiv.org"

    # Find all definition list items which contain article info
    dts = soup.find_all('dt')
    dds = soup.find_all('dd')

    if len(dts) != len(dds):
        logger.warning("Mismatch between number of <dt> and <dd> tags. Parsing might be incomplete.")
        # Attempt to parse based on the shorter list length to avoid index errors
        min_len = min(len(dts), len(dds))
        dts = dts[:min_len]
        dds = dds[:min_len]

    for dt, dd in zip(dts, dds):
        try:
            article = {}

            # Extract links and ID from <dt>
            links = dt.find_all('a')
            if not links: continue # Skip if no links found

            # Abstract link and ID
            abstract_link_tag = links[0]
            article['abstract_url'] = urljoin(base_url, abstract_link_tag['href'])
            arxiv_id_text = abstract_link_tag.get_text(strip=True)
            if arxiv_id_text.startswith("arXiv:"):
                article['arxiv_id'] = arxiv_id_text.split("arXiv:")[1]
            else:
                # Fallback or logging if format is unexpected
                logger.warning(f"Could not parse arXiv ID from text: {arxiv_id_text}")
                continue # Skip this article if ID is crucial and missing

            # PDF link
            pdf_link_tag = next((link for link in links if 'pdf' in link.get_text(strip=True).lower()), None)
            if pdf_link_tag and pdf_link_tag.has_attr('href'):
                 article['pdf_url'] = urljoin(base_url, pdf_link_tag['href'])
            else:
                 article['pdf_url'] = None # Or handle as needed

            # Extract details from <dd>
            title_div = dd.find('div', class_='list-title')
            article['title'] = title_div.get_text(strip=True).replace("Title:", "").strip() if title_div else "N/A"

            authors_div = dd.find('div', class_='list-authors')
            authors_text = authors_div.get_text(strip=True).replace("Authors:", "").strip() if authors_div else ""
            # Remove potential HTML tags within author names if any (though usually just text)
            authors_soup = BeautifulSoup(authors_text, 'html.parser')
            article['authors'] = [a.strip() for a in authors_soup.get_text().split(',')] if authors_text else []


            subjects_div = dd.find('div', class_='list-subjects')
            subjects_text = subjects_div.get_text(strip=True).replace("Subjects:", "").strip() if subjects_div else ""
            article['subjects'] = [s.strip() for s in subjects_text.split(';')] if subjects_text else []

            primary_subject_span = dd.find('span', class_='primary-subject')
            article['primary_subject'] = primary_subject_span.get_text(strip=True) if primary_subject_span else ("N/A" if not article['subjects'] else article['subjects'][0].split(' (')[0]) # Fallback logic

            comments_div = dd.find('div', class_='list-comments')
            article['comments'] = comments_div.get_text(strip=True).replace("Comments:", "").strip() if comments_div else None

            articles_data.append(article)

        except Exception as e:
            logger.error(f"Error parsing an article entry: {e}", exc_info=True)
            # Optionally skip this article or handle error differently
            continue

    return articles_data

# --- Main Functions ---
async def fetch_articles(date: str) -> List[ArticleData]:
    """Fetch latest articles from arXiv using the specified date."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://arxiv.org/catchup/cs/{date}")
        response.raise_for_status()  # Raise for HTTP errors
        # Parsing of HTML would be handled here
        articles_data = parse_html(response.text)  # Implement real parsing.
        return [ArticleData(**article) for article in articles_data]

async def research_and_summarize_articles(ctx: RunContext, user_info: UserInfo, ranked_articles_input: List[RankedArticle]) -> List[EnrichedArticle]:
    enriched_articles = []
    user_info_str = format_user_info_for_prompt(user_info.dict())

    for ranked_article in ranked_articles_input:
        article = ranked_article.article
        article_str = json.dumps(article.dict(), indent=2) # Format single article for prompt

        # Format the prompt with user and article info
        formatted_prompt = agent_prompts.INSIGHTS_PROMPT.format(
            user_info=user_info_str,
            article=article_str
        )

        # Call the model, expecting an ArticleInsights object back
        # pydantic-ai handles parsing and validation via response_model
        insights_response = await ctx.call_model(
            prompt=formatted_prompt,
            response_model=ArticleInsights
        )
        insights = insights_response.data # .data should contain the validated ArticleInsights instance

        enriched_article = EnrichedArticle(**article.dict(), **insights.dict(), rank=ranked_article.rank, reason_for_ranking=ranked_article.reason_for_ranking)
        enriched_articles.append(enriched_article)

    return enriched_articles

async def generate_markdown_newsletter(ctx: RunContext, user_info: UserInfo, enriched_articles: List[EnrichedArticle]) -> str:
    # Format inputs for the prompt
    user_info_str = format_user_info_for_prompt(user_info.dict())
    enriched_articles_str = format_articles_for_prompt([article.dict() for article in enriched_articles])

    # Format the main newsletter prompt
    formatted_prompt = agent_prompts.NEWSLETTER_PROMPT.format(
        user_info=user_info_str,
        enriched_articles=enriched_articles_str
    )

    # Call the model, expecting a string back
    response = await ctx.call_model(prompt=formatted_prompt) # No response_model needed for string
    return response.text.strip() # Assuming .text contains the raw string output

async def main(ctx: RunContext, user_info_data: dict, date: str) -> str:
    try:
        user_info = UserInfo(**user_info_data)
        logger.info(f"User info validated: {user_info.first_name}")
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return "Invalid user information provided."

    articles = await fetch_articles(date)
    # Format inputs for the ranking prompt
    user_info_str = format_user_info_for_prompt(user_info.dict())
    articles_str = format_articles_for_prompt([article.dict() for article in articles])

    # Format the ranking prompt
    formatted_ranking_prompt = agent_prompts.RANKING_PROMPT.format(
        user_info=user_info_str,
        articles=articles_str
    )

    # Call the model, expecting a List[RankedArticle] back
    # pydantic-ai handles parsing and validation via response_model
    ranked_articles_response = await ctx.call_model(
        prompt=formatted_ranking_prompt,
        response_model=List[RankedArticle]
    )
    ranked_articles = ranked_articles_response.data # .data should contain the validated List[RankedArticle]
    enriched_articles = await research_and_summarize_articles(ctx, user_info, ranked_articles)
    markdown_newsletter = await generate_markdown_newsletter(ctx, user_info, enriched_articles)
    return markdown_newsletter

if __name__ == "__main__":
    logger.info("Starting agent execution...")

    # Load environment variables from .env file
    dotenv.load_dotenv()
    logger.info("Loaded environment variables.")

    # Retrieve OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        print("Error: OPENAI_API_KEY environment variable not set. Please create a .env file with this key.")
        exit(1) # Exit if key is missing
    logger.info("OpenAI API key loaded.")

    # Instantiate the OpenAI client
    # You might want to specify a model, e.g., model="gpt-4-turbo-preview" or model="gpt-3.5-turbo"
    # Depending on your OpenAI access and desired capability/cost balance.
    openai_client = OpenAI(api_key=api_key)
    logger.info(f"OpenAI client instantiated with model: {openai_client.model}") # Log the model being used

    # Instantiate RunContext
    ctx = RunContext(llm=openai_client)
    logger.info("RunContext instantiated.")

    # Define placeholder user info and date
    # In a real application, this would come from command-line arguments, a config file, or an API request.
    placeholder_user_info = {
        "first_name": "Alex",
        "job_title": "AI Researcher", # Using corrected field name
        "company": "FutureTech",
        "goal": "Stay updated on the latest advancements in computer vision and natural language processing."
    }
    # Use today's date or a specific date for testing
    # Format YYYYMMDD is expected by fetch_articles based on the example URL structure
    placeholder_date = datetime.now().strftime('%Y%m%d') # e.g., '20250407'
    logger.info(f"Using placeholder user info for {placeholder_user_info['first_name']} and date {placeholder_date}")


    # Run the main async function
    try:
        logger.info("Executing main function...")
        # Note: Ensure agent_prompts.py defines the necessary prompts for the call_model functions
        # ("rank_articles", "generate_article_insights", "generate_markdown_newsletter")
        # and that they are registered with the RunContext if using Agent features,
        # or handled appropriately if RunContext is used directly as shown.
        newsletter = asyncio.run(main(ctx, placeholder_user_info, placeholder_date))
        logger.info("Main function executed successfully.")
        print("\n--- Generated Newsletter ---")
        print(newsletter)
        print("--------------------------\n")
    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error occurred during article fetching: {http_err}", exc_info=True)
        print(f"\n--- Error ---")
        print(f"Failed to fetch articles for date {placeholder_date}. Status code: {http_err.response.status_code}")
        print(f"Content: {http_err.response.text[:500]}...") # Print first 500 chars of error response
        print("-------------\n")
    except ValidationError as val_err:
        logger.error(f"Data validation error: {val_err}", exc_info=True)
        print(f"\n--- Error ---")
        print(f"Data validation failed: {val_err}")
        print("-------------\n")
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}", exc_info=True)
        print(f"\n--- Error ---")
        print(f"An unexpected error occurred: {e}")
        print("-------------\n")