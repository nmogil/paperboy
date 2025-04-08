# agent_prompts.py
from __future__ import annotations as _annotations
import json
from typing import List, Dict, Any
# Import Pydantic models for type hinting and structure reference if needed
# from agent import UserInfo, ArticleData, RankedArticle, ArticleInsights, EnrichedArticle

# Note: While we could import the Pydantic models, for clarity in the prompts,
# we will explicitly describe the expected JSON structure within the prompt strings.

RANKING_PROMPT = """
You are an AI assistant tasked with ranking scientific articles based on their relevance to a user's profile.
Analyze the provided list of articles and the user's information, then return a ranked list.

User Information:
{user_info}

Articles to Rank:
{articles}

Instructions:
1. Review each article's title, abstract (implicitly via metadata), authors, and subjects.
2. Compare the article content and subjects against the user's job title, company, and stated goals.
3. Assign a rank to each article, starting from 1 for the most relevant.
4. Provide a brief justification (reason_for_ranking) for each article's rank, explaining why it is relevant (or not) to the user.
5. Output the results as a JSON list of objects. Each object must strictly adhere to the following structure:
   {{
     "rank": <integer>,
     "article": {{ <original article data as provided> }},
     "reason_for_ranking": "<string justification>"
   }}

Ensure the output is a valid JSON list containing all the provided articles, each with a rank and reason.
The 'article' field in the output must contain the *exact* same data structure and content as the corresponding article in the input list.
"""

INSIGHTS_PROMPT = """
You are an AI assistant tasked with generating detailed insights for a specific scientific article, tailored to a user's profile.

User Information:
{user_info}

Article Data:
{article}

Instructions:
1. Thoroughly analyze the provided article data (title, abstract, subjects, etc.).
2. Generate the following insights, keeping the user's profile (job title, company, goal) in mind:
    * ai_summary: A concise summary of the article's main points and findings.
    * ai_key_take_aways: A list of the most important findings or conclusions from the article.
    * personalized_summary: A summary specifically explaining the article's content in the context of the user's interests and goals.
    * why_it_matters_to_user: An explanation of the article's significance and potential impact on the user's work or field.
    * relevance_score: A numerical score between 0.0 and 1.0 indicating the article's relevance to the user (0.0 = not relevant, 1.0 = highly relevant).
    * tags: A list of relevant keywords or tags for the article.
    * length_minutes: An estimated reading time in minutes (optional, can be null if unknown).
3. Output the results as a single JSON object. The object must strictly adhere to the following structure:
   {{
     "ai_summary": "<string>",
     "ai_key_take_aways": ["<string>", ...],
     "personalized_summary": "<string>",
     "why_it_matters_to_user": "<string>",
     "relevance_score": <float between 0.0 and 1.0>,
     "tags": ["<string>", ...],
     "length_minutes": <integer or null>
   }}

Ensure the output is a valid JSON object matching this structure exactly.
"""

NEWSLETTER_PROMPT = """
You are an AI assistant tasked with creating a personalized newsletter in Markdown format summarizing a list of relevant scientific articles for a user.

User Information:
{user_info}

Enriched Articles (Ranked and Summarized):
{enriched_articles}

Instructions:
1. Review the user information and the list of enriched articles provided. Each article includes its rank, original data, AI-generated summaries, key takeaways, relevance score, and personalized insights.
2. Compose a friendly and informative newsletter addressed to the user ({user_info['first_name']}).
3. The newsletter should highlight the key articles based on their rank and relevance.
4. For each highlighted article, include:
    * Title (linked to the abstract_url if possible in Markdown)
    * Rank
    * Personalized Summary
    * Why it Matters to the User
    * Key Takeaways (briefly, perhaps as bullet points)
5. Structure the newsletter logically, perhaps starting with the highest-ranked articles.
6. Use Markdown formatting for headings, lists, bold text, and links to enhance readability.
7. Ensure the tone is appropriate for a professional newsletter aimed at keeping the user informed about relevant research.

Output the complete newsletter content as a single Markdown string.
"""

# Helper function to format inputs for prompts if needed (optional)
def format_articles_for_prompt(articles: List[Dict[str, Any]]) -> str:
    """Formats a list of article dictionaries for inclusion in a prompt."""
    return json.dumps([article for article in articles], indent=2)

def format_user_info_for_prompt(user_info: Dict[str, Any]) -> str:
    """Formats user info dictionary for inclusion in a prompt."""
    return json.dumps(user_info, indent=2)