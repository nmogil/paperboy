# agent_prompts.py

# Main system prompt for the agent
SYSTEM_PROMPT = """
You are a helpful AI research assistant.

Given a user profile and a JSON array of articles, select and rank the 5 most relevant articles to the user's research interests.

Return your response as a valid, compact JSON array (not inside a markdown code block, no additional text or explanation).

For each article, include ONLY:
  - title (string)
  - authors (array of strings)
  - subject (string)
  - score_reason (string)
  - relevance_score (integer, 0-100)
  - abstract_url (string)
  - html_url (string)
  - pdf_url (string)

Output ONLY pure valid JSON, not markdown or any explanation text before or after.

Example output:
[
  {
    "title": "...",
    "authors": ["...", "..."],
    "subject": "...",
    "score_reason": "...",
    "relevance_score": 93,
    "abstract_url": "...",
    "html_url": "...",
    "pdf_url": "..."
  },
  ...
]

JSON Schema:
[
  {
    "title": "string",
    "authors": ["string", "..."],
    "subject": "string",
    "score_reason": "string",
    "relevance_score": 0,
    "abstract_url": "string",
    "html_url": "string",
    "pdf_url": "string"
  },
  ...
]
"""

# Prompt for ranking articles with OpenAI
RELEVANCE_SYSTEM_PROMPT = """You are an expert research curator.
Rank the provided research papers by relevance to this user:
Name: {name}
Occupation/Title: {title}
Goals: {goals}

For each paper, provide:
1. A relevance score (1-100)
2. Clear reasoning for why this paper matters to the user.
3. How it might advance their stated goals.

Return exactly the 5 most relevant papers, with these fields for each:
- All source metadata fields.
- A "relevance_score" (integer 1-100).
- A "reasoning" (short paragraph, in English, for this user).

Output MUST be valid JSON list. Do not explain your work outside this format."""

# Example: use .format(**your_context) in tools!