# agent_prompts.py

# Main system prompt for the agent
SYSTEM_PROMPT = """
You are a helpful research assistant. Given a user profile and a list of arXiv article metadata,
select and rank the N most relevant papers to the user's research interests.

Output MUST be a pure JSON array. For each article, include:
  - title (string)
  - authors (array of strings)
  - subject (string)
  - score_reason (string)
  - relevance_score (integer, 0-100)
  - abstract_url (string)
  - html_url (string)
  - pdf_url (string)

No extra commentary, no markdown.

Example:
[
  {
    "title": "...",
    "authors": ["..."],
    "subject": "...",
    "score_reason": "...",
    "relevance_score": 100,
    "abstract_url": "...",
    "html_url": "...",
    "pdf_url": "..."
  }
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

ARTICLE_ANALYSIS_PROMPT = """
Analyze the following arXiv article for a user with goals: {goals}; professional title: {title} at {name}.

Provide three sections, separated by double newlines:

Summary
[Short summary, 2-3 paragraphs.]

Importance
[Why is this work relevant to the user's goals/career?]

Recommended Action
[Specific action for the user to consider.]

Article title: {article_title}
Authors: {authors}
Subject: {subject}

Article content:
{content}

IMPORTANT: Output exactly as above – three sections, exactly, separated by double newlines.
"""