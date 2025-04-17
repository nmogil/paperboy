# Prompt for AI Research Personalization Application

## Project Overview

You are tasked with creating a comprehensive Python application that automatically discovers, filters, and presents personalized research papers from arXiv. This application will leverage the Pydantic AI agent framework to create multiple specialized AI agents that work together to deliver customized research insights in an elegant newsletter format.

## Core Requirements

Build a multi-agent system that:

1. Gathers user information to personalize research findings
2. Fetches the latest arXiv papers using asynchronous requests
3. Intelligently selects the most relevant papers based on user context
4. Provides in-depth analysis and summaries of selected papers
5. Generates a professional newsletter in markdown format

## User Input Collection

The application should begin by gathering essential information from the user:

- First Name: To personalize the newsletter
- Title/Role: To understand their professional context
- Goals: To identify their research interests and career objectives
- Field(s) of Interest: To focus the paper selection (e.g., Machine Learning, NLP, Computer Vision)
- Technical Expertise Level: To adjust the depth of explanations

## Detailed Agent Workflow

### 1. Data Collection Agent

- **Primary Tool**: `crawl4ai` library with `asyncio` for parallel processing
- **Function**: Fetch the latest articles from arXiv using the endpoint pattern: `https://arxiv.org/catchup/{category}/{YYYY-MM-DD}`
- **Implementation Requirements**:
  - Use yesterday's date by default (programmable)
  - Support multiple categories (cs, math, physics, etc.)
  - Handle pagination and rate limiting
  - Store HTML responses for parsing
  - Implement proper error handling for network issues

```python
# Sample code structure
async def fetch_papers(date=None, categories=["cs"]):
    """Fetch papers from specified categories and date"""
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    tasks = []
    for category in categories:
        url = f"https://arxiv.org/catchup/{category}/{date}"
        tasks.append(crawl4ai.fetch(url))

    return await asyncio.gather(*tasks)
```

### 2. Parser Agent

- **Function**: Extract structured data from the HTML responses
- **Implementation Requirements**:
  - Parse paper titles, authors, abstracts, categories, and links
  - Handle HTML structure changes gracefully
  - Clean and normalize extracted text
  - Output a standardized JSON format for each paper

```python
def parse_arxiv_html(html_content):
    """Extract paper details from arXiv HTML content"""
    # Implementation based on example_parse.js logic
    parsed_papers = []
    # Parsing logic here...
    return parsed_papers
```

### 3. Relevance Ranking Agent

- **Framework**: Pydantic AI agent with custom schema
- **Function**: Analyze papers and select the 5 most relevant to the user's context
- **Implementation Requirements**:
  - Define a clear relevance scoring algorithm
  - Consider user's role, goals, and interests
  - Provide justification for each selected paper
  - Output a ranked list with relevance scores and reasoning

```python
from pydantic_ai import Agent, Message, run_agent

class RelevanceRankingAgent(Agent):
    """Agent that ranks papers by relevance to user context"""

    async def rank_papers(self, papers, user_info):
        """Rank papers based on relevance to user"""
        prompt = Message(
            role="system",
            content=f"""You are an expert research curator.
            Rank the provided research papers by relevance to this user:
            Name: {user_info['name']}
            Title: {user_info['title']}
            Goals: {user_info['goals']}

            For each paper, provide:
            1. A relevance score (1-100)
            2. Clear reasoning for why this paper matters to the user
            3. How it might advance their stated goals

            Select only the 5 most relevant papers."""
        )

        # Add papers to the conversation
        paper_content = format_papers_for_prompt(papers)
        prompt.add(Message(role="user", content=paper_content))

        response = await run_agent(self, messages=[prompt])
        return parse_ranking_response(response)
```

### 4. Research Analysis Agent

- **Framework**: Pydantic AI agent
- **Function**: Generate in-depth summaries and analyses of each selected paper
- **Implementation Requirements**:
  - Process full paper content when available
  - Extract key innovations, methodologies, and findings
  - Identify practical applications relevant to the user's goals
  - Generate technical evaluations appropriate to user expertise level
  - Format output according to example_researcher_output.json

```python
class ResearcherAgent(Agent):
    """Agent that deeply analyzes research papers"""

    async def analyze_paper(self, paper, user_info):
        """Generate comprehensive analysis of a paper"""
        # Implementation details here
        return analysis_result
```

### 5. Newsletter Generation Agent

- **Framework**: Pydantic AI agent
- **Function**: Craft a polished, engaging newsletter from the research analyses
- **Implementation Requirements**:
  - Create personalized introduction and conclusion sections
  - Format paper summaries into cohesive sections
  - Add contextual connections between papers when relevant
  - Generate final markdown with proper styling
  - Include actionable insights based on user goals

```python
class CopywriterAgent(Agent):
    """Agent that formats research into engaging newsletters"""

    async def generate_newsletter(self, analyses, user_info):
        """Create formatted newsletter from paper analyses"""
        # Implementation details here
        return markdown_newsletter
```

## Technical Implementation Details

### Core Tech Stack

- **Pydantic AI**: Primary framework for agent definition and orchestration
- **Archon MCP**: Source of truth for Pydantic AI code and implementation
- **crawl4ai**: Web crawling and content extraction
- **asyncio**: Asynchronous processing for performance optimization
- **Python 3.9+**: Base language requirement

### Architecture Pattern

Implement a pipeline architecture where each agent is a separate module with:

- Clear input/output schemas using Pydantic models
- Comprehensive error handling and fallback mechanisms
- Configurable parameters for fine-tuning behavior
- Well-documented interfaces for maintainability

### Example Agent Schema Definition

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Paper(BaseModel):
    """Schema for a research paper"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    categories: List[str]
    publication_date: str

class PaperAnalysis(BaseModel):
    """Schema for paper analysis results"""
    paper: Paper
    relevance_score: int = Field(..., ge=1, le=100)
    relevance_reasoning: str
    key_innovations: List[str]
    methodology_summary: str
    practical_applications: List[str]
    user_relevance: str
```

## Output Format

The final output should be a professionally formatted markdown newsletter with:

1. Personalized greeting and introduction
2. Curated list of papers with:
   - Paper title (linked to original)
   - Author list
   - Publication date
   - Concise summary (2-3 paragraphs)
   - Key innovations
   - Why it matters to the user
3. Conclusion with action items or suggestions
4. Metadata for tracking and future improvements

## Implementation Approach

1. Start by setting up the Pydantic AI environment following Archon's guidance
2. Implement and test each agent individually
3. Create integration layer to connect agent workflow
4. Add user interface for input collection
5. Implement robust logging and error handling
6. Create unit and integration tests
7. Optimize for performance and resource usage

## Example Main Script Structure

```python
async def main():
    # 1. Collect user information
    user_info = collect_user_info()

    # 2. Fetch latest papers
    html_responses = await fetch_papers()

    # 3. Parse papers from HTML
    papers = [parse_arxiv_html(html) for html in html_responses]
    papers = [paper for sublist in papers for paper in sublist]  # Flatten

    # 4. Rank and select papers
    ranker = RelevanceRankingAgent()
    ranked_papers = await ranker.rank_papers(papers, user_info)

    # 5. Analyze selected papers
    researcher = ResearcherAgent()
    analyses = await asyncio.gather(*[
        researcher.analyze_paper(paper, user_info)
        for paper in ranked_papers[:5]
    ])

    # 6. Generate newsletter
    copywriter = CopywriterAgent()
    newsletter = await copywriter.generate_newsletter(analyses, user_info)

    # 7. Output or send the newsletter
    with open("research_newsletter.md", "w") as f:
        f.write(newsletter)

    print("Newsletter generated successfully!")

if __name__ == "__main__":
    asyncio.run(main())
```
