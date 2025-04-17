import pytest
import json
from agent import ArticleRankAgent, RankedArticle
from settings import AgentSettings
from state import AgentState
from unittest.mock import AsyncMock, patch

class MockAgent:
    async def run(self, prompt):
        return [
            {
                "title": "Good Article",
                "authors": ["Author 1", "Author 2"],
                "subject": "cs.AI",
                "score_reason": "Highly relevant to AI research",
                "relevance_score": 90,
                "abstract_url": "https://arxiv.org/abs/1234.5678",
                "html_url": "https://arxiv.org/html/1234.5678",
                "pdf_url": "https://arxiv.org/pdf/1234.5678.pdf"
            },
            {
                "title": "Neutral Article",
                "authors": ["Author 4"],
                "subject": "cs.CL",
                "score_reason": "Somewhat relevant to NLP research",
                "relevance_score": 70,
                "abstract_url": "https://arxiv.org/abs/1234.5680",
                "html_url": "https://arxiv.org/html/1234.5680",
                "pdf_url": "https://arxiv.org/pdf/1234.5680.pdf"
            }
        ]

class MockOpenAIModel:
    async def run(self, prompt):
        # Mock response for analyze_articles
        return {
            "data": json.dumps({
                "summary": "Test summary",
                "importance": "Test importance",
                "recommended_action": "Test action"
            })
        }

@pytest.fixture
def agent_settings():
    """Fixture to provide test settings"""
    return AgentSettings(
        openai_api_key="test_key",
        openai_model="gpt-4.1-mini-2025-04-14",
        max_articles=10,
        max_concurrent_scrapes=2,
        log_level="DEBUG"
    )

@pytest.fixture
def agent_state():
    """Fixture to provide test state"""
    return AgentState("test_state.json")

@pytest.fixture
def article_rank_agent(agent_settings, agent_state):
    """Fixture to provide an initialized ArticleRankAgent with mocked OpenAI model"""
    agent = ArticleRankAgent(agent_settings, agent_state)
    agent.llm = MockOpenAIModel()
    agent.agent = MockAgent()
    return agent

@pytest.mark.asyncio
async def test_rank_articles(article_rank_agent):
    """Test the rank_articles method"""
    user_info = {
        "name": "Test User",
        "title": "Researcher",
        "goals": "AI and Machine Learning"
    }
    
    articles = [
        {
            "title": "Good Article",
            "authors": ["Author 1", "Author 2"],
            "subject": "cs.AI",
            "abstract_url": "https://arxiv.org/abs/1234.5678",
            "html_url": "https://arxiv.org/html/1234.5678",
            "pdf_url": "https://arxiv.org/pdf/1234.5678.pdf",
            "body": "This is excellent content about AI."
        },
        {
            "title": "Bad Article",
            "authors": ["Author 3"],
            "subject": "cs.RO",
            "abstract_url": "https://arxiv.org/abs/1234.5679",
            "html_url": "https://arxiv.org/html/1234.5679",
            "pdf_url": "https://arxiv.org/pdf/1234.5679.pdf",
            "body": "This is poor content about robotics."
        },
        {
            "title": "Neutral Article",
            "authors": ["Author 4"],
            "subject": "cs.CL",
            "abstract_url": "https://arxiv.org/abs/1234.5680",
            "html_url": "https://arxiv.org/html/1234.5680",
            "pdf_url": "https://arxiv.org/pdf/1234.5680.pdf",
            "body": "This is okay content about NLP."
        }
    ]

    ranked = await article_rank_agent.rank_articles(user_info, articles, top_n=2)

    assert len(ranked) == 2
    assert ranked[0].relevance_score >= ranked[1].relevance_score
    assert all(hasattr(article, 'title') for article in ranked)
    assert all(hasattr(article, 'authors') for article in ranked)
    assert all(hasattr(article, 'subject') for article in ranked)
    assert all(hasattr(article, 'score_reason') for article in ranked)
    assert all(hasattr(article, 'relevance_score') for article in ranked)
    assert all(hasattr(article, 'abstract_url') for article in ranked)
    assert all(hasattr(article, 'html_url') for article in ranked)
    assert all(hasattr(article, 'pdf_url') for article in ranked)

@pytest.mark.asyncio
async def test_analyze_ranked_articles(article_rank_agent):
    """Test the analyze_ranked_articles method"""
    user_info = {
        "name": "Test User",
        "title": "Researcher",
        "goals": "AI and Machine Learning"
    }
    
    ranked_articles = [
        RankedArticle(
            title="Test Article 1",
            authors=["Author 1"],
            subject="cs.AI",
            score_reason="Highly relevant to AI research",
            relevance_score=90,
            abstract_url="https://arxiv.org/abs/1234.5678",
            html_url="https://arxiv.org/html/1234.5678",
            pdf_url="https://arxiv.org/pdf/1234.5678.pdf"
        ),
        RankedArticle(
            title="Test Article 2",
            authors=["Author 2"],
            subject="cs.AI",
            score_reason="Somewhat relevant to AI research",
            relevance_score=70,
            abstract_url="https://arxiv.org/abs/1234.5679",
            html_url="https://arxiv.org/html/1234.5679",
            pdf_url="https://arxiv.org/pdf/1234.5679.pdf"
        )
    ]

    # Mock the state to return original articles
    article_rank_agent.state.get_last_processed = lambda: {
        "original_articles": [
            {
                "title": "Test Article 1",
                "body": "Test content 1"
            },
            {
                "title": "Test Article 2",
                "body": "Test content 2"
            }
        ]
    }

    analyses = await article_rank_agent.analyze_ranked_articles(user_info, ranked_articles, top_n=2)

    assert len(analyses) == 2
    assert all(hasattr(analysis, 'title') for analysis in analyses)
    assert all(hasattr(analysis, 'authors') for analysis in analyses)
    assert all(hasattr(analysis, 'subject') for analysis in analyses)
    assert all(hasattr(analysis, 'summary') for analysis in analyses)
    assert all(hasattr(analysis, 'importance') for analysis in analyses)
    assert all(hasattr(analysis, 'recommended_action') for analysis in analyses)
    assert all(hasattr(analysis, 'abstract_url') for analysis in analyses)
    assert all(hasattr(analysis, 'html_url') for analysis in analyses)
    assert all(hasattr(analysis, 'pdf_url') for analysis in analyses)
    assert all(hasattr(analysis, 'relevance_score') for analysis in analyses)
    assert all(hasattr(analysis, 'score_reason') for analysis in analyses) 