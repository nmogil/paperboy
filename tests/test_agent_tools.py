import pytest
from src.agent_tools import analyze_article, ArticleAnalysisResult
from pydantic import BaseModel

class MockUserContext(BaseModel):
    name: str
    title: str
    goals: str

class MockAgent:
    async def run(self, prompt):
        return {
            "summary": "Test summary",
            "importance": "Test importance",
            "recommended_action": "Test action"
        }

@pytest.mark.asyncio
async def test_analyze_article_basic():
    """Test the analyze_article function with basic content"""
    ctx = MockAgent()
    user_context = MockUserContext(
        name="Test User",
        title="Researcher",
        goals="AI Research"
    )
    article_metadata = {
        "title": "Test Article",
        "authors": ["Test Author"],
        "subject": "cs.AI"
    }
    article_text = "This is a good article about AI."
    
    result = await analyze_article(ctx, article_text, user_context, article_metadata)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "importance" in result
    assert "recommended_action" in result

@pytest.mark.asyncio
async def test_analyze_article_empty():
    """Test the analyze_article function with empty content"""
    ctx = MockAgent()
    user_context = MockUserContext(
        name="Test User",
        title="Researcher",
        goals="AI Research"
    )
    article_metadata = {
        "title": "Empty Article",
        "authors": ["Test Author"],
        "subject": "cs.AI"
    }
    article_text = ""
    
    result = await analyze_article(ctx, article_text, user_context, article_metadata)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "importance" in result
    assert "recommended_action" in result

@pytest.mark.asyncio
async def test_analyze_article_missing_metadata():
    """Test the analyze_article function with missing metadata"""
    ctx = MockAgent()
    user_context = MockUserContext(
        name="Test User",
        title="Researcher",
        goals="AI Research"
    )
    article_metadata = {
        "title": "No Metadata",
        "authors": ["Unknown"],
        "subject": "Not specified"
    }
    article_text = "Some content"
    
    result = await analyze_article(ctx, article_text, user_context, article_metadata)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "importance" in result
    assert "recommended_action" in result

@pytest.mark.asyncio
async def test_analyze_article_long_content():
    """Test the analyze_article function with long content"""
    ctx = MockAgent()
    user_context = MockUserContext(
        name="Test User",
        title="Researcher",
        goals="AI Research"
    )
    article_metadata = {
        "title": "Long Article",
        "authors": ["Test Author"],
        "subject": "cs.AI"
    }
    article_text = "This is a very long article about artificial intelligence and machine learning. " * 10
    
    result = await analyze_article(ctx, article_text, user_context, article_metadata)

    assert isinstance(result, dict)
    assert "summary" in result
    assert "importance" in result
    assert "recommended_action" in result 