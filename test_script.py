#!/usr/bin/env python3
# test_script.py

import asyncio
import json
import os
from dotenv import load_dotenv

from settings import AgentSettings
from state import AgentState
from agent import ArticleRankAgent

async def main():
    """Test the article ranking agent with sample data"""
    # Load environment variables
    load_dotenv()
    
    # Initialize settings and state
    settings = AgentSettings()
    state = AgentState()
    
    # Initialize agent
    agent = ArticleRankAgent(settings, state)
    
    # Sample user info
    user_info = {
        "name": "Dr. Jane Smith",
        "title": "Computer Science Professor",
        "goals": "I'm researching new approaches to natural language processing and looking for papers on transformer architectures and their applications."
    }
    
    # Sample articles with mock content
    articles = [
        {
            "title": "Advanced Transformer Architectures for NLP",
            "authors": ["John Doe", "Jane Smith"],
            "subject": "cs.CL",
            "abstract_url": "https://arxiv.org/abs/2403.12345",
            "html_url": "https://arxiv.org/html/2403.12345",
            "pdf_url": "https://arxiv.org/pdf/2403.12345.pdf",
            "body": """
            Title: Advanced Transformer Architectures for NLP
            Authors: John Doe, Jane Smith
            
            Abstract:
            This paper presents novel approaches to transformer architectures for natural language processing tasks.
            We introduce several innovations that improve performance while reducing computational complexity.
            
            1. Introduction
            Transformer architectures have revolutionized NLP since their introduction. This paper explores new
            variations that push the boundaries of what's possible with these models.
            
            2. Methods
            We present three key innovations:
            - Adaptive attention mechanisms
            - Sparse transformer layers
            - Dynamic pruning techniques
            
            3. Results
            Our experiments show significant improvements in both performance and efficiency.
            
            4. Conclusion
            The proposed architectures represent a significant step forward in NLP capabilities.
            """
        },
        {
            "title": "Reinforcement Learning for Robotics",
            "authors": ["Alice Johnson", "Bob Wilson"],
            "subject": "cs.RO",
            "abstract_url": "https://arxiv.org/abs/2403.12346",
            "html_url": "https://arxiv.org/html/2403.12346",
            "pdf_url": "https://arxiv.org/pdf/2403.12346.pdf",
            "body": """
            Title: Reinforcement Learning for Robotics
            Authors: Alice Johnson, Bob Wilson
            
            Abstract:
            This paper explores novel reinforcement learning approaches for robotic control systems.
            We demonstrate improved performance in complex manipulation tasks.
            
            1. Introduction
            Reinforcement learning has shown great promise in robotics, but challenges remain.
            
            2. Methods
            We introduce a new algorithm that combines model-based and model-free approaches.
            
            3. Results
            Our method achieves state-of-the-art performance on standard benchmarks.
            
            4. Conclusion
            The proposed approach opens new possibilities for robotic learning.
            """
        }
    ]
    
    print("Ranking articles...")
    ranked_articles = await agent.rank_articles(user_info, articles, top_n=2)
    
    if not ranked_articles:
        print("Failed to rank articles")
        return
    
    print(f"\nTop {len(ranked_articles)} ranked articles:")
    for i, article in enumerate(ranked_articles, 1):
        print(f"\n{i}. {article.title}")
        print(f"   Authors: {', '.join(article.authors)}")
        print(f"   Subject: {article.subject}")
        print(f"   Relevance Score: {article.relevance_score}/100")
        print(f"   Reasoning: {article.score_reason}")
    
    print("\nAnalyzing ranked articles...")
    analyses = await agent.analyze_ranked_articles(user_info, ranked_articles, top_n=2)
    
    if not analyses:
        print("Failed to analyze articles")
        return
    
    print(f"\nAnalyses for top {len(analyses)} articles:")
    for i, analysis in enumerate(analyses, 1):
        print(f"\n{i}. {analysis.title}")
        print(f"   Authors: {', '.join(analysis.authors)}")
        print(f"   Subject: {analysis.subject}")
        print(f"   Relevance Score: {analysis.relevance_score}/100")
        print(f"   Summary: {analysis.summary}")
        print(f"   Importance: {analysis.importance}")
        print(f"   Recommended Action: {analysis.recommended_action}")

if __name__ == "__main__":
    asyncio.run(main()) 