# state.py

import json
import logging
from typing import Dict, Any, Optional, List
from .models import AgentStateModel, RankedArticle, UserContext

logger = logging.getLogger("agent_state")

class AgentState:
    """Handler for agent state persistence using the central AgentStateModel"""
    
    def __init__(self, state_file: str = "data/agent_state.json"):
        self.state_file = state_file
        self.state: AgentStateModel = AgentStateModel()
        self.load()
    
    def save(self) -> None:
        """Save current state (AgentStateModel) to file"""
        try:
            import os
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            with open(self.state_file, 'w') as f:
                f.write(self.state.model_dump_json(indent=2))
            logger.info(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}", exc_info=True)
    
    def load(self) -> None:
        """Load state from file into AgentStateModel"""
        try:
            with open(self.state_file, 'r') as f:
                self.state = AgentStateModel.model_validate_json(f.read())
            logger.info(f"State loaded from {self.state_file}")
        except FileNotFoundError:
            logger.info(f"No state file found at {self.state_file}, using defaults. Will create on first save.")
        except Exception as e:
            logger.error(f"Failed to load or validate state from {self.state_file}: {e}", exc_info=True)
            self.state = AgentStateModel()
            logger.warning("Falling back to default agent state.")
    
    def update_last_processed(self, articles: Dict[str, RankedArticle]) -> None:
        """Update last processed articles (expects Dict[arxiv_id, RankedArticle])"""
        self.state.last_processed_articles = articles
        self.save()
    
    def update_user_preferences(self, preferences: Dict[str, UserContext]) -> None:
        """Update user preferences (expects Dict[user_id, UserContext])"""
        self.state.user_preferences = preferences
        self.save()

    def update_session_data(self, data: Dict[str, Any]) -> None:
        """Update generic session data"""
        self.state.session_data = data
        self.save()
    
    def get_last_processed(self) -> Dict[str, RankedArticle]:
        """Get last processed articles (returns Dict[arxiv_id, RankedArticle])"""
        return self.state.last_processed_articles
    
    def get_user_preferences(self) -> Dict[str, UserContext]:
        """Get user preferences (returns Dict[user_id, UserContext])"""
        return self.state.user_preferences

    def get_session_data(self) -> Dict[str, Any]:
        """Get generic session data"""
        return self.state.session_data
    
    def reset(self) -> None:
        """Reset state to defaults using AgentStateModel"""
        self.state = AgentStateModel()
        self.save() 