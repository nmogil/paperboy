# state.py

import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("agent_state")

class AgentStateModel(BaseModel):
    """Model for agent state"""
    last_processed_articles: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    session_data: Dict[str, Any] = Field(default_factory=dict)

class AgentState:
    """Handler for agent state persistence"""
    
    def __init__(self, state_file: str = "agent_state.json"):
        self.state_file = state_file
        self.state = AgentStateModel()
        self.load()
    
    def save(self) -> None:
        """Save current state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state.model_dump(), f, indent=2)
            logger.info(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load(self) -> None:
        """Load state from file"""
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.state = AgentStateModel.model_validate(data)
            logger.info(f"State loaded from {self.state_file}")
        except FileNotFoundError:
            logger.info(f"No state file found at {self.state_file}, using defaults")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def update_last_processed(self, articles: Dict[str, Any]) -> None:
        """Update last processed articles"""
        self.state.last_processed_articles = articles
        self.save()
    
    def update_user_preferences(self, preferences: Dict[str, Any]) -> None:
        """Update user preferences"""
        self.state.user_preferences = preferences
        self.save()
    
    def get_last_processed(self) -> Dict[str, Any]:
        """Get last processed articles"""
        return self.state.last_processed_articles
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences"""
        return self.state.user_preferences
    
    def reset(self) -> None:
        """Reset state to defaults"""
        self.state = AgentStateModel()
        self.save() 