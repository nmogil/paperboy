"""Supabase-based state management for Paperboy."""
import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from supabase import create_client, Client
import json
import logfire
import asyncio

from .models import DigestStatus, TaskStatus, ArticleAnalysis


class SupabaseTaskStateManager:
    """Task state manager using Supabase for persistence."""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize Supabase client with connection pooling."""
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        self.table = 'digest_tasks'
        self._lock = asyncio.Lock()
        
        logfire.info("SupabaseTaskStateManager initialized")
    
    async def create_task(self, task_id: str, status: DigestStatus, user_info: Dict[str, Any] = None) -> None:
        """Create a new task with initial status."""
        try:
            data = {
                'task_id': task_id,
                'status': status.status.value,
                'message': status.message,
                'result': status.result,
                'user_info': user_info,
                'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
            }
            
            response = self.client.table(self.table).insert(data).execute()
            
            if not response.data:
                raise Exception("Failed to create task in Supabase")
            
            logfire.info(f"Task {task_id} created with status {status.status.value}")
            
        except Exception as e:
            logfire.error(f"Failed to create task {task_id}: {str(e)}")
            raise
    
    async def update_task(self, task_id: str, status: DigestStatus) -> None:
        """Update task status."""
        try:
            # Prepare articles data if present
            articles_data = None
            if status.articles:
                articles_data = [
                    article.model_dump(mode='json', exclude_none=True) 
                    for article in status.articles
                ]
            
            data = {
                'status': status.status.value,
                'message': status.message,
                'result': status.result,
                'articles': articles_data,
                'updated_at': datetime.now().isoformat()
            }
            
            response = self.client.table(self.table).update(data).eq('task_id', task_id).execute()
            
            if not response.data:
                logfire.warn(f"No task found with ID {task_id} to update")
                # Create the task if it doesn't exist
                await self.create_task(task_id, status)
            else:
                logfire.info(f"Task {task_id} updated to status {status.status.value}")
            
        except Exception as e:
            logfire.error(f"Failed to update task {task_id}: {str(e)}")
            raise
    
    async def get_task(self, task_id: str) -> Optional[DigestStatus]:
        """Get task status from Supabase."""
        try:
            response = self.client.table(self.table).select("*").eq('task_id', task_id).execute()
            
            if response.data and len(response.data) > 0:
                data = response.data[0]
                
                # Reconstruct articles if present
                articles = None
                if data.get('articles'):
                    articles = []
                    for article_data in data['articles']:
                        try:
                            articles.append(ArticleAnalysis(**article_data))
                        except Exception as e:
                            logfire.error(f"Failed to parse article: {e}")
                
                return DigestStatus(
                    status=TaskStatus(data['status']),
                    message=data.get('message', ''),
                    result=data.get('result'),
                    articles=articles
                )
            
            return None
            
        except Exception as e:
            logfire.error(f"Failed to get task {task_id}: {str(e)}")
            return None
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove expired tasks."""
        try:
            cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
            
            response = self.client.table(self.table).delete().lt('created_at', cutoff).execute()
            
            deleted_count = len(response.data) if response.data else 0
            logfire.info(f"Cleaned up {deleted_count} old tasks")
            
            return deleted_count
            
        except Exception as e:
            logfire.error(f"Failed to cleanup old tasks: {str(e)}")
            return 0
    
    async def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tasks for monitoring."""
        try:
            response = self.client.table(self.table).select("*").order('created_at', desc=True).limit(limit).execute()
            return response.data if response.data else []
        except Exception as e:
            logfire.error(f"Failed to get recent tasks: {str(e)}")
            return []
