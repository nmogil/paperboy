"""Supabase-based state management for Paperboy."""
import os
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from supabase import create_client, Client
import json
import logfire

from .models import DigestStatus, TaskStatus, ArticleAnalysis


class TaskStateManager:
    """Task state manager using Supabase for persistence."""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        
        self.client = create_client(self.supabase_url, self.supabase_key)
        self.table = 'digest_tasks'
        
        logfire.info("TaskStateManager initialized with Supabase")
    
    async def create_task(self, task_id: str, status: DigestStatus, user_info: Dict[str, Any] = None) -> None:
        """Create a new task - simplified version"""
        data = {
            'task_id': task_id,
            'status': status.status.value,
            'message': status.message,
            'result': status.result,
            'user_info': user_info,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        self.client.table(self.table).insert(data).execute()
        logfire.info(f"Task {task_id} created")
    
    async def update_task(self, task_id: str, status: DigestStatus) -> None:
        """Update task - simplified version"""
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
        
        self.client.table(self.table).update(data).eq('task_id', task_id).execute()
        logfire.info(f"Task {task_id} updated to status {status.status.value}")
    
    async def get_task(self, task_id: str) -> Optional[DigestStatus]:
        """Get task status from Supabase."""
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
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove expired tasks."""
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        
        response = self.client.table(self.table).delete().lt('created_at', cutoff).execute()
        
        deleted_count = len(response.data) if response.data else 0
        logfire.info(f"Cleaned up {deleted_count} old tasks")
        
        return deleted_count
    
    async def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tasks for monitoring."""
        response = self.client.table(self.table).select("*").order('created_at', desc=True).limit(limit).execute()
        return response.data if response.data else []
    
    async def create_fetch_task(self, task_id: str, source_date: str, callback_url: str = None) -> None:
        """Create a new fetch task record."""
        data = {
            'task_id': task_id,
            'source_date': source_date,
            'status': 'pending',
            'callback_url': callback_url,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        self.client.table('fetch_tasks').insert(data).execute()
        logfire.info(f"Fetch task {task_id} created for date {source_date}")
    
    async def update_fetch_task(self, task_id: str, status: str, error: str = None, result: Dict[str, Any] = None) -> None:
        """Update fetch task status."""
        data = {
            'status': status,
            'updated_at': datetime.now().isoformat()
        }
        
        if error:
            data['error'] = error
        
        if result:
            data['result'] = result
        
        self.client.table('fetch_tasks').update(data).eq('task_id', task_id).execute()
        logfire.info(f"Fetch task {task_id} updated to status {status}")
    
    async def get_fetch_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get fetch task status from Supabase."""
        response = self.client.table('fetch_tasks').select("*").eq('task_id', task_id).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        
        return None
    
    async def update_task_with_source_date(self, task_id: str, status: DigestStatus, source_date: str = None, digest_type: str = None) -> None:
        """Update task with source_date and digest_type fields."""
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
        
        if source_date:
            data['source_date'] = source_date
        
        if digest_type:
            data['digest_type'] = digest_type
        
        self.client.table(self.table).update(data).eq('task_id', task_id).execute()
        logfire.info(f"Task {task_id} updated to status {status.status.value} with source_date {source_date}")
    
    async def create_task_with_source_date(self, task_id: str, status: DigestStatus, user_info: Dict[str, Any] = None, source_date: str = None, digest_type: str = None) -> None:
        """Create a new task with source_date and digest_type fields."""
        data = {
            'task_id': task_id,
            'status': status.status.value,
            'message': status.message,
            'result': status.result,
            'user_info': user_info,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        if source_date:
            data['source_date'] = source_date
        
        if digest_type:
            data['digest_type'] = digest_type
        
        self.client.table(self.table).insert(data).execute()
        logfire.info(f"Task {task_id} created with source_date {source_date} and digest_type {digest_type}")
