from typing import Dict, Any, Optional
from .models import DigestStatus
import asyncio
import logfire

class TaskStateManager:
    """Simple in-memory task state management."""

    def __init__(self):
        self.tasks: Dict[str, DigestStatus] = {}
        self._lock = asyncio.Lock()

    async def create_task(self, task_id: str, status: DigestStatus) -> None:
        """Create a new task with initial status."""
        async with self._lock:
            self.tasks[task_id] = status
            logfire.info(f"Task {task_id} created with status {status.status.value}")

    async def update_task(self, task_id: str, status: DigestStatus) -> None:
        """Update task status."""
        async with self._lock:
            self.tasks[task_id] = status
            logfire.info(f"Task {task_id} updated to status {status.status.value}")

    async def get_task(self, task_id: str) -> Optional[DigestStatus]:
        """Get task status."""
        async with self._lock:
            return self.tasks.get(task_id)

    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove tasks older than max_age_hours."""
        # Implementation for cleanup job
        # This would need task creation timestamps
        pass