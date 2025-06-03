import asyncio
from typing import Any, Optional
from datetime import datetime, timedelta
import json
import hashlib

class SimpleCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._cache = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if datetime.now() < expiry:
                    return value
                else:
                    del self._cache[key]
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with TTL."""
        async with self._lock:
            expiry = datetime.now() + timedelta(seconds=self.ttl)
            self._cache[key] = (value, expiry)
    
    async def clear_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            now = datetime.now()
            expired = [k for k, (_, exp) in self._cache.items() if now >= exp]
            for key in expired:
                del self._cache[key]