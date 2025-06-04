"""Supabase-backed cache implementation."""
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from supabase import Client
import json
import hashlib
import logfire
from collections import OrderedDict

from .cache import SimpleCache


class SupabaseCache:
    """Supabase-backed cache with TTL and local memory cache for hot data."""
    
    def __init__(self, client: Client, ttl: int = 3600):
        self.client = client
        self.ttl = ttl
        self.table = 'cache_entries'
        
        # Keep small in-memory LRU cache for hot data
        self.memory_cache = LRUCache(maxsize=100)
        
        logfire.info("SupabaseCache initialized with TTL: {ttl}s", ttl=ttl)
    
    def _generate_key(self, key: str) -> str:
        """Generate a consistent hash key for long cache keys."""
        if len(key) > 255:  # PostgreSQL text limit considerations
            return hashlib.sha256(key.encode()).hexdigest()
        return key
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        hashed_key = self._generate_key(key)
        
        # Check memory cache first
        value = self.memory_cache.get(hashed_key)
        if value is not None:
            logfire.debug("Cache hit (memory): {key}", key=key[:50])
            return value
        
        try:
            # Check Supabase
            response = self.client.table(self.table).select("*").eq('cache_key', hashed_key).execute()
            
            if response.data and len(response.data) > 0:
                entry = response.data[0]
                expires_at = datetime.fromisoformat(entry['expires_at'].replace('Z', '+00:00'))
                
                if datetime.now(expires_at.tzinfo) < expires_at:
                    value = entry['cache_value']
                    # Store in memory cache
                    self.memory_cache.put(hashed_key, value)
                    logfire.debug("Cache hit (Supabase): {key}", key=key[:50])
                    return value
                else:
                    # Expired, delete it
                    self.client.table(self.table).delete().eq('cache_key', hashed_key).execute()
                    logfire.debug("Cache expired: {key}", key=key[:50])
            else:
                logfire.debug("Cache miss: {key}", key=key[:50])
            
            return None
            
        except Exception as e:
            logfire.error("Cache get error: {error}", error=str(e))
            # Fail open - don't break if cache is down
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with TTL."""
        hashed_key = self._generate_key(key)
        
        try:
            expires_at = datetime.now() + timedelta(seconds=self.ttl)
            
            data = {
                'cache_key': hashed_key,
                'cache_value': value,
                'expires_at': expires_at.isoformat()
            }
            
            # Upsert to handle existing keys
            self.client.table(self.table).upsert(data).execute()
            
            # Also store in memory cache
            self.memory_cache.put(hashed_key, value)
            
            logfire.debug("Cache set: {key}", key=key[:50])
            
        except Exception as e:
            logfire.error("Cache set error: {error}", error=str(e))
            # Fail open - don't break if cache is down
    
    async def clear_expired(self) -> int:
        """Remove expired entries."""
        try:
            response = self.client.table(self.table).delete().lt('expires_at', datetime.now().isoformat()).execute()
            count = len(response.data) if response.data else 0
            logfire.info("Cleared {count} expired cache entries", count=count)
            return count
        except Exception as e:
            logfire.error("Failed to clear expired cache: {error}", error=str(e))
            return 0


class LRUCache:
    """Simple LRU cache implementation for in-memory caching."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value and move to end (most recently used)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Add or update value."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # Remove oldest if over capacity
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


class HybridCache:
    """Hybrid cache using both in-memory and Supabase for different use cases."""
    
    def __init__(self, supabase_client: Client, memory_ttl: int = 300, persistent_ttl: int = 3600):
        """
        Initialize hybrid cache.
        
        Args:
            supabase_client: Supabase client instance
            memory_ttl: TTL for in-memory cache (default 5 minutes)
            persistent_ttl: TTL for Supabase cache (default 1 hour)
        """
        self.memory_cache = SimpleCache(ttl=memory_ttl)
        self.supabase_cache = SupabaseCache(supabase_client, ttl=persistent_ttl)
    
    async def get(self, key: str, use_persistent: bool = True) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            use_persistent: If True, check Supabase after memory cache
        """
        # Always check memory first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Check persistent if enabled
        if use_persistent:
            value = await self.supabase_cache.get(key)
            if value is not None:
                # Populate memory cache
                await self.memory_cache.set(key, value)
                return value
        
        return None
    
    async def set(self, key: str, value: Any, use_persistent: bool = True) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            use_persistent: If True, also store in Supabase
        """
        # Always set in memory
        await self.memory_cache.set(key, value)
        
        # Set in persistent if enabled
        if use_persistent:
            await self.supabase_cache.set(key, value)
    
    async def clear_expired(self) -> None:
        """Clear expired entries from both caches."""
        await self.memory_cache.clear_expired()
        await self.supabase_cache.clear_expired()
