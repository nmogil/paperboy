# Cloud Run Reliability Improvements for Paperboy

## Overview

This document outlines multiple approaches to improve Paperboy's reliability on Google Cloud Run, addressing key issues identified from Cloud Run best practices and current architecture limitations.

## Key Issues and Solutions

### 1. In-Memory State Management

**Current Issue**: TaskStateManager and SimpleCache use in-memory storage, limiting concurrency to 1 and causing data loss on instance restarts.

#### Option A: Supabase (Recommended - Already in Use!)
```python
# Requirements: supabase-py

# src/state_supabase.py
from supabase import create_client, Client
from typing import Optional
from datetime import datetime, timedelta
from .models import DigestStatus, TaskStatus
import json

class SupabaseTaskStateManager:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.table = 'digest_tasks'
        
    async def create_table_if_not_exists(self):
        """Create the tasks table if it doesn't exist."""
        # This would typically be done via Supabase dashboard or migration
        # SQL for reference:
        """
        CREATE TABLE IF NOT EXISTS digest_tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            message TEXT,
            result JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '24 hours',
            user_info JSONB
        );
        
        -- Add index for cleanup queries
        CREATE INDEX idx_expires_at ON digest_tasks(expires_at);
        """
    
    async def create_task(self, task_id: str, status: DigestStatus) -> None:
        data = {
            'task_id': task_id,
            'status': status.status.value,
            'message': status.message,
            'result': status.result,
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        response = self.client.table(self.table).insert(data).execute()
        if not response.data:
            raise Exception("Failed to create task")
    
    async def update_task(self, task_id: str, status: DigestStatus) -> None:
        data = {
            'status': status.status.value,
            'message': status.message,
            'result': status.result,
            'updated_at': datetime.now().isoformat()
        }
        
        response = self.client.table(self.table).update(data).eq('task_id', task_id).execute()
        if not response.data:
            raise Exception("Failed to update task")
    
    async def get_task(self, task_id: str) -> Optional[DigestStatus]:
        response = self.client.table(self.table).select("*").eq('task_id', task_id).execute()
        
        if response.data and len(response.data) > 0:
            data = response.data[0]
            return DigestStatus(
                status=TaskStatus(data['status']),
                message=data.get('message'),
                result=data.get('result')
            )
        return None
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Remove expired tasks."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        response = self.client.table(self.table).delete().lt('created_at', cutoff.isoformat()).execute()
        return len(response.data) if response.data else 0
```

**Pros**: 
- Already integrated in your stack
- PostgreSQL-based with full ACID compliance
- Built-in REST API
- Real-time subscriptions available
- Row Level Security (RLS) for multi-tenancy
- Automatic backups
- No additional infrastructure needed

**Cons**: 
- Network latency (15-30ms typical)
- Rate limits on free tier
- Requires proper connection management

#### Option B: Redis (Alternative for High-Performance Caching)
```python
# Requirements: redis, aioredis

# src/state_redis.py
import aioredis
import json
from typing import Optional
from .models import DigestStatus

class RedisTaskStateManager:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None
    
    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def create_task(self, task_id: str, status: DigestStatus) -> None:
        await self.redis.setex(
            f"task:{task_id}",
            86400,  # 24 hour TTL
            status.json()
        )
    
    async def get_task(self, task_id: str) -> Optional[DigestStatus]:
        data = await self.redis.get(f"task:{task_id}")
        return DigestStatus.parse_raw(data) if data else None
```

**Pros**: 
- Fully managed on GCP (Memorystore)
- Sub-millisecond latency
- Built-in TTL support
- Supports high concurrency

**Cons**: 
- Additional service cost
- Requires VPC connector for Cloud Run

#### Option B: Cloud Firestore
```python
# Requirements: google-cloud-firestore

# src/state_firestore.py
from google.cloud import firestore
from typing import Optional
from .models import DigestStatus

class FirestoreTaskStateManager:
    def __init__(self):
        self.db = firestore.AsyncClient()
        self.collection = self.db.collection('tasks')
    
    async def create_task(self, task_id: str, status: DigestStatus) -> None:
        await self.collection.document(task_id).set({
            'status': status.status.value,
            'message': status.message,
            'result': status.result,
            'created_at': firestore.SERVER_TIMESTAMP,
            'ttl': firestore.SERVER_TIMESTAMP + 86400  # 24 hours
        })
    
    async def get_task(self, task_id: str) -> Optional[DigestStatus]:
        doc = await self.collection.document(task_id).get()
        if doc.exists:
            data = doc.to_dict()
            return DigestStatus(
                status=TaskStatus(data['status']),
                message=data.get('message'),
                result=data.get('result')
            )
        return None
```

**Pros**: 
- Serverless, no infrastructure management
- Native GCP integration
- Real-time updates possible
- Good for document-style data

**Cons**: 
- Higher latency than Redis (5-10ms)
- More complex queries

#### Option C: Cloud SQL (PostgreSQL)
```python
# Requirements: asyncpg, sqlalchemy

# src/state_sql.py
import asyncpg
from typing import Optional
from .models import DigestStatus

class PostgresTaskStateManager:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def connect(self):
        self.pool = await asyncpg.create_pool(self.connection_string)
        # Create table if not exists
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    message TEXT,
                    result JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP + INTERVAL '24 hours'
                )
            ''')
    
    async def create_task(self, task_id: str, status: DigestStatus) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                '''INSERT INTO tasks (task_id, status, message, result) 
                   VALUES ($1, $2, $3, $4)''',
                task_id, status.status.value, status.message, status.result
            )
```

**Pros**: 
- Full ACID compliance
- Complex queries possible
- Can store additional analytics
- Good for reporting

**Cons**: 
- Requires connection pooling management
- Higher operational overhead
- Need VPC connector

### 2. Cache Layer Improvements

#### Option A: Supabase Cache Table
```python
# src/cache_supabase.py
from supabase import Client
from typing import Any, Optional
from datetime import datetime, timedelta
import json
import hashlib

class SupabaseCache:
    """Supabase-backed cache with TTL."""
    
    def __init__(self, client: Client, ttl: int = 3600):
        self.client = client
        self.ttl = ttl
        self.table = 'cache_entries'
        # Keep small in-memory cache for hot data
        self.memory_cache = {}
        
    async def create_table_if_not_exists(self):
        """Create cache table schema."""
        # SQL for reference:
        """
        CREATE TABLE IF NOT EXISTS cache_entries (
            cache_key TEXT PRIMARY KEY,
            cache_value JSONB NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Index for cleanup
        CREATE INDEX idx_cache_expires ON cache_entries(expires_at);
        
        -- Optional: Automatic cleanup via Supabase Edge Function
        """
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        # Check memory cache first
        if key in self.memory_cache:
            value, expiry = self.memory_cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self.memory_cache[key]
        
        # Check Supabase
        response = self.client.table(self.table).select("*").eq('cache_key', key).execute()
        
        if response.data and len(response.data) > 0:
            entry = response.data[0]
            expires_at = datetime.fromisoformat(entry['expires_at'])
            
            if datetime.now() < expires_at:
                value = entry['cache_value']
                # Store in memory cache
                self.memory_cache[key] = (value, expires_at)
                return value
            else:
                # Expired, delete it
                self.client.table(self.table).delete().eq('cache_key', key).execute()
        
        return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with TTL."""
        expires_at = datetime.now() + timedelta(seconds=self.ttl)
        
        data = {
            'cache_key': key,
            'cache_value': value,
            'expires_at': expires_at.isoformat()
        }
        
        # Upsert to handle existing keys
        self.client.table(self.table).upsert(data).execute()
        
        # Also store in memory cache
        self.memory_cache[key] = (value, expires_at)
    
    async def clear_expired(self) -> int:
        """Remove expired entries."""
        response = self.client.table(self.table).delete().lt('expires_at', datetime.now().isoformat()).execute()
        return len(response.data) if response.data else 0
```

**Pros**:
- Uses existing Supabase connection
- No additional infrastructure
- Can leverage Supabase Edge Functions for auto-cleanup
- Persistent cache survives container restarts

**Cons**:
- Higher latency than in-memory (15-30ms)
- Counts against database operations quota

#### Option B: Redis with Fallback
```python
# src/cache_redis.py
class RedisCache:
    def __init__(self, redis_url: str, local_cache_size: int = 100):
        self.redis = aioredis.from_url(redis_url)
        # Keep small local LRU cache for hot data
        self.local_cache = LRUCache(maxsize=local_cache_size)
    
    async def get(self, key: str) -> Optional[Any]:
        # Check local first
        if key in self.local_cache:
            return self.local_cache[key]
        
        # Then Redis
        value = await self.redis.get(key)
        if value:
            parsed = json.loads(value)
            self.local_cache[key] = parsed
            return parsed
        return None
```

#### Option B: Hybrid Approach
```python
# Use in-memory for short TTL, external for long TTL
class HybridCache:
    def __init__(self, redis_url: str):
        self.memory_cache = SimpleCache(ttl=300)  # 5 min
        self.redis_cache = RedisCache(redis_url)
    
    async def get(self, key: str, use_persistent: bool = False):
        if not use_persistent:
            return await self.memory_cache.get(key)
        return await self.redis_cache.get(key)
```

### 3. Graceful Shutdown Handling

```python
# src/main.py modifications
import signal
import sys

class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.active_requests = 0
        self.lock = asyncio.Lock()
    
    async def increment(self):
        async with self.lock:
            self.active_requests += 1
    
    async def decrement(self):
        async with self.lock:
            self.active_requests -= 1
            if self.active_requests == 0 and self.shutdown_event.is_set():
                self.notify_complete()
    
    async def wait_for_shutdown(self):
        await self.shutdown_event.wait()
        # Wait for active requests to complete (max 30s)
        for _ in range(300):
            async with self.lock:
                if self.active_requests == 0:
                    break
            await asyncio.sleep(0.1)

# In lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    shutdown_handler = GracefulShutdown()
    app.state.shutdown = shutdown_handler
    
    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig, 
            lambda: asyncio.create_task(shutdown_handler.shutdown_event.set())
        )
    
    yield
    
    # Shutdown
    await shutdown_handler.wait_for_shutdown()
    # Close connections
    await app.state.digest_service.cleanup()
```

### 4. Improved Error Handling and Circuit Breakers

```python
# src/circuit_breaker.py
from datetime import datetime, timedelta
import asyncio

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        async with self.lock:
            if self.state == "OPEN":
                if datetime.now() - self.last_failure > timedelta(seconds=self.timeout):
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            async with self.lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            async with self.lock:
                self.failure_count += 1
                self.last_failure = datetime.now()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
            raise

# Usage in services
class DigestService:
    def __init__(self):
        self.openai_breaker = CircuitBreaker()
        self.news_breaker = CircuitBreaker()
    
    async def fetch_news(self, queries):
        try:
            return await self.news_breaker.call(
                self.news_fetcher.fetch_news, queries
            )
        except Exception as e:
            logger.error(f"News fetch failed (circuit state: {self.news_breaker.state})")
            return []  # Graceful degradation
```

### 5. Startup Optimization

```python
# src/main.py
# Lazy loading approach
class LazyDigestService:
    def __init__(self):
        self._service = None
        self._lock = asyncio.Lock()
    
    async def get_service(self):
        if self._service is None:
            async with self._lock:
                if self._service is None:
                    self._service = DigestService()
                    await self._service.initialize()
        return self._service

# Global initialization of expensive objects
# These persist across requests
OPENAI_CLIENT = None
HTTPX_CLIENT = None

async def get_openai_client():
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        OPENAI_CLIENT = AsyncOpenAI(api_key=settings.openai_api_key)
    return OPENAI_CLIENT

# Startup CPU boost utilization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm connections during startup
    await get_openai_client()
    
    # Pre-compile regex patterns
    app.state.patterns = {
        'arxiv': re.compile(r'arxiv\.org/abs/(\d+\.\d+)'),
        'date': re.compile(r'\d{4}-\d{2}-\d{2}')
    }
    
    yield
```

### 6. Request-Scoped Logging and Tracing

```python
# src/middleware.py
from contextvars import ContextVar
import uuid

request_id_var: ContextVar[str] = ContextVar('request_id')

class RequestTracingMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            request_id = str(uuid.uuid4())
            request_id_var.set(request_id)
            
            # Add to response headers
            async def send_wrapper(message):
                if message['type'] == 'http.response.start':
                    headers = MutableHeaders(message)
                    headers['X-Request-ID'] = request_id
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# Structured logging
import structlog

def setup_logging():
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory()
    )

logger = structlog.get_logger()

# Usage
async def generate_digest(...):
    logger.info("digest.started", 
                task_id=task_id, 
                user_name=user_info['name'])
```

## Implementation Priorities

### Phase 1: Critical (Immediate)
1. **External State Storage**: Implement Redis or Firestore for TaskStateManager
2. **Graceful Shutdown**: Add proper SIGTERM handling
3. **Basic Circuit Breakers**: Protect external API calls

### Phase 2: Important (Week 1-2)
1. **Startup Optimization**: Lazy loading and connection pooling
2. **Improved Error Handling**: Comprehensive try-catch with fallbacks
3. **Request Tracing**: Add request IDs and structured logging

### Phase 3: Nice to Have (Month 1)
1. **Advanced Caching**: Implement hybrid cache strategy
2. **Metrics Dashboard**: Add Prometheus metrics
3. **Load Testing**: Validate improvements under load

## Configuration Changes

### Environment Variables
```bash
# New variables to add
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
ENABLE_CIRCUIT_BREAKER=true
MAX_CONCURRENCY=10  # Can increase from 1 with external state
STARTUP_TIMEOUT=30
SHUTDOWN_TIMEOUT=30
REQUEST_TIMEOUT=60
ENABLE_TRACING=true
USE_EXTERNAL_STATE=true  # Feature flag for gradual rollout
CACHE_STRATEGY=hybrid  # memory, supabase, or hybrid
```

### Cloud Run Configuration
```yaml
# cloudbuild.yaml updates
- name: 'gcr.io/cloud-builders/gcloud'
  args:
    - 'run'
    - 'deploy'
    - 'paperboy'
    - '--concurrency=10'  # Increased from 1
    - '--min-instances=1'  # Keep warm
    - '--max-instances=50'
    - '--cpu-boost'  # Enable startup CPU boost
    - '--service-account=paperboy-sa@$PROJECT_ID.iam.gserviceaccount.com'
    # No VPC connector needed for Supabase (uses public internet)
```

## Testing Strategy

### Unit Tests
```python
# tests/test_state_external.py
@pytest.mark.asyncio
async def test_redis_state_manager():
    # Test with mock Redis
    manager = RedisTaskStateManager("redis://mock")
    await manager.create_task("test-id", DigestStatus(...))
    result = await manager.get_task("test-id")
    assert result is not None

# tests/test_circuit_breaker.py
@pytest.mark.asyncio
async def test_circuit_breaker_opens():
    breaker = CircuitBreaker(failure_threshold=2)
    
    async def failing_func():
        raise Exception("Service down")
    
    # First failures
    with pytest.raises(Exception):
        await breaker.call(failing_func)
    with pytest.raises(Exception):
        await breaker.call(failing_func)
    
    # Circuit should be open
    assert breaker.state == "OPEN"
```

### Load Testing
```bash
# Using hey or ab
hey -n 1000 -c 10 -m POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_info": {...}}' \
  https://paperboy-xxx.run.app/generate-digest
```

## Monitoring and Alerts

### Key Metrics
- Request latency (p50, p95, p99)
- Circuit breaker state changes
- Cache hit/miss ratios
- External API call durations
- Task completion rates
- Instance startup times

### Alert Conditions
- Circuit breaker open for > 5 minutes
- Task failure rate > 10%
- Request latency p95 > 10s
- Redis connection failures
- Memory usage > 80%

## Rollback Strategy

1. Keep current in-memory implementation as fallback
2. Use feature flags for gradual rollout
3. Monitor error rates during deployment
4. Automated rollback on error threshold

## Cost Implications

### Supabase
- Free tier: 500MB database, 2GB bandwidth, 50K requests/month
- Pro tier: $25/month - 8GB database, 50GB bandwidth, unlimited requests
- Already in use, so no additional cost for state management

### Redis (Memorystore)
- Basic tier: ~$35/month for 1GB
- Standard tier: ~$150/month for 1GB HA

### Firestore
- Reads: $0.06 per 100,000
- Writes: $0.18 per 100,000
- Storage: $0.18/GB/month

### Cloud SQL
- db-f1-micro: ~$15/month
- db-n1-standard-1: ~$50/month

## Decision Matrix

| Solution | Latency | Cost | Complexity | Reliability | Scalability | Integration |
|----------|---------|------|------------|-------------|-------------|-------------|
| Supabase | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Redis | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Firestore | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Cloud SQL | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

## Recommended Approach

1. **Use Supabase** for state management (already integrated, no additional cost)
2. **Implement hybrid caching** (memory + Supabase for persistence)
3. **Add circuit breakers** for all external services
4. **Implement graceful shutdown** handling
5. **Enable request tracing** with structured logging
6. **Increase concurrency to 10** once Supabase state is implemented
7. **Add connection pooling** for Supabase to optimize performance

## Supabase-Specific Optimizations

### 1. Connection Pooling
```python
# src/supabase_pool.py
from supabase import create_client, Client
import asyncio
from typing import Optional

class SupabaseConnectionPool:
    def __init__(self, url: str, key: str, pool_size: int = 5):
        self.url = url
        self.key = key
        self.pool = asyncio.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        
    async def initialize(self):
        """Pre-create connections."""
        for _ in range(self.pool_size):
            client = create_client(self.url, self.key)
            await self.pool.put(client)
    
    async def acquire(self) -> Client:
        """Get a client from the pool."""
        return await self.pool.get()
    
    async def release(self, client: Client):
        """Return a client to the pool."""
        await self.pool.put(client)
```

### 2. Batch Operations
```python
# Batch inserts/updates for better performance
async def batch_update_tasks(self, updates: List[Dict]):
    """Update multiple tasks in one request."""
    # Supabase supports bulk operations
    response = self.client.table('digest_tasks').upsert(updates).execute()
    return response.data
```

### 3. Real-time Subscriptions (Optional)
```python
# For real-time task status updates
def subscribe_to_task_updates(self, task_id: str, callback):
    """Subscribe to real-time changes for a specific task."""
    channel = self.client.channel(f'task:{task_id}')
    channel.on('postgres_changes', 
               event='UPDATE', 
               schema='public',
               table='digest_tasks',
               filter=f'task_id=eq.{task_id}').subscribe(callback)
    return channel
```

### 4. Database Schema Optimizations
```sql
-- Optimized schema for Paperboy
CREATE TABLE digest_tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL,
    message TEXT,
    result JSONB,
    user_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '24 hours'
);

-- Indexes for performance
CREATE INDEX idx_task_status ON digest_tasks(status);
CREATE INDEX idx_task_created ON digest_tasks(created_at DESC);
CREATE INDEX idx_task_expires ON digest_tasks(expires_at);

-- Automatic updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_digest_tasks_updated_at 
BEFORE UPDATE ON digest_tasks 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (optional)
ALTER TABLE digest_tasks ENABLE ROW LEVEL SECURITY;

-- Policy for API access
CREATE POLICY "API can manage all tasks" ON digest_tasks
    FOR ALL USING (true);
```

### 5. Edge Function for Cleanup
```typescript
// supabase/functions/cleanup-tasks/index.ts
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

serve(async (req) => {
  const supabaseClient = createClient(
    Deno.env.get('SUPABASE_URL') ?? '',
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
  )

  // Delete expired tasks
  const { data, error } = await supabaseClient
    .from('digest_tasks')
    .delete()
    .lt('expires_at', new Date().toISOString())

  return new Response(
    JSON.stringify({ deleted: data?.length || 0 }),
    { headers: { "Content-Type": "application/json" } }
  )
})

// Schedule this to run hourly via Supabase Cron
```