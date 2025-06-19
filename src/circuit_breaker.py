"""Circuit breaker pattern implementation for external service calls."""
from datetime import datetime, timedelta, timezone
from enum import Enum
import asyncio
from typing import Optional, Callable, Any, Dict
import logfire
from supabase import Client


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"      # Failing, reject calls
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for protecting external service calls."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        supabase_client: Optional[Client] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Service name for identification
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            supabase_client: Optional Supabase client for distributed state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()
        self.supabase_client = supabase_client
        self.table = 'circuit_breaker_state'
        
        # Load state from Supabase if available
        if self.supabase_client:
            asyncio.create_task(self._load_state())
    
    async def _load_state(self) -> None:
        """Load circuit breaker state from Supabase."""
        try:
            response = self.supabase_client.table(self.table).select("*").eq('service_name', self.name).execute()
            
            if response.data and len(response.data) > 0:
                data = response.data[0]
                self.state = CircuitState(data['state'])
                self.failure_count = data.get('failure_count', 0)
                if data.get('last_failure_at'):
                    # Ensure timezone-aware datetime from Supabase
                    last_failure_str = data['last_failure_at']
                    if 'Z' in last_failure_str:
                        last_failure_str = last_failure_str.replace('Z', '+00:00')
                    self.last_failure_time = datetime.fromisoformat(last_failure_str)
                    # If still timezone-naive, assume UTC
                    if self.last_failure_time.tzinfo is None:
                        self.last_failure_time = self.last_failure_time.replace(tzinfo=timezone.utc)
                
                logfire.info("Loaded circuit breaker state for {name}: {state}", 
                           name=self.name, state=self.state.value)
        except Exception as e:
            logfire.error("Failed to load circuit breaker state: {error}", error=str(e))
    
    async def _save_state(self) -> None:
        """Save circuit breaker state to Supabase."""
        if not self.supabase_client:
            return
        
        try:
            data = {
                'service_name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'last_failure_at': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'last_success_at': datetime.now(timezone.utc).isoformat() if self.state == CircuitState.CLOSED else None
            }
            
            # Upsert state
            self.supabase_client.table(self.table).upsert(data).execute()
            
        except Exception as e:
            logfire.error("Failed to save circuit breaker state: {error}", error=str(e))
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logfire.info("Circuit breaker {name} entering HALF_OPEN state", name=self.name)
                else:
                    time_until_retry = self.recovery_timeout - (datetime.now(timezone.utc) - self.last_failure_time).seconds
                    raise CircuitOpenError(
                        f"Circuit breaker {self.name} is OPEN. Retry in {time_until_retry} seconds"
                    )
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Success - update state
            await self._on_success()
            return result
            
        except Exception as e:
            # Failure - update state
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            datetime.now(timezone.utc) - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logfire.info("Circuit breaker {name} recovered, now CLOSED", name=self.name)
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
            
            await self._save_state()
    
    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logfire.warn("Circuit breaker {name} opened after {count} failures", 
                           name=self.name, count=self.failure_count)
            
            await self._save_state()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        async with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout
            }
    
    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
            await self._save_state()
            logfire.info("Circuit breaker {name} manually reset", name=self.name)


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class ServiceCircuitBreakers:
    """Manage circuit breakers for all external services."""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        self.supabase_client = supabase_client
        self.breakers: Dict[str, CircuitBreaker] = {}
        
        # Initialize breakers for known services
        self._init_breakers()
    
    def _init_breakers(self) -> None:
        """Initialize circuit breakers for external services."""
        # OpenAI - higher threshold as it's critical
        self.breakers['openai'] = CircuitBreaker(
            'openai',
            failure_threshold=10,
            recovery_timeout=120,
            supabase_client=self.supabase_client
        )
        
        # NewsAPI - lower threshold
        self.breakers['newsapi'] = CircuitBreaker(
            'newsapi',
            failure_threshold=5,
            recovery_timeout=60,
            supabase_client=self.supabase_client
        )
        
        # Tavily - medium threshold
        self.breakers['tavily'] = CircuitBreaker(
            'tavily',
            failure_threshold=5,
            recovery_timeout=90,
            supabase_client=self.supabase_client
        )
        
        # ArXiv - higher threshold, longer recovery
        self.breakers['arxiv'] = CircuitBreaker(
            'arxiv',
            failure_threshold=3,
            recovery_timeout=180,
            supabase_client=self.supabase_client
        )
    
    def get(self, service_name: str) -> CircuitBreaker:
        """Get circuit breaker for a service."""
        if service_name not in self.breakers:
            # Create default breaker if not exists
            self.breakers[service_name] = CircuitBreaker(
                service_name,
                supabase_client=self.supabase_client
            )
        return self.breakers[service_name]
    
    async def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        status = {}
        for name, breaker in self.breakers.items():
            status[name] = await breaker.get_status()
        return status
