"""Graceful shutdown handling for Cloud Run."""
import asyncio
import signal
import sys
from typing import Set, Optional
import logfire
from contextlib import asynccontextmanager


class GracefulShutdown:
    """Handle graceful shutdown of the application."""
    
    def __init__(self, timeout: int = 30):
        """
        Initialize graceful shutdown handler.
        
        Args:
            timeout: Maximum seconds to wait for active requests to complete
        """
        self.shutdown_event = asyncio.Event()
        self.active_requests: Set[str] = set()
        self.lock = asyncio.Lock()
        self.timeout = timeout
        self._shutdown_tasks: list = []
        
        logfire.info("GracefulShutdown handler initialized with timeout: {timeout}s", timeout=timeout)
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        
        # Handle SIGTERM (Cloud Run shutdown signal) and SIGINT (Ctrl+C)
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._signal_handler(s))
            )
        
        logfire.info("Signal handlers registered for graceful shutdown")
    
    async def _signal_handler(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        logfire.info("Received signal {signal}, initiating graceful shutdown", signal=sig.name)
        self.shutdown_event.set()
    
    @asynccontextmanager
    async def track_request(self, request_id: str):
        """
        Context manager to track active requests.
        
        Args:
            request_id: Unique identifier for the request
        """
        async with self.lock:
            self.active_requests.add(request_id)
            logfire.debug("Request {id} started, active: {count}", 
                        id=request_id, count=len(self.active_requests))
        
        try:
            yield
        finally:
            async with self.lock:
                self.active_requests.discard(request_id)
                logfire.debug("Request {id} completed, active: {count}", 
                            id=request_id, count=len(self.active_requests))
    
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal and active requests to complete."""
        # Wait for shutdown signal
        await self.shutdown_event.wait()
        
        logfire.info("Shutdown initiated, waiting for {count} active requests", 
                   count=len(self.active_requests))
        
        # Wait for active requests with timeout
        start_time = asyncio.get_event_loop().time()
        
        while self.active_requests and (asyncio.get_event_loop().time() - start_time) < self.timeout:
            async with self.lock:
                if not self.active_requests:
                    break
                active_count = len(self.active_requests)
            
            logfire.info("Waiting for {count} active requests to complete", count=active_count)
            await asyncio.sleep(0.5)
        
        if self.active_requests:
            logfire.warn("Shutdown timeout reached with {count} active requests", 
                       count=len(self.active_requests))
        else:
            logfire.info("All requests completed, proceeding with shutdown")
        
        # Run shutdown tasks
        await self._run_shutdown_tasks()
    
    def register_shutdown_task(self, task) -> None:
        """Register a coroutine to run during shutdown."""
        self._shutdown_tasks.append(task)
    
    async def _run_shutdown_tasks(self) -> None:
        """Run all registered shutdown tasks."""
        if not self._shutdown_tasks:
            return
        
        logfire.info("Running {count} shutdown tasks", count=len(self._shutdown_tasks))
        
        # Run shutdown tasks with timeout
        tasks = [asyncio.create_task(task()) for task in self._shutdown_tasks]
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=10  # 10 second timeout for shutdown tasks
            )
        except asyncio.TimeoutError:
            logfire.error("Shutdown tasks timed out")
        
        logfire.info("Shutdown tasks completed")
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self.shutdown_event.is_set()


class RequestTracker:
    """Middleware to track requests for graceful shutdown."""
    
    def __init__(self, shutdown_handler: GracefulShutdown):
        self.shutdown_handler = shutdown_handler
    
    async def __call__(self, request, call_next):
        """Track request lifecycle."""
        import uuid
        request_id = str(uuid.uuid4())
        
        # Don't track health checks
        if request.url.path in ['/health', '/digest-status/health']:
            return await call_next(request)
        
        # Track request
        async with self.shutdown_handler.track_request(request_id):
            # Add request ID to headers
            response = await call_next(request)
            response.headers['X-Request-ID'] = request_id
            return response
