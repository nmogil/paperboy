import time
from functools import wraps
import logfire

class NewsMetrics:
    """Track news fetching performance."""

    @staticmethod
    def track_api_call(api_name: str):
        """Decorator to track API call metrics."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                error = None

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = type(e).__name__
                    raise
                finally:
                    duration = time.time() - start_time
                    logfire.info(
                        f"{api_name} API call",
                        extra={
                            "duration": duration,
                            "success": error is None,
                            "error": error,
                            "api_name": api_name
                        }
                    )

            return wrapper
        return decorator

    @staticmethod
    def track_content_extraction(func):
        """Decorator specifically for tracking content extraction."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            articles_count = 0
            successful_extractions = 0
            error = None

            try:
                result = await func(*args, **kwargs)
                
                # Count articles and successful extractions
                if isinstance(result, list):
                    articles_count = len(result)
                    successful_extractions = sum(
                        1 for article in result 
                        if article.get('extraction_success', False)
                    )
                
                return result
            except Exception as e:
                error = type(e).__name__
                raise
            finally:
                duration = time.time() - start_time
                logfire.info(
                    "Content extraction completed",
                    extra={
                        "duration": duration,
                        "articles_processed": articles_count,
                        "successful_extractions": successful_extractions,
                        "success_rate": successful_extractions / articles_count if articles_count > 0 else 0,
                        "error": error
                    }
                )

        return wrapper

    @staticmethod
    def track_query_generation(func):
        """Decorator for tracking query generation performance."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            query_count = 0
            error = None

            try:
                result = await func(*args, **kwargs)
                
                if isinstance(result, list):
                    query_count = len(result)
                
                return result
            except Exception as e:
                error = type(e).__name__
                raise
            finally:
                duration = time.time() - start_time
                logfire.info(
                    "Query generation completed",
                    extra={
                        "duration": duration,
                        "queries_generated": query_count,
                        "error": error
                    }
                )

        return wrapper 