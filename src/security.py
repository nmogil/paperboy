from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os

API_KEY_NAME = "X-API-Key"

api_key_header_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

VALID_API_KEY = os.getenv("API_KEY")

async def validate_api_key(api_key_header: str = Security(api_key_header_scheme)):
    """
    FastAPI dependency to validate the API key provided in the request header.

    Raises:
        HTTPException(403): If the API key is missing or invalid.
        HTTPException(500): If the server hasn't been configured with an API_KEY.
    """
    if not VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: API Key not configured."
        )

    if not api_key_header:
         raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated: API key header missing."
        )

    if api_key_header == VALID_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid API Key provided."
        ) 