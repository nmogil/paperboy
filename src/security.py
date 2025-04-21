from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os

# Define the name of the header the client must send
API_KEY_NAME = "X-API-Key"

# Create the scheme instance
# auto_error=False allows us to provide custom error messages
api_key_header_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Retrieve the *actual* secret key from environment variables
# IMPORTANT: Never hardcode the secret key directly in the code.
VALID_API_KEY = os.getenv("API_KEY")

async def validate_api_key(api_key_header: str = Security(api_key_header_scheme)):
    """
    FastAPI dependency to validate the API key provided in the request header.

    Raises:
        HTTPException(403): If the API key is missing or invalid.
        HTTPException(500): If the server hasn't been configured with an API_KEY.
    """
    if not VALID_API_KEY:
        # Server-side configuration error
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
        # Key is valid, allow request to proceed
        return api_key_header # You can return the key or True, etc.
    else:
        # Key is invalid
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid API Key provided."
        ) 