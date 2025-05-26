# Lightweight Version Implementation Plan for Cloud Run

This document provides a step-by-step implementation plan for converting Paperboy to use the lightweight version (without Playwright) for Cloud Run deployment.

## Overview

The lightweight version replaces the resource-intensive Playwright/Chromium web scraping with a simpler HTTP-based approach using `httpx`. This significantly reduces container size, memory usage, and startup time, making it ideal for serverless deployments.

**Important Note**: The existing codebase uses `crawl4ai` which already includes Playwright. The lightweight version will need to completely replace `crawl4ai` with direct HTTP requests.

## Implementation Steps

### Step 1: Create Lightweight Fetcher Module

Create a new file `src/fetcher_lightweight.py` that replaces Crawl4AI/Playwright with httpx:

```python
# Key changes:
# - Replace AsyncWebCrawler with httpx.AsyncClient
# - Use BeautifulSoup for HTML parsing instead of browser rendering
# - Implement simple retry logic for failed requests
# - Remove all browser-specific configurations
```

**Tasks:**
1. Copy `src/fetcher.py` to `src/fetcher_lightweight.py`
2. Remove all Crawl4AI imports and dependencies
3. Replace with httpx and BeautifulSoup4
4. Implement `fetch_arxiv_page()` using httpx GET requests
5. Parse HTML content to extract article URLs
6. Maintain the same return format as the original fetcher

### Step 2: Create Lightweight Agent Tools

Create `src/agent_tools_lightweight.py` that uses httpx for article content extraction:

```python
# Key changes:
# - Replace browser-based scraping with direct HTTP requests
# - Use regex or BeautifulSoup for content extraction
# - Simplify the extraction logic for arXiv's predictable HTML structure
```

**Tasks:**
1. Copy `src/agent_tools.py` to `src/agent_tools_lightweight.py`
2. Replace `AsyncWebCrawler` with httpx client
3. Implement `extract_article_content()` using HTTP requests
4. Parse arXiv HTML to extract title, abstract, authors
5. Ensure compatibility with the agent's expected data format

### Step 3: Create Lightweight Requirements File

Create `requirements.lightweight.txt` with minimal dependencies:

```txt
# Core framework
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.6
pydantic-settings==2.7.1

# AI/ML
openai==1.59.0
pydantic-ai==0.0.24

# HTTP and parsing
httpx==0.28.1
beautifulsoup4==4.12.3
lxml==5.3.0

# Utilities
python-multipart==0.0.20
logfire==2.8.0
typing-extensions==4.12.2
```

**Tasks:**
1. Create the requirements file without Playwright/Crawl4AI
2. Add httpx and beautifulsoup4 for web scraping
3. Keep all other core dependencies
4. Verify no browser-related packages are included

### Step 4: Create Lightweight Dockerfile

Create `Dockerfile.lightweight` optimized for Cloud Run:

```dockerfile
# Key changes:
# - Single-stage build (no need for Playwright installation)
# - Smaller base image (python:3.10-slim)
# - No browser dependencies
# - Faster build and startup times
```

**Tasks:**
1. Create new Dockerfile based on the production one
2. Remove all Playwright installation steps
3. Remove browser dependency packages
4. Simplify to single-stage build
5. Maintain security features (non-root user, read-only filesystem)

### Step 5: Update Configuration for Lightweight Mode

Modify `src/config.py` to support lightweight mode selection:

```python
# Add configuration option
use_lightweight: bool = Field(default=True, env="USE_LIGHTWEIGHT")
```

**Tasks:**
1. Add `USE_LIGHTWEIGHT` environment variable
2. Set default to `True` for production
3. Update import logic to conditionally load lightweight modules

### Step 6: Update Main Application

Modify `src/main.py` to conditionally import lightweight modules:

```python
# Conditional imports based on configuration
if settings.use_lightweight:
    from src.fetcher_lightweight import fetch_arxiv_page
    from src.agent_tools_lightweight import extract_article_content
else:
    from src.fetcher import fetch_arxiv_page
    from src.agent_tools import extract_article_content
```

**Tasks:**
1. Add conditional import logic at the top of main.py
2. Ensure all function signatures remain compatible
3. Test that both modes work correctly
4. Update any direct module references in the code

### Step 7: Create Cloud Run Deployment Script

Create `deploy_cloudrun.sh` for easy deployment:

```bash
#!/bin/bash
# Build and deploy to Cloud Run
gcloud run deploy paperboy \
  --source . \
  --dockerfile Dockerfile.lightweight \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars USE_LIGHTWEIGHT=true
```

**Tasks:**
1. Create deployment script
2. Configure appropriate Cloud Run settings
3. Set memory limit to 1GB (reduced from 2GB)
4. Enable autoscaling with min instances = 0
5. Configure environment variables from Secret Manager

### Step 8: Update Docker Compose for Testing

Create `docker-compose.lightweight.yaml` for local testing:

```yaml
# Lightweight version for local testing
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.lightweight
    # ... rest of configuration
```

**Tasks:**
1. Create lightweight docker-compose file
2. Reference the lightweight Dockerfile
3. Set USE_LIGHTWEIGHT=true in environment
4. Reduce memory limits for testing
5. Maintain all security configurations

## Testing Plan

1. **Unit Tests**: Create tests for lightweight modules
   - Test httpx-based fetching
   - Test HTML parsing logic
   - Verify output format compatibility

2. **Integration Tests**: 
   - Test full workflow with lightweight modules
   - Compare outputs with original implementation
   - Verify performance improvements

3. **Load Tests**:
   - Measure memory usage reduction
   - Test concurrent request handling
   - Verify faster cold start times

## Success Criteria

- [ ] Container size reduced by >50% (target: <500MB)
- [ ] Memory usage reduced by >60% (target: <512MB)
- [ ] Cold start time <5 seconds
- [ ] All existing API endpoints work identically
- [ ] No functionality regression
- [ ] Successful deployment to Cloud Run

## Notes for Implementation

1. **Preserve API Compatibility**: The lightweight version must maintain 100% API compatibility
2. **Error Handling**: Implement robust retry logic for HTTP requests
3. **Rate Limiting**: Add exponential backoff for arXiv requests
4. **Logging**: Maintain detailed logging for debugging
5. **Fallback**: Consider implementing fallback to full version if lightweight fails

## Environment Variables for Lightweight Mode

```env
# Add to .env file
USE_LIGHTWEIGHT=true
HTTP_TIMEOUT=30
MAX_RETRIES=3
RETRY_DELAY=1
```

This implementation plan provides a clear path to create a production-ready lightweight version optimized for Cloud Run deployment.