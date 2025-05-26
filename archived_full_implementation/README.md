# Archived Full Implementation

This directory contains the original full implementation of Paperboy that uses:
- crawl4ai with AsyncWebCrawler
- Playwright for browser automation
- Full scraping capabilities

These components were archived when we moved to the lightweight implementation
for better Cloud Run compatibility. They remain here for:
- Reference
- Potential future projects
- Rollback capability

## Directory Contents

### Core Implementation Files
- `src/fetcher.py` - Original fetcher using crawl4ai and AsyncWebCrawler
- `src/agent_tools.py` - Original agent tools using crawl4ai for scraping

### Requirements Files
- `requirements.txt` - Original requirements with crawl4ai[async,torch]
- `requirements.prod.txt` - Production requirements with crawl4ai
- `requirements.lock.txt` - Lock file with specific Playwright dependency versions

### Docker Files
- `docker-compose.yaml` - Original compose file for full implementation
- `Dockerfile.full` - Multi-stage Dockerfile with Playwright dependencies

### Other
- `browser_config/` - Directory for Playwright browser configuration
- `LIGHTWEIGHT_IMPLEMENTATION_PLAN.md` - The plan that led to the lightweight implementation

## To Use These Components

1. Copy the files back to their original locations:
   ```bash
   cp -r src/* ../src/
   cp requirements.txt ../
   cp docker-compose.yaml ../
   ```

2. Set the environment variable:
   ```bash
   export USE_LIGHTWEIGHT=false
   ```

3. Use the appropriate Dockerfile and requirements for deployment

## Why Archived?

The lightweight implementation was created to:
- Improve Cloud Run compatibility (no browser dependencies)
- Reduce container size and startup time
- Simplify deployment and maintenance
- Lower resource requirements

The full implementation remains valuable for scenarios requiring:
- JavaScript-heavy page rendering
- Complex browser interactions
- Full page screenshots
- Advanced scraping capabilities

Archived on: 2025-05-26