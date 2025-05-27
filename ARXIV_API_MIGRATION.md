# arXiv API Migration Guide

## Current Limitation

The arXiv API doesn't support querying by submission date in real-time. The daily "new" and "recent" listings on arxiv.org/catchup are generated internally and not available through the API. This is why we currently scrape the HTML catchup pages.

## Future API Implementation (When arXiv Adds Date Support)

If arXiv adds proper submission date querying to their API, here's how to migrate:

### Step 1: Create New API Fetcher

Create `src/fetcher_api.py`:

```python
"""
ArXiv API fetcher - More efficient than HTML scraping
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import httpx
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

async def fetch_arxiv_cs_submissions(target_date: str, client: Optional[httpx.AsyncClient] = None) -> List[Dict[str, Any]]:
    """
    Fetch CS submissions from arXiv API for a specific date.
    
    Note: This assumes arXiv adds proper date filtering to their API.
    Currently, the API doesn't support reliable submission date queries.
    """
    try:
        # Parse date
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        
        # Build query - THIS IS THE PART THAT DOESN'T WORK YET
        # Future API might support: submitted_date:2025-05-27
        query = f"cat:cs.* AND submitted_date:{target_date}"
        
        # API endpoint
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 200,  # Get all CS papers for the day
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        # Make request
        if client is None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as new_client:
                response = await new_client.get(base_url, params=params)
        else:
            response = await client.get(base_url, params=params)
        
        response.raise_for_status()
        
        # Parse XML response
        articles = parse_arxiv_xml(response.text)
        
        logger.info(f"Fetched {len(articles)} articles from arXiv API")
        return articles
        
    except Exception as e:
        logger.error(f"Error fetching from arXiv API: {str(e)}", exc_info=True)
        return []

def parse_arxiv_xml(xml_content: str) -> List[Dict[str, Any]]:
    """Parse arXiv API XML response."""
    root = ET.fromstring(xml_content)
    
    # Define namespaces
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    articles = []
    for entry in root.findall('atom:entry', ns):
        article = {}
        
        # Extract ID from URL
        id_elem = entry.find('atom:id', ns)
        if id_elem is not None:
            article['arxiv_id'] = id_elem.text.split('/')[-1]
        
        # Extract basic fields
        title = entry.find('atom:title', ns)
        article['title'] = title.text.strip() if title is not None else ""
        
        summary = entry.find('atom:summary', ns)
        article['abstract'] = summary.text.strip() if summary is not None else ""
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)
        article['authors'] = ', '.join(authors)
        
        # Extract categories
        categories = []
        for category in entry.findall('atom:category', ns):
            term = category.get('term')
            if term:
                categories.append(term)
        article['subjects'] = ', '.join(categories)
        
        # Extract URLs
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                article['pdf_url'] = link.get('href')
            elif link.get('type') == 'text/html':
                article['abstract_url'] = link.get('href')
        
        # Ensure URLs exist
        if 'arxiv_id' in article:
            article.setdefault('pdf_url', f"https://arxiv.org/pdf/{article['arxiv_id']}.pdf")
            article.setdefault('abstract_url', f"https://arxiv.org/abs/{article['arxiv_id']}")
        
        articles.append(article)
    
    return articles
```

### Step 2: Update Configuration

Add to `src/config.py`:

```python
# API vs HTML scraping
use_arxiv_api: bool = Field(default=False, validation_alias='USE_ARXIV_API', 
                           description="Use arXiv API instead of HTML scraping (when available)")
```

### Step 3: Update Main Module

In `src/main.py`, conditionally import:

```python
# Import based on configuration
if settings.use_arxiv_api:
    from .fetcher_api import fetch_arxiv_cs_submissions
else:
    from .fetcher_lightweight import fetch_arxiv_cs_submissions
```

### Step 4: Benefits of API Approach

1. **Performance**: ~100KB XML vs 3.1MB HTML
2. **Reliability**: Structured data, no HTML parsing
3. **Speed**: 10x faster response times
4. **Efficiency**: Lower memory usage
5. **Stability**: API format rarely changes

### Step 5: Testing Migration

```bash
# Test with API
export USE_ARXIV_API=true
python -m pytest tests/test_fetcher_api.py

# Compare results
python scripts/compare_fetchers.py --date=2025-05-27
```

## Alternative: OAI-PMH Protocol

arXiv also supports OAI-PMH (Open Archives Initiative Protocol for Metadata Harvesting):

```python
# Get records from specific date
url = "http://export.arxiv.org/oai2"
params = {
    "verb": "ListRecords",
    "from": "2025-05-27",
    "until": "2025-05-27",
    "metadataPrefix": "arXiv",
    "set": "cs"  # Computer Science
}
```

This might work better for date-based queries but requires different parsing logic.

## Current Workaround: Caching

Since arXiv updates once daily, implement caching:

```python
import hashlib
from google.cloud import storage

class ArxivCache:
    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket('paperboy-cache')
    
    async def get_or_fetch(self, date: str):
        cache_key = f"arxiv/cs/{date}.json"
        blob = self.bucket.blob(cache_key)
        
        if blob.exists():
            logger.info(f"Using cached arXiv data for {date}")
            return json.loads(blob.download_as_text())
        
        # Fetch fresh data
        articles = await fetch_arxiv_cs_submissions(date)
        
        # Cache for 24 hours
        blob.upload_from_string(
            json.dumps(articles),
            content_type='application/json'
        )
        
        return articles
```

## Monitoring for API Availability

Check these resources periodically:
- [arXiv API Documentation](https://arxiv.org/help/api)
- [arXiv API User Group](https://groups.google.com/g/arxiv-api)
- [arXiv Blog](https://blog.arxiv.org/) for announcements

When proper date filtering becomes available, the migration will be straightforward using the structure above.