# Paperboy: AI-Powered Research Digest System

## Overview

Paperboy is an intelligent research assistant that automatically curates personalized digests of academic papers and industry news. The system leverages OpenAI's language models to rank and analyze content from arXiv (academic papers) and NewsAPI (industry news) based on individual researcher profiles, delivering tailored insights through beautifully formatted HTML digests.

The system includes enhanced reliability features including circuit breakers, Supabase integration for distributed state management, graceful shutdown handling, and comprehensive error handling to ensure robust operation in production environments.

## Core Purpose

The system addresses the information overload problem faced by researchers and technical professionals by:

- Automatically fetching the latest academic papers from arXiv
- Discovering relevant industry news through intelligent query generation
- Ranking content based on personal research interests and goals
- Providing in-depth analysis of the most relevant materials
- Delivering curated digests with actionable insights

## Technology Stack

- **Language**: Python 3.10+
- **Web Framework**: FastAPI with async/await patterns
- **AI/ML**: OpenAI GPT-4 for ranking and analysis
- **External APIs**:
  - arXiv for academic papers
  - NewsAPI for industry news
  - Tavily for content extraction
  - Supabase for distributed state and caching
- **HTTP Client**: httpx (lightweight, async)
- **HTML Parsing**: BeautifulSoup4 with lxml
- **State Management**: Supabase (distributed) with in-memory fallback
- **Reliability**: Circuit breakers, graceful shutdown, retry logic
- **Containerization**: Docker with security hardening
- **Deployment**: Google Cloud Run with auto-scaling
- **Monitoring**: Logfire for production observability

## Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        A[API Client] -->|POST /generate-digest| B[FastAPI Server]
        A -->|GET /digest-status/task_id| B
    end

    subgraph "API Layer"
        B --> C[Security Middleware]
        C --> D[Background Tasks]
        B --> E[State Manager]
        B --> GS[Graceful Shutdown]
    end

    subgraph "Reliability Layer"
        CB[Circuit Breakers]
        MT[Metrics Tracking]
    end

    subgraph "Service Layer"
        D --> F[Digest Service Enhanced]
        F --> G[ArXiv Fetcher]
        F --> H[News Fetcher]
        F --> I[LLM Client]
        F --> J[Content Extractor]

        H --> K[Query Generator]
        K --> I

        G --> CB
        H --> CB
        I --> CB
        J --> CB
        CB --> MT
    end

    subgraph "External Services"
        G -->|Scrape| L[arXiv Website]
        H -->|API| M[NewsAPI]
        J -->|API| N[Tavily API]
        I -->|API| O[OpenAI API]
        P -->|API| SB[Supabase]
        Q -->|API| SB
    end

    subgraph "Data Layer"
        F --> P[Hybrid Cache]
        F --> Q[State Storage]
        F --> R[HTML Generator]
        P --> PC[In-Memory Cache]
        Q --> QM[In-Memory State]
    end

    style A fill:#e1f5fe
    style B fill:#81c784
    style F fill:#ff9800
    style I fill:#9c27b0
    style O fill:#f44336
    style CB fill:#ffc107
    style SB fill:#00bcd4
```

## Main Components

### 1. **API Layer** (`src/main.py`)

- FastAPI application with async request handling
- RESTful endpoints for digest generation and status checking
- Background task execution for long-running operations
- API key authentication via middleware
- Health check endpoints for monitoring

### 2. **Digest Service** (`src/digest_service_enhanced.py`)

- Orchestrates the complete digest generation workflow
- Coordinates parallel fetching of papers and news
- Manages content ranking and analysis pipeline
- Generates formatted HTML output
- Handles webhook callbacks for task completion

### 3. **Content Fetchers**

#### ArXiv Fetcher (`src/fetcher_lightweight.py`)

- Scrapes daily computer science papers from arXiv catchup pages
- Parses HTML to extract paper metadata
- Supports connection pooling for performance
- Handles multiple arXiv categories

#### News Fetcher (`src/news_fetcher.py`)

- Integrates with NewsAPI for industry news
- Implements intelligent deduplication
- Rate limiting and caching strategies
- Relevance scoring based on query matches

### 4. **Intelligence Layer**

#### LLM Client (`src/llm_client.py`)

- Direct OpenAI API integration
- Handles paper and news ranking with structured outputs
- Performs deep content analysis
- Retry logic with exponential backoff
- Mixed content ranking with type awareness

#### Query Generator (`src/query_generator.py`)

- AI-powered news query generation
- Extracts queries from user profiles
- Temporal awareness for recent news
- Company and role-based query optimization

### 5. **Content Processing**

#### Content Extractor (`src/content_extractor.py`)

- Tavily API integration for full article extraction
- Priority-based extraction with quota management
- Batch processing with rate limiting
- Fallback content strategies

### 6. **Infrastructure Components**

#### State Management

##### In-Memory State (`src/state.py`)
- Local task state persistence (fallback)
- Thread-safe operations with asyncio locks
- Task lifecycle management

##### Distributed State (`src/state_supabase.py`)
- Supabase-based distributed state management
- Enables higher concurrency in Cloud Run
- Automatic fallback to in-memory when unavailable
- Task state stored in `digest_tasks` table

#### Cache System

##### In-Memory Cache (`src/cache.py`)
- TTL-based in-memory caching (fallback)
- Reduces redundant API calls
- Automatic expiration handling

##### Hybrid Cache (`src/cache_supabase.py`)
- Two-tier caching: in-memory LRU + Supabase persistence
- Distributed cache sharing across instances
- Cache entries stored in `cache_entries` table
- Automatic synchronization between tiers

#### Reliability Components

##### Circuit Breaker (`src/circuit_breaker.py`)
- Prevents cascading failures from external services
- Automatic service recovery detection
- Per-service circuit breaker instances
- State optionally persisted to Supabase

##### Graceful Shutdown (`src/graceful_shutdown.py`)
- Proper SIGTERM handling for Cloud Run
- Tracks in-flight requests
- Configurable shutdown timeout
- Ensures clean task completion

##### Metrics (`src/metrics.py`)
- Performance monitoring
- API call tracking
- Success/failure rates
- Latency measurements

#### Security (`src/security.py`)
- API key validation middleware
- FastAPI dependency injection
- Header-based authentication

## Data Flow

1. **Request Initiation**

   - Client sends user profile and preferences to `/generate-digest`
   - System creates task ID and returns immediately
   - Background task begins processing

2. **Content Discovery**

   - ArXiv fetcher scrapes latest CS papers
   - Query generator creates personalized news searches
   - News fetcher retrieves relevant articles
   - Both sources fetched in parallel

3. **Intelligent Ranking**

   - LLM analyzes all content against user profile
   - Mixed ranking considers both papers and news
   - Top N items selected based on relevance scores

4. **Deep Analysis**

   - Content extractor fetches full text for top items
   - LLM performs detailed analysis of each item
   - Generates summaries, key findings, and recommendations

5. **Digest Generation**

   - HTML generator creates formatted output
   - Separate sections for papers and news
   - Rich metadata and visual styling
   - Mobile-responsive design

6. **Result Delivery**
   - Task status updated to completed
   - Client polls `/digest-status/{task_id}` for results
   - Optional webhook callback notification

## Key Design Patterns

- **Async/Await**: All I/O operations use async patterns for scalability
- **Dependency Injection**: Configuration management via Pydantic BaseSettings
- **Background Tasks**: Long operations handled asynchronously
- **Circuit Breaker**: Prevents cascading failures with automatic recovery
- **Graceful Shutdown**: Clean termination with request tracking
- **Hybrid Caching**: Two-tier cache system for performance and distribution
- **Distributed State**: Enables horizontal scaling with Supabase
- **Rate Limiting**: Semaphore-based concurrency control
- **Type Safety**: Comprehensive Pydantic models throughout
- **Fallback Strategies**: Automatic degradation to in-memory when external services fail

## Configuration

The system uses environment-based configuration with sensible defaults:

### Core Configuration
- `OPENAI_API_KEY`: OpenAI API access (required)
- `API_KEY`: Authentication for API endpoints (required)
- `OPENAI_MODEL`: Model selection (default: gpt-4o-mini-2024-07-18)
- `TOP_N_ARTICLES`: Number of articles to analyze (default: 5)
- `LOG_LEVEL`: Logging verbosity (default: INFO)

### Performance Settings
- `HTTP_TIMEOUT`: HTTP request timeout in seconds (default: 30)
- `TASK_TIMEOUT`: Max digest generation time (default: 300s)
- `AGENT_RETRIES`: LLM retry attempts (default: 2)
- `ANALYSIS_CONTENT_MAX_CHARS`: Max content for analysis (default: 20000)
- `RANKING_INPUT_MAX_ARTICLES`: Max articles for ranking (default: 30)
- `EXTRACT_MAX_CONCURRENT`: Concurrent extraction limit (default: 5)
- `EXTRACT_TIMEOUT`: Per-extraction timeout (default: 30s)
- `SHUTDOWN_TIMEOUT`: Graceful shutdown timeout (default: 30s)

### Feature Toggles
- `NEWS_ENABLED`: Enable news fetching (default: true)
- `NEWS_MAX_ARTICLES`: Max news articles to fetch (default: 50)
- `NEWS_MAX_EXTRACT`: Max articles to extract full content (default: 10)
- `NEWS_CACHE_TTL`: News cache duration in seconds (default: 3600)
- `USE_EXTERNAL_STATE`: Enable Supabase for state/cache (default: false)

### External Services
- `NEWSAPI_KEY`: NewsAPI key for news fetching (optional)
- `TAVILY_API_KEY`: Tavily API key for content extraction (optional)
- `SUPABASE_URL`: Supabase project URL (required if USE_EXTERNAL_STATE=true)
- `SUPABASE_KEY`: Supabase anon key (required if USE_EXTERNAL_STATE=true)

### Deployment Settings
- `USE_LIGHTWEIGHT`: Use httpx instead of Playwright (default: true)
- `LOGFIRE_TOKEN`: Monitoring service token (optional)

## Deployment Architecture

- **Container**: Lightweight Docker image with security hardening
- **Memory**: 512Mi for lightweight version (reduced from 1Gi)
- **Scaling**: Auto-scales 0-50 instances based on demand
- **Concurrency**: 
  - Set to 1 with in-memory state (default)
  - Set to 5 with Supabase distributed state
- **Security**: 
  - Non-root user (UID 10001)
  - Read-only filesystem with tmpfs mounts
  - Dropped capabilities except NET_BIND_SERVICE
  - No-new-privileges security option
- **Health Checks**: Available at `/digest-status/health`
- **Monitoring**: Integrated Logfire for production observability
- **Graceful Shutdown**: Proper SIGTERM handling for Cloud Run

### Supabase Schema Requirements

When using distributed state (`USE_EXTERNAL_STATE=true`), the following tables are required:

```sql
-- Task state storage
CREATE TABLE digest_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,
    user_info JSONB,
    result JSONB,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Distributed cache
CREATE TABLE cache_entries (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Circuit breaker state (optional)
CREATE TABLE circuit_breaker_state (
    service_name TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    failure_count INTEGER DEFAULT 0,
    last_failure TIMESTAMP WITH TIME ZONE,
    last_success TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Error Handling

- Comprehensive exception handling at all layers
- Graceful degradation when optional services fail
- Circuit breakers prevent cascading failures
- Automatic fallback from Supabase to in-memory storage
- Detailed logging with structured context
- User-friendly error messages in API responses
- Request tracking ensures clean shutdown
- Retry logic with exponential backoff for transient failures
