# Deletable Files Analysis

This document identifies files that can be safely deleted from the paperboy repository after the refactoring to the lightweight implementation.

## Summary

The repository has undergone a refactoring from an agent-based implementation to a service-based architecture. Many files from the old implementation and refactoring process can now be cleaned up.

## ✅ SAFE TO DELETE - Refactoring Artifacts

These files were created during the refactoring process and are no longer needed:

### Refactoring Documentation
- `REFACTOR_GUIDE.md` - Temporary refactoring documentation
- `REFACTOR_SUMMARY.md` - Summary of refactoring changes

### Refactored Build Files  
- `docker-compose.refactored.yaml` - Temporary refactored compose file
- `deploy_refactored.sh` - Temporary refactored deployment script
- `architecture_refactored.mmd` - Refactored architecture diagram
- `data-refactored/` - Empty refactored data directory

### Refactoring Tests
- `tests/test_refactored.py` - Tests created during refactoring

## ✅ SAFE TO DELETE - Old Implementation Tests

These tests reference the old agent-based implementation that has been archived:

- `tests/test_agent.py` - Tests for old agent.py (now archived)
- `tests/test_agent_tools.py` - Tests for old agent tools (now archived)

## ⚠️ REVIEW BEFORE DELETING - Development/Testing Files

These files may have value but should be reviewed:

### Testing Scripts
- `run_tests.py` - Custom test runner (check if pytest is sufficient)
- `test_api_local.py` - Local API testing script (may be useful for development)
- `test_noah_payload.sh` - Specific test script (personal testing artifact?)
- `test_payload.json` - Test data (check if used by active tests)
- `tests/test_architecture.py` - Architecture tests (may still be relevant)

### Documentation
- `TESTING_GUIDE.md` - Testing documentation (may be valuable)
- `CONTRIBUTING.md` - Contribution guidelines (valuable for open source)

### Build Configuration
- `cloudbuild-dynamic.yaml` - Dynamic cloud build (check if used vs cloudbuild.yaml)

### Architecture Diagrams
- `architecture.mmd` - Original architecture diagram (documentation value)
- `architecture_simplified.mmd` - Simplified architecture (documentation value)

## 🔒 KEEP - Active Implementation Files

These files are part of the current lightweight implementation:

### Core Application
- `src/main.py` - FastAPI application
- `src/digest_service.py` - New service-based implementation
- `src/llm_client.py` - LLM client
- `src/fetcher_lightweight.py` - ArXiv fetcher
- `src/config.py` - Configuration
- `src/models.py` - Data models
- `src/api_models.py` - API models
- `src/security.py` - Security middleware
- `src/state.py` - State management

### Configuration & Deployment
- `Dockerfile` - Production container
- `docker-compose.lightweight.yaml` - Development compose
- `requirements.lightweight.txt` - Dependencies
- `config/settings.py` - Settings module
- `deploy_cloudrun.sh` - Cloud Run deployment
- `cloudbuild.yaml` - Google Cloud Build

### Documentation
- `README.md` - Project documentation
- `CLAUDE.md` - AI assistant instructions
- `LICENSE` - License file

### Tests & Data
- `tests/conftest.py` - Test configuration
- `tests/integration_test.py` - Integration tests
- `data/` directory - State persistence
- `pipedream/` - External integrations

### Archived Files
- `archived/` directory - Contains old agent implementation for reference

## Recommended Cleanup Commands

```bash
# Delete refactoring artifacts
rm REFACTOR_GUIDE.md REFACTOR_SUMMARY.md
rm docker-compose.refactored.yaml deploy_refactored.sh
rm architecture_refactored.mmd
rm -rf data-refactored/
rm tests/test_refactored.py

# Delete old implementation tests
rm tests/test_agent.py tests/test_agent_tools.py

# Review and potentially delete (check usage first)
# rm run_tests.py test_api_local.py test_noah_payload.sh test_payload.json
# rm tests/test_architecture.py
# rm TESTING_GUIDE.md cloudbuild-dynamic.yaml
```

## Notes

1. The `archived/` directory contains the old agent implementation and should be kept for reference
2. Current implementation uses digest_service.py and llm_client.py instead of agent.py
3. References to "agent" in config files are legacy naming (agent_retries, etc.) but are still used
4. Always run tests after cleanup to ensure nothing was accidentally removed