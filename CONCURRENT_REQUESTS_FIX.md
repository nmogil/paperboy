# Fix for 50 Concurrent Request Processing

## Problem
When sending up to 50 requests simultaneously from Pipedream, only one user's digest was being generated.

## Root Cause
- Cloud Run was configured with `max-instances=1`
- The service uses in-memory state (`tasks` dict) to track request status
- Multiple requests per instance would cause state conflicts

## Solution

Updated Cloud Run configuration:
- **MAX_INSTANCES**: 50 (one instance per concurrent request)
- **CONCURRENCY**: 1 (MUST remain 1 due to in-memory state)
- **MEMORY**: 512Mi per instance (sufficient for lightweight version)

## Why Concurrency Must Be 1

Your app stores task state in memory:
```python
tasks = {}  # In-memory storage
tasks[task_id] = {"status": "PENDING", ...}
```

If multiple requests share an instance:
- Request A creates task_id_1 in instance 1
- Request B creates task_id_2 in instance 1  
- Later, `/digest-status/task_id_1` might route to instance 2 (404 error)

## Cost Analysis

With 50 instances × 512Mi each:
- **If all 50 run simultaneously**: ~$0.001 per event
- **Daily cost** (50 requests): ~$0.05
- **Monthly cost**: ~$1.50

This is still very reasonable since:
- Instances only run during processing (~1 minute each)
- Cloud Run bills per 100ms of actual usage
- Instances auto-scale down to 0 when idle

## Deployment

```bash
# Deploy the updated configuration
./deploy_cloudrun.sh
```

## Alternative Architecture (Future)

To avoid needing 50 instances, consider:
1. **Remove status endpoint** - Use webhooks only, no polling
2. **Use Cloud Firestore** - Shared state across instances
3. **Use Cloud Tasks** - Proper job queue

But for 50 requests/day, the current approach is fine and simple.

## Testing with Pipedream

Your script should now work with all 50 users:
1. Update line 9 to use your actual Cloud Run URL
2. Keep the 1-second delay to avoid overwhelming the service
3. All users should receive their digests via webhook