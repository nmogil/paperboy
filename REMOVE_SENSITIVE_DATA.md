# Removing Sensitive Data from Git History

## Issue
The file `pipedream/trigger-generate-digest.js` contains an exposed API key and has been committed to git history.

## Steps Completed
1. ✅ Removed file from git tracking: `git rm --cached pipedream/trigger-generate-digest.js`
2. ✅ Added `pipedream/` to `.gitignore`

## Next Steps to Clean History

### Option 1: Use git filter-branch (Traditional)
```bash
# Remove file from all commits
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch pipedream/trigger-generate-digest.js" \
  --prune-empty --tag-name-filter cat -- --all
```

### Option 2: Use BFG Repo-Cleaner (Easier)
```bash
# Install BFG
brew install bfg

# Remove the file from history
bfg --delete-files trigger-generate-digest.js

# Clean up
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

### Option 3: Use git-filter-repo (Recommended)
```bash
# Install git-filter-repo
pip install git-filter-repo

# Remove the file
git filter-repo --path pipedream/trigger-generate-digest.js --invert-paths
```

## After Cleaning History

1. **Force push to remote**:
   ```bash
   git push origin --force --all
   git push origin --force --tags
   ```

2. **Rotate the exposed API key immediately**:
   - The API key `4b9e132b4145b86bcce9adae2a4f4f2cd8398d6a3f022a7dc9a03711e6167d52` is exposed
   - Generate a new one and update your Pipedream environment

3. **Notify collaborators** to re-clone the repository

## Safer Alternative for Pipedream

Instead of storing the script in git, consider:
1. Copying the script directly to Pipedream
2. Using Pipedream environment variables for all sensitive data
3. Keeping only a template file in git (with placeholders)

## Template File Example

Create `pipedream/trigger-generate-digest.template.js`:
```javascript
// Configuration - use Pipedream environment variables
const TOP_N_ARTICLES = 5;
const CALLBACK_URL = process.env.CALLBACK_URL;
const API_KEY = process.env.PAPERBOY_API_KEY;
const ENDPOINT_URL = process.env.PAPERBOY_ENDPOINT;
```