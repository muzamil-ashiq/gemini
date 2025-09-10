#!/usr/bin/env bash
# Helper to add remote and push current repo to GitHub. Replace REMOTE_URL before running.
set -euo pipefail
REMOTE_URL="REPLACE_WITH_REMOTE_URL"
if [ "$REMOTE_URL" = "REPLACE_WITH_REMOTE_URL" ]; then
  echo "Please edit this file and set REMOTE_URL to your GitHub repo HTTPS URL" >&2
  exit 1
fi
# add remote if not exists
if ! git remote | grep -q origin; then
  git remote add origin "$REMOTE_URL"
fi
# ensure we don't accidentally push large files
git add -A
git commit -m "Prepare repo for remote training: .gitignore and helper" || true
git push -u origin main
