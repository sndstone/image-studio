#!/bin/bash
# Points git at .githooks so scripts/ci.sh runs before every push.
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
touch "$(git rev-parse --git-dir)/info/private-patterns"
echo "pre-push hook installed; add personal patterns to $(git rev-parse --git-dir)/info/private-patterns"
