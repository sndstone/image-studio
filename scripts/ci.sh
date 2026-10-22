#!/bin/bash
# Local CI: runs before every push (see scripts/install-hooks.sh) instead of hosted CI.
# Checks: shell syntax, backend unit tests, Swift build without warnings, privacy scan of tracked files.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n== %s\n' "$1"; }

step "Shell syntax"
for f in build.sh backend/setup.sh scripts/*.sh .githooks/*; do bash -n "$f"; done

step "Backend tests"
PYTHON=${IMAGE_STUDIO_PYTHON:-"$HOME/Library/Application Support/Image Studio/venv/bin/python"}
[ -x "$PYTHON" ] || PYTHON=python3
(cd backend && PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m unittest discover -s tests -q)

step "Swift build (warnings are errors)"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
swiftc -O -wmo -swift-version 5 -parse-as-library -warnings-as-errors \
  -target arm64-apple-macos14.0 -o "$OUT/ImageStudio" Sources/*.swift

step "Privacy scan"
# Generic patterns: home-folder paths, private IPv4 addresses, mDNS hostnames, tokens and keys.
# Personal names, hostnames or file names go in .git/info/private-patterns (one extended regex
# per line); that file lives inside .git, so it is never committed.
PATTERNS='/Users/[A-Za-z0-9._-]+|/home/[a-z][a-z0-9_-]+/|\b(10|192\.168|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]+\.[0-9]+(\.[0-9]+)?\b|[A-Za-z0-9-]+\.local\b|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[opsu]_[A-Za-z0-9]{20,}|PRIVATE KEY-----'
PRIVATE=$(git rev-parse --git-dir)/info/private-patterns
if [ -s "$PRIVATE" ]; then
  PATTERNS="$PATTERNS|$(grep -v '^[[:space:]]*\(#\|$\)' "$PRIVATE" | paste -sd'|' -)"
fi
FILES=()
while IFS= read -r -d '' f; do FILES+=("$f"); done < <(git ls-files -z --cached --others --exclude-standard)
if HITS=$(grep -nIiE "$PATTERNS" "${FILES[@]}" | grep -v '^scripts/ci\.sh:'); then
  echo "$HITS"; echo "privacy scan: remove the matches above" >&2; exit 1
fi
# Only the app icon may be committed as an image; example images tend to come from personal photos.
if IMAGES=$(printf '%s\n' "${FILES[@]}" | grep -iE '\.(png|jpe?g|heic|webp|gif|tiff?)$' | grep -v '^assets/icon\.png$'); then
  echo "$IMAGES"; echo "privacy scan: images other than assets/icon.png are not allowed" >&2; exit 1
fi

printf '\nCI passed\n'
