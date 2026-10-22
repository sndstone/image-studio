#!/bin/bash
# Builds "Image Studio-<version>.dmg" and publishes it as GitHub release v<version>.
# Usage: scripts/release.sh [--draft]   (version comes from VERSION)
# Versioning: x.y.0 for feature releases, x.y.z (z ≥ 1) for minor fixes.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=$(tr -d '[:space:]' < VERSION)
TAG="v$VERSION"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "VERSION must be x.y.z, got '$VERSION'" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean; commit first" >&2; exit 1; }
if gh release view "$TAG" >/dev/null 2>&1; then echo "release $TAG already exists" >&2; exit 1; fi

scripts/ci.sh
IMAGE_STUDIO_ADHOC_SIGN=1 ./build.sh

DIST=dist; STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$DIST"
DMG="$DIST/Image-Studio-$VERSION.dmg"
rm -f "$DMG"
ditto "build/Image Studio.app" "$STAGE/Image Studio.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "Image Studio $VERSION" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$DMG" >/dev/null
hdiutil verify "$DMG" >/dev/null
(cd "$DIST" && shasum -a 256 "$(basename "$DMG")" > "$(basename "$DMG").sha256")

git tag -a "$TAG" -m "Image Studio $VERSION" 2>/dev/null || true
git push origin "$TAG"
NOTES=$(awk -v v="$VERSION" '$0 ~ "^## "v"( |$)" {f=1; next} /^## / {f=0} f' CHANGELOG.md)
gh release create "$TAG" "$DMG" "$DMG.sha256" --title "Image Studio $VERSION" --notes "$NOTES" ${1:+"$1"}
