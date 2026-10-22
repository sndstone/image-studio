#!/bin/bash
# Builds "Image Studio.app" from Sources/ and backend/.
set -euo pipefail
cd "$(dirname "$0")"

APP="build/Image Studio.app"
VERSION=$(tr -d '[:space:]' < VERSION)
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

swiftc -O -wmo -swift-version 5 -parse-as-library \
  -target arm64-apple-macos14.0 \
  -o "$APP/Contents/MacOS/ImageStudio" Sources/*.swift

cp backend/*.py backend/setup.sh backend/requirements.lock "$APP/Contents/Resources/"

if [ -f assets/icon.png ]; then
  ICONSET=build/AppIcon.iconset
  rm -rf "$ICONSET" && mkdir -p "$ICONSET"
  for s in 16 32 128 256 512; do
    sips -z $s $s assets/icon.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    sips -z $((s*2)) $((s*2)) assets/icon.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
fi

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Image Studio</string>
  <key>CFBundleDisplayName</key><string>Image Studio</string>
  <key>CFBundleIdentifier</key><string>local.imagestudio</string>
  <key>CFBundleExecutable</key><string>ImageStudio</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST

# Release builds (IMAGE_STUDIO_ADHOC_SIGN=1) are signed ad-hoc so no local signing identity ends up in a download.
# A stable identity keeps the app's Keychain permission ("Always Allow") valid across rebuilds.
if [ "${IMAGE_STUDIO_ADHOC_SIGN:-0}" = 1 ]; then
  codesign --force --sign - "$APP"
elif SIGNING=$(scripts/signing.sh); then
  IFS=$'\t' read -r KEYCHAIN IDENTITY <<<"$SIGNING"
  codesign --force --keychain "$KEYCHAIN" --sign "$IDENTITY" "$APP"
else
  SIGNING_STATUS=$?
  if [ "$SIGNING_STATUS" -eq 127 ]; then
    echo "warning: signing tools unavailable, signing ad-hoc (macOS will re-ask for Keychain access)" >&2
    codesign --force --sign - "$APP"
  else
    echo "error: signing identity setup failed; build aborted" >&2
    exit "$SIGNING_STATUS"
  fi
fi
echo "Built $APP"
