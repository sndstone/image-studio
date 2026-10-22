#!/bin/bash
# Keeps one self-signed code-signing identity for Image Studio builds, in a private keychain file under Application Support
# outside the project (the login keychain is not touched). A stable signature lets macOS remember "Always Allow"
# for the app's Keychain item across rebuilds; ad-hoc signatures change with every build.
# Usage: scripts/signing.sh  -> prints "KEYCHAIN<TAB>IDENTITY_SHA1" (creates them on first run)
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="$HOME/Library/Application Support/Image Studio Signing"
if [ -d .signing ]; then
  if [ -e "$DIR" ]; then
    echo "Both .signing and $DIR exist; refusing to replace either signing identity." >&2
    exit 1
  fi
fi
# Only missing tools permit the build to fall back to ad-hoc signing. Check
# before migrating or creating any signing material.
if ! command -v openssl >/dev/null || ! command -v security >/dev/null; then
  echo "Signing requires openssl and security; one or both tools are unavailable." >&2
  exit 127
fi
if [ -d .signing ]; then
  mkdir -p "$(dirname "$DIR")"
  mv .signing "$DIR"
fi
KC="$DIR/imagestudio-signing.keychain-db"
NAME="Image Studio Local Signing"
mkdir -p "$DIR" && chmod 700 "$DIR"

identity_ok() {
  [ -f "$KC" ] && [ -f "$DIR/keychain-password" ] &&
    security unlock-keychain -p "$(cat "$DIR/keychain-password")" "$KC" 2>/dev/null &&
    security find-identity -p codesigning "$KC" 2>/dev/null | grep -q "$NAME"
}

if ! identity_ok; then
  # (Re)create from scratch: a half-finished earlier attempt is discarded.
  rm -f "$KC" "$DIR/keychain-password"
  # create-keychain may add the new keychain to the user's search list; put the list back afterwards.
  SEARCH_OUTPUT=$(security list-keychains -d user)
  SEARCH=()
  while IFS= read -r keychain; do
    SEARCH+=("$keychain")
  done < <(printf '%s\n' "$SEARCH_OUTPUT" | sed -e 's/^ *"//' -e 's/"$//')
  restore_search() { security list-keychains -d user -s "${SEARCH[@]}"; }
  TMP=$(mktemp -d)
  cleanup() {
    result=$?
    trap - EXIT
    restore_search || result=1
    rm -rf "$TMP"
    exit "$result"
  }
  trap cleanup EXIT
  PASS=$(openssl rand -hex 24)
  cat > "$TMP/cert.cnf" <<CNF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $NAME
[ext]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
CNF
  openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$TMP/cert.cnf" \
    -keyout "$TMP/key.pem" -out "$TMP/cert.pem" 2>/dev/null
  P12PASS=$(openssl rand -hex 12)
  # macOS's security tool needs the legacy PKCS#12 algorithms; LibreSSL has no -legacy flag (and doesn't need it).
  openssl pkcs12 -export -legacy -inkey "$TMP/key.pem" -in "$TMP/cert.pem" -name "$NAME" \
    -out "$TMP/id.p12" -passout "pass:$P12PASS" 2>/dev/null \
  || openssl pkcs12 -export -inkey "$TMP/key.pem" -in "$TMP/cert.pem" -name "$NAME" \
    -out "$TMP/id.p12" -passout "pass:$P12PASS"
  security create-keychain -p "$PASS" "$KC"
  security set-keychain-settings "$KC"  # no auto-lock timeout
  security import "$TMP/id.p12" -k "$KC" -P "$P12PASS" -T /usr/bin/codesign >/dev/null
  security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$PASS" "$KC" >/dev/null
  printf '%s' "$PASS" > "$DIR/keychain-password" && chmod 600 "$DIR/keychain-password"
  identity_ok || { echo "signing identity setup failed" >&2; rm -f "$KC" "$DIR/keychain-password"; exit 1; }
fi

SHA=$(security find-certificate -c "$NAME" -Z "$KC" | awk '/SHA-1/{print $NF; exit}')
printf '%s\t%s\n' "$KC" "$SHA"
