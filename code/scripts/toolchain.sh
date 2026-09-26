#!/bin/sh
# downloads the pinned bun and node for this os and cpu into $QFC_HOME/toolchain, verifies them
# against the publishers' sha256 lists, and prints the PATH to use. needs only curl, tar, unzip.
#   eval "$(sh scripts/toolchain.sh)"
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
. "$HERE/toolchain.env"
QFC_HOME=${QFC_HOME:-$HOME/.qwenfast-code}
TC="$QFC_HOME/toolchain"
mkdir -p "$TC"

case "$(uname -s)" in Darwin) OS=darwin ;; Linux) OS=linux ;; *) echo "unsupported os $(uname -s)" >&2; exit 1 ;; esac
case "$(uname -m)" in arm64|aarch64) ARCH=arm64; BUNARCH=aarch64 ;; x86_64|amd64) ARCH=x64; BUNARCH=x64 ;; *) echo "unsupported cpu $(uname -m)" >&2; exit 1 ;; esac

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }

BUN_DIR="$TC/bun-$BUN_VERSION"
if [ ! -x "$BUN_DIR/bun" ]; then
  asset="bun-$OS-$BUNARCH.zip"
  tmp=$(mktemp -d)
  fetch "https://github.com/oven-sh/bun/releases/download/bun-v$BUN_VERSION/$asset" "$tmp/$asset"
  fetch "https://github.com/oven-sh/bun/releases/download/bun-v$BUN_VERSION/SHASUMS256.txt" "$tmp/sums"
  want=$(grep " $asset\$" "$tmp/sums" | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$(sha256 "$tmp/$asset")" ] || { echo "bun checksum mismatch" >&2; exit 1; }
  (cd "$tmp" && unzip -q "$asset")
  mkdir -p "$BUN_DIR" && mv "$tmp/bun-$OS-$BUNARCH/bun" "$BUN_DIR/bun" && rm -rf "$tmp"
fi

NODE_DIR="$TC/node-$NODE_VERSION"
if [ ! -x "$NODE_DIR/bin/node" ]; then
  asset="node-v$NODE_VERSION-$OS-$ARCH.tar.gz"
  tmp=$(mktemp -d)
  fetch "https://nodejs.org/dist/v$NODE_VERSION/$asset" "$tmp/$asset"
  fetch "https://nodejs.org/dist/v$NODE_VERSION/SHASUMS256.txt" "$tmp/sums"
  want=$(grep "  $asset\$" "$tmp/sums" | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$(sha256 "$tmp/$asset")" ] || { echo "node checksum mismatch" >&2; exit 1; }
  tar -xzf "$tmp/$asset" -C "$tmp"
  rm -rf "$NODE_DIR" && mv "$tmp/node-v$NODE_VERSION-$OS-$ARCH" "$NODE_DIR" && rm -rf "$tmp"
fi

echo "export PATH=\"$BUN_DIR:$NODE_DIR/bin:\$PATH\" QFC_OS=$OS QFC_ARCH=$ARCH"
