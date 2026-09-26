#!/bin/sh
# builds the qfc binary for this os and cpu from the pinned opencode tag plus patches/apply.py.
#   sh scripts/build.sh            -> dist/qfc-<os>-<arch>
# reproducible: pinned toolchain (toolchain.env), pinned upstream tag, frozen lockfile.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
. "$HERE/toolchain.env"
eval "$(sh "$HERE/scripts/toolchain.sh")"
BUILD=${QFC_BUILD:-$HERE/.build}
SRC="$BUILD/opencode"
mkdir -p "$BUILD" "$HERE/dist"

if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch -q --depth 1 origin "refs/tags/$OPENCODE_TAG:refs/tags/$OPENCODE_TAG" 2>/dev/null || true
  git -C "$SRC" reset -q --hard
  git -C "$SRC" checkout -q -f "$OPENCODE_TAG"
else
  git clone -q --depth 1 --branch "$OPENCODE_TAG" "$OPENCODE_REPO" "$SRC"
fi
echo "upstream $OPENCODE_TAG at $(git -C "$SRC" rev-parse --short HEAD)"

python3 "$HERE/patches/apply.py" "$SRC"
(cd "$SRC" && bun install --frozen-lockfile >/dev/null)
(cd "$SRC/packages/opencode" && rm -rf dist && \
  OPENCODE_VERSION="$QFC_VERSION" bun run script/build.ts --single --skip-embed-web-ui --skip-install | tail -2)

BIN=$(ls "$SRC"/packages/opencode/dist/opencode-"$QFC_OS"-"$QFC_ARCH"*/bin/opencode | head -1)
OUT="$HERE/dist/qfc-$QFC_OS-$QFC_ARCH"
cp "$BIN" "$OUT" && chmod +x "$OUT"
"$OUT" --version
echo "built $OUT"
