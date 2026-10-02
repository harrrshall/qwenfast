#!/bin/sh
# installs qwenfast code for the current user on macos or linux, from this checkout.
#
#   sh code/install.sh
#
# everything lands in ~/.qwenfast-code (QFC_HOME) plus a `qfc` link in ~/.local/bin:
#   toolchain/   pinned bun and node (toolchain.env), checksum verified
#   bin/         qfc (launcher), qfc-bin (the built binary), qfc-server
#   daemon/      the local gateway + router + gpu box keeper (../agent)
#   agent/       daemon state: config.json, secrets/, tasks/, logs/
#   secrets/     server password (0600)
#   logs/        server log
# config for the tui: ~/.config/qwenfast-code/opencode.json (written once, then yours to edit)
#
# gpu backend: with ../.secrets/{agent_key,agent_box_id} (and jl logged in, or
# ../.secrets/jarvislabs.backup.env) the daemon manages the jarvislabs box itself. otherwise set
# QFC_API_KEY plus QFC_BIG_URL and QFC_SMALL_URL (any two openai compatible servers).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
QFC_HOME=${QFC_HOME:-$HOME/.qwenfast-code}
export QFC_HOME
. "$HERE/toolchain.env"
say() { printf '\033[1m%s\033[0m\n' "$*"; }

say "prerequisites"
missing=""
for t in curl git python3 unzip tar ssh make c++; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
if [ -n "$missing" ]; then
  echo "missing:$missing" >&2
  if [ "$(uname -s)" = Darwin ]; then echo "install the xcode command line tools: xcode-select --install" >&2
  else echo "debian/ubuntu: sudo apt-get install -y curl git python3 unzip xz-utils openssh-client build-essential" >&2
       echo "fedora: sudo dnf install -y curl git python3 unzip xz openssh-clients make gcc-c++" >&2; fi
  exit 1
fi

say "toolchain"
eval "$(sh "$HERE/scripts/toolchain.sh")"
ln -sfn "$QFC_HOME/toolchain/node-$NODE_VERSION" "$QFC_HOME/toolchain/node"
ln -sfn "$QFC_HOME/toolchain/bun-$BUN_VERSION" "$QFC_HOME/toolchain/bun"
echo "bun $(bun --version), node $(node --version)"

say "qfc binary"
BINSRC="$HERE/dist/qfc-$QFC_OS-$QFC_ARCH"
if [ ! -x "$BINSRC" ] || [ "$("$BINSRC" --version 2>/dev/null)" != "$QFC_VERSION" ]; then
  sh "$HERE/scripts/build.sh"
fi
mkdir -p "$QFC_HOME/bin" "$QFC_HOME/logs" "$QFC_HOME/secrets" "$QFC_HOME/agent/secrets"
chmod 700 "$QFC_HOME/secrets" "$QFC_HOME/agent/secrets"
cp "$BINSRC" "$QFC_HOME/bin/qfc-bin.new" && mv "$QFC_HOME/bin/qfc-bin.new" "$QFC_HOME/bin/qfc-bin"
cp "$HERE/bin/qfc" "$QFC_HOME/bin/qfc" && chmod +x "$QFC_HOME/bin/qfc"
cp "$HERE/scripts/services.sh" "$QFC_HOME/services.sh"
cat > "$QFC_HOME/bin/qfc-server" <<EOF
#!/bin/sh
# the qfc session server, 127.0.0.1 only, basic auth with the password from secrets/
OPENCODE_SERVER_PASSWORD=\$(cat "$QFC_HOME/secrets/server_password") exec "$QFC_HOME/bin/qfc-bin" serve \\
  --hostname 127.0.0.1 --port \${QFC_SERVER_PORT:-4097} >> "$QFC_HOME/logs/server.log" 2>&1
EOF
chmod +x "$QFC_HOME/bin/qfc-server"
"$QFC_HOME/bin/qfc-bin" --version

say "daemon"
mkdir -p "$QFC_HOME/daemon"
for f in package.json package-lock.json; do cp "$REPO/agent/$f" "$QFC_HOME/daemon/$f"; done
rm -rf "$QFC_HOME/daemon/src" && cp -R "$REPO/agent/src" "$QFC_HOME/daemon/src"
(cd "$QFC_HOME/daemon" && npm ci --omit=dev --no-audit --no-fund --loglevel=error)

say "secrets and config"
[ -s "$QFC_HOME/secrets/server_password" ] || (umask 077; node -e 'console.log(require("crypto").randomBytes(24).toString("base64url"))' > "$QFC_HOME/secrets/server_password")
copied=""
for f in agent_key agent_box_id jarvislabs.backup.env jarvislabs.env; do
  if [ -s "$REPO/.secrets/$f" ] && [ ! -s "$QFC_HOME/agent/secrets/$f" ]; then
    (umask 077; cp "$REPO/.secrets/$f" "$QFC_HOME/agent/secrets/$f"); copied="$copied $f"
  fi
done
[ -n "${QFC_API_KEY:-}" ] && (umask 077; printf '%s\n' "$QFC_API_KEY" > "$QFC_HOME/agent/secrets/agent_key")
[ -n "$copied" ] && echo "copied from .secrets:$copied"
if [ ! -f "$QFC_HOME/agent/config.json" ]; then
  if [ -n "${QFC_BIG_URL:-}" ]; then
    printf '{"endpoints": {"big": "%s", "small": "%s"}, "jarvis": {"enabled": false}}\n' "$QFC_BIG_URL" "${QFC_SMALL_URL:-$QFC_BIG_URL}" > "$QFC_HOME/agent/config.json"
  else
    echo '{"jarvis": {"idlePauseMinutes": 60}}' > "$QFC_HOME/agent/config.json"
  fi
fi
[ -s "$QFC_HOME/agent/secrets/agent_key" ] || echo "warning: no model server key; set QFC_API_KEY and rerun" >&2
CFG="${XDG_CONFIG_HOME:-$HOME/.config}/qwenfast-code"
mkdir -p "$CFG"
if [ -f "$CFG/opencode.json" ] && ! cmp -s "$HERE/config/opencode.json" "$CFG/opencode.json"; then
  cp "$HERE/config/opencode.json" "$CFG/opencode.default.json"
  echo "kept your $CFG/opencode.json (the shipped default is next to it as opencode.default.json)"
else
  cp "$HERE/config/opencode.json" "$CFG/opencode.json"
fi

if grep -q '"enabled": false' "$QFC_HOME/agent/config.json" 2>/dev/null; then :; elif ! command -v jl >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/jl" ]; then
  say "jarvislabs cli"
  command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
  "$HOME/.local/bin/uv" tool install "jarvislabs==0.2.17" 2>/dev/null || uv tool install "jarvislabs==0.2.17"
fi

say "services"
sh "$QFC_HOME/services.sh" install
sh "$QFC_HOME/services.sh" restart

mkdir -p "$QFC_HOME/daemon/bin" "$HOME/.local/bin"
cp "$REPO/agent/bin/qfa" "$QFC_HOME/daemon/bin/qfa" && chmod +x "$QFC_HOME/daemon/bin/qfa"
ln -sfn "$QFC_HOME/bin/qfc" "$HOME/.local/bin/qfc"
ln -sfn "$QFC_HOME/daemon/bin/qfa" "$HOME/.local/bin/qfa"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "add ~/.local/bin to your PATH to use qfc" ;; esac
say "done: run 'qfc status', then 'qfc' in any project"
