#!/bin/sh
# one time setup of the benchmark vm (x86_64 with docker): node for the agent cli, the agent
# package, and a venv with harbor plus this adapter. idempotent.
set -eu
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export NVM_DIR=$HOME/.nvm
[ -s "$NVM_DIR/nvm.sh" ] || curl -fsSL -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
. "$NVM_DIR/nvm.sh"
nvm install 24 >/dev/null && nvm alias default 24 >/dev/null
cd "$HOME/qfa/agent" && npm ci --no-audit --no-fund >/dev/null
[ -d "$HOME/qfa/venv" ] || uv venv -q -p 3.12 "$HOME/qfa/venv"
uv pip install -q -p "$HOME/qfa/venv/bin/python" harbor "$HOME/qfa/agent/tbench"
sudo usermod -aG docker "$(whoami)" 2>/dev/null || true
echo "node $(node --version), harbor $("$HOME/qfa/venv/bin/harbor" --version)"
