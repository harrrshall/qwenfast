#!/bin/sh
# one command gpu backend on jarvislabs: creates (or reuses) an h200 box, uploads the engine and
# these scripts, and starts both models under supervision. it also writes .secrets/agent_key and
# .secrets/agent_box_id, which code/install.sh reads so the gateway can manage the box.
#
#   sh scripts/jarvis_setup.sh               create a new box named qwenfast-agent
#   sh scripts/jarvis_setup.sh --box <id>    use an existing box (a paused one is resumed)
#
# needs the jarvislabs cli, logged in (`jl setup`). the first start downloads about 65 gb of
# weights and builds the python environment, which takes 15 to 25 minutes; later starts take two.
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
SECRETS="$REPO/.secrets"
BOX=""
[ "${1:-}" = "--box" ] && BOX=${2:?usage: jarvis_setup.sh [--box <id>]}

command -v jl >/dev/null 2>&1 || { echo "install the jarvislabs cli first: uv tool install jarvislabs, then jl setup" >&2; exit 1; }
jl list --json >/dev/null 2>&1 || { echo "the jarvislabs cli is not logged in: run jl setup" >&2; exit 1; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); v=d.get('$1'); print('' if v is None else v)"; }

mkdir -p "$SECRETS" && chmod 700 "$SECRETS"
if [ ! -s "$SECRETS/agent_key" ]; then
  (umask 077; python3 -c "import secrets; print('qf-' + secrets.token_urlsafe(32))" > "$SECRETS/agent_key")
  echo "generated a model server key in .secrets/agent_key"
fi

if [ -z "$BOX" ]; then
  echo "creating an h200 box (on demand, region in2, 250 gb disk)"
  out=$(jl create --gpu H200 --region IN2 --storage 250 --template pytorch --http-ports 8000,8001,8080 \
        --name qwenfast-agent --yes --json)
  BOX=$(printf '%s' "$out" | json machine_id)
  [ -n "$BOX" ] || { echo "box creation failed: $out" >&2; exit 1; }
else
  status=$(jl get "$BOX" --json | json status)
  if [ "$status" = "Paused" ]; then
    echo "resuming box $BOX"
    BOX=$(jl resume "$BOX" --http-ports 8000,8001,8080 --yes --json | json machine_id)
  fi
fi
echo "$BOX" > "$SECRETS/agent_box_id"
echo "box $BOX"

# the box layout the scripts expect: /home/engine/qwenfast, /home/*.sh, /home/.secrets/agent_key
tmp=$(mktemp -d)
(cd "$REPO" && tar czf "$tmp/src.tgz" --exclude __pycache__ engine/qwenfast engine/requirements.txt scripts)
jl upload "$BOX" "$tmp/src.tgz" /home/qwenfast-src.tgz >/dev/null
jl upload "$BOX" "$SECRETS/agent_key" /home/agent_key.upload >/dev/null
rm -rf "$tmp"
jl exec "$BOX" -- sh -lc '
  set -e
  rm -rf /home/qf-src && mkdir -p /home/qf-src /home/engine /home/.secrets && chmod 700 /home/.secrets
  tar xzf /home/qwenfast-src.tgz -C /home/qf-src
  rm -rf /home/engine/qwenfast && cp -R /home/qf-src/engine/qwenfast /home/engine/qwenfast
  cp /home/qf-src/engine/requirements.txt /home/engine/requirements.txt
  cp /home/qf-src/scripts/*.sh /home/
  mv /home/agent_key.upload /home/.secrets/agent_key && chmod 600 /home/.secrets/agent_key
  rm -rf /home/qf-src /home/qwenfast-src.tgz
' >/dev/null

run=$(jl run --on "$BOX" --no-follow --json --yes -- sh /home/remote_agent_bringup.sh | json run_id)
echo "bring up started ($run): weights, environment, then both servers under supervision"
echo "next: sh code/install.sh, then qfc status shows both models once they are up"
