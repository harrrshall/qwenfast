#!/bin/sh
# runs terminal-bench with the qwenfast pi agent on the benchmark vm.
#   sh run.sh <small|medium|large|auto> [harbor args...]
#   N=8 DATASET=terminal-bench@2.0 sh run.sh auto --n-tasks 10
# needs ~/qfa/bench.env with QFA_BIG_URL, QFA_SMALL_URL, QFA_API_KEY (mode 600).
set -eu
TIER=${1:?tier: small, medium, large or auto}; shift
. "$HOME/qfa/bench.env"
# the jarvislabs https proxy closes responses after about two minutes, which truncates long
# thinking turns. with QFA_BOX_SSH set (user@host of the gpu box) the models are reached through an
# ssh tunnel bound to this vm's private address, which the task containers can reach.
if [ -n "${QFA_BOX_SSH:-}" ]; then
  IP=$(hostname -I | awk '{print $1}')
  ssh -f -N -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
    -L "$IP:18000:127.0.0.1:8000" -L "$IP:18001:127.0.0.1:8001" "$QFA_BOX_SSH" 2>/dev/null || true
  QFA_BIG_URL="http://$IP:18000"; QFA_SMALL_URL="http://$IP:18001"
  curl -sf -m 5 "$QFA_BIG_URL/health" >/dev/null || { echo "tunnel to $QFA_BOX_SSH is not answering" >&2; exit 1; }
  trap 'pkill -f "ssh -f -N .*$QFA_BOX_SSH" 2>/dev/null || true' EXIT
fi
export QFA_TIER=$TIER QFA_BIG_URL QFA_SMALL_URL QFA_API_KEY QFA_AGENT_DIR="$HOME/qfa/agent"
export NVM_DIR=$HOME/.nvm; . "$NVM_DIR/nvm.sh" >/dev/null
JOB=${JOB:-tb-$TIER-$(date -u +%Y%m%d-%H%M%S)}
cd "$HOME/qfa"
"$HOME/qfa/venv/bin/harbor" run \
  --dataset "${DATASET:-terminal-bench@2.0}" \
  --agent qwenfast_tbench:QwenfastPiAgent --model qwenfast/qwen3.8-27b --ak version=0.87.1 \
  --jobs-dir "$HOME/qfa/jobs" --job-name "$JOB" --n-concurrent "${N:-8}" \
  --agent-setup-timeout-multiplier 3 \
  --max-retries 2 --retry-include AgentSetupTimeoutError --retry-include NetworkConnectionError \
  --yes "$@"
"$HOME/qfa/venv/bin/python" "$HOME/qfa/agent/tbench/report.py" "$HOME/qfa/jobs/$JOB"
