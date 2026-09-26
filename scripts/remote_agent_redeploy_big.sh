#!/bin/sh
# swap in a freshly uploaded engine and restart only the big tier (the small tier keeps serving).
#   upload:  jl upload <id> engine/qwenfast /home/engine_new/qwenfast   (lands in .../qwenfast/qwenfast)
#   then:    jl run --on <id> --no-follow -- sh /home/remote_agent_redeploy_big.sh
set -u
BASE=/home/qwenfast-results/agent
. /home/supervisor_lock.sh
NEW=/home/engine_new/qwenfast/qwenfast
[ -f "$NEW/runtime/serve.py" ] || { echo "no uploaded engine at $NEW"; exit 2; }
TS=$(date -u +%Y%m%d%H%M%S)
mkdir -p /home/engine_backups
mv /home/engine/qwenfast "/home/engine_backups/qwenfast-$TS"
mv "$NEW" /home/engine/qwenfast
rm -rf /home/engine_new
echo "engine swapped (backup /home/engine_backups/qwenfast-$TS)"
# stop the keeper first (it would restart the big supervisor), then the big supervisor
for L in "$BASE/keeper.lock" "$BASE/big/supervisor.lock"; do
  p=$(lock_holder "$L")
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then kill -TERM "$p"; echo "stopped pid $p ($L)"; fi
done
# both must be gone before the new keeper starts, or it finds the old keeper's lock and exits
for i in $(seq 1 120); do
  lock_alive "$BASE/big/supervisor.lock" || lock_alive "$BASE/keeper.lock" || break
  sleep 1
done
echo "big stopped; starting keeper (regenerates big.sh, starts big, leaves small alone)"
exec sh /home/remote_agent_stack.sh
