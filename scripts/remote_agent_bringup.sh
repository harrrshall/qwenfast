#!/bin/sh
# fresh box -> serving agent backend. idempotent; safe to re-run after a resume.
#   jl run --on <id> --no-follow -- sh /home/remote_agent_bringup.sh
# downloads only what the agent stack serves (no bf16 27b), builds the pinned venv, then becomes
# the stack keeper (remote_agent_stack.sh) for the life of the box.
set -u
export MODELS="Qwen/Qwen3.8-27B-FP8 Qwen/Qwen3.6-35B-A3B-FP8"
need=0
for m in Qwen--Qwen3.8-27B-FP8 Qwen--Qwen3.6-35B-A3B-FP8; do
  d=$(ls -d /home/hf/hub/models--$m/snapshots/*/ 2>/dev/null | head -1)
  { [ -n "$d" ] && ls "$d"*.safetensors >/dev/null 2>&1; } || need=1
done
[ "$need" = "1" ] && sh /home/remote_download_weights.sh
if ! /home/venv_vllm/bin/python -c "import vllm" 2>/dev/null; then
  # the bootstrap's weight step wants bf16 too; point it at a no-op downloader for this run
  sed 's#/home/remote_download_weights.sh#/dev/null/none#g' /home/remote_bootstrap_box.sh > /home/.bootstrap_noweights.sh
  sh /home/.bootstrap_noweights.sh || { echo "bootstrap failed"; exit 1; }
fi
# a resumed container starts its pids from scratch, so a lock written before the pause can name a
# pid that now belongs to an unrelated process and would make the keeper think it is already
# running. a lock only counts when its process really is that keeper or supervisor.
BASE=/home/qwenfast-results/agent
pgrep -f "sh /home/remote_agent_stack.sh" >/dev/null 2>&1 || rm -rf "$BASE/keeper.lock"
for n in big small; do
  pgrep -f "remote_supervise.sh sh $BASE/$n.sh" >/dev/null 2>&1 || rm -rf "$BASE/$n/supervisor.lock"
done
exec sh /home/remote_agent_stack.sh
