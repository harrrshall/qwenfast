#!/bin/sh
# The two model agent backend on one h200, kept alive for days.
#
#   big   port 8000  qwenfast serving Qwen3.8-27B-FP8        (hard tasks, thinking on)
#   small port 8001  vllm serving Qwen3.6-35B-A3B-FP8         (easy tasks, 3B active, fast)
#
# start with `jl run --on <id> --no-follow -- sh /home/remote_agent_stack.sh`. idempotent: a
# second copy finds the supervisors' locks held and just becomes another keeper, which exits
# because the keeper lock is taken. layers of recovery, innermost first:
#   1. remote_supervise.sh restarts a dead or wedged server with backoff;
#   2. this keeper restarts a supervisor that itself died;
#   3. the local agent daemon (agent/) re-runs this script after a box resume or reboot.
#
# memory: qwenfast sizes its pools once from its flags (about 83 GiB with the values below) and
# vllm takes SMALL_GPU_UTIL of the device. the big tier starts first and must be healthy before
# the small one is launched, so each always finds its share free, also after a lone restart.

set -u
SECRETS_DIR=${SECRETS_DIR:-/home/.secrets}
BASE=${BASE:-/home/qwenfast-results/agent}
mkdir -p "$BASE"
. /home/supervisor_lock.sh
KLOG="$BASE/keeper.log"
klog() { printf '%s keeper: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$KLOG"; }

if ! lock_acquire "$BASE/keeper.lock" "$$"; then
  klog "keeper already running (pid $(lock_holder "$BASE/keeper.lock"))"; exit 0
fi
trap 'lock_release "$BASE/keeper.lock" "$$"; exit 0' INT TERM HUP

KEY_FILE="$SECRETS_DIR/agent_key"
[ -s "$KEY_FILE" ] || { klog "FATAL: $KEY_FILE missing"; exit 2; }
KEYS_JSON="$SECRETS_DIR/agent_keys.json"
# qwenfast reads a keys file (never a key on the command line, where ps would show it)
( umask 077; python3 -c 'import json,sys; print(json.dumps({"keys":[{"key":open(sys.argv[1]).read().strip(),"name":"agent","admin":True}]}))' "$KEY_FILE" > "$KEYS_JSON" )

. /home/venv_vllm/bin/activate
export PYTHONPATH=/home/engine HF_HOME=/home/hf HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=0
export CUDA_HOME=/home/venv_vllm/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

snap() { ls -d "/home/hf/hub/models--$1/snapshots"/*/ 2>/dev/null | head -1; }
BIG_MODEL=$(snap Qwen--Qwen3.8-27B-FP8)
SMALL_MODEL=$(snap Qwen--Qwen3.6-35B-A3B-FP8)
[ -n "$BIG_MODEL" ] || { klog "FATAL: big weights missing"; exit 2; }

BIG_SEQS=${BIG_SEQS:-24}
BIG_CTX=${BIG_CTX:-131072}
BIG_KV_PAGES=${BIG_KV_PAGES:-20480}          # 16 tokens per page: 327k tokens of kv across slots and cache
BIG_PREFIX_ENTRIES=${BIG_PREFIX_ENTRIES:-16}  # turn to turn prefix cache: 75 MiB of gdn state each
SMALL_CTX=${SMALL_CTX:-131072}
SMALL_SEQS=${SMALL_SEQS:-32}
SMALL_GPU_UTIL=${SMALL_GPU_UTIL:-0.38}

cat > "$BASE/big.sh" <<EOF
exec python -m qwenfast.runtime.serve --model "$BIG_MODEL" --served-model-name qwen3.8-27b \
  --preset fastest --spec-k 3 --spec-max-batch 16 --spec-sampling --mixed-forward --mixed-graphs --overlap \
  --overlap-min-fill 0.75 --prefill-chunk-tokens 8192 --detok-workers 0 --async-scheduling \
  --max-num-seqs $BIG_SEQS --max-model-len $BIG_CTX --n-kv-pages $BIG_KV_PAGES \
  --prefix-cache-entries $BIG_PREFIX_ENTRIES --prefix-cache-min-tokens 512 \
  --gpu-memory-utilization 0.99 --host 0.0.0.0 --port 8000 --http-keep-alive-timeout 600 \
  --default-max-tokens 8192 --max-output-tokens 32768 --min-completion-tokens 1024 \
  --max-inflight-requests 64 --max-request-bytes 16000000 --max-messages 4000 \
  --max-streams-per-key 64 --max-streams-per-ip 64 --request-timeout 3600 --drain-timeout 60 \
  --api-keys-file $KEYS_JSON --usage-db $BASE/big-usage.sqlite --gpu-rate-inr-per-hour 378.27
EOF

cat > "$BASE/small.sh" <<EOF
export VLLM_API_KEY=\$(cat $KEY_FILE)
# vllm sizes its kv cache from free memory at startup: never measure while the big tier is still
# loading (or restarting), or the two race for the same gigabytes. give up after 30 min and let the
# supervisor's backoff try again.
i=0
while [ "\$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/health)" != 200 ] && [ \$i -lt 180 ]; do
  sleep 10; i=\$((i + 1))
done
exec vllm serve "$SMALL_MODEL" --served-model-name qwen3.6-35b-a3b --host 0.0.0.0 --port 8001 \
  --max-model-len $SMALL_CTX --max-num-seqs $SMALL_SEQS --gpu-memory-utilization $SMALL_GPU_UTIL \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-prefix-caching --disable-uvicorn-access-log --uvicorn-log-level warning
EOF

WARM='K=$(cat '"$KEY_FILE"'); for PM in 8000:qwen3.8-27b 8001:qwen3.6-35b-a3b; do curl -s -o /dev/null --max-time 300 -X POST http://127.0.0.1:${PM%%:*}/v1/chat/completions -H "Authorization: Bearer $K" -H "Content-Type: application/json" -d "{\"model\":\"${PM#*:}\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":false}}"; done'

sup_alive() { lock_alive "$BASE/$1/supervisor.lock" 2>/dev/null; }
start_sup() {
  name=$1; port=$2; script=$3; grace=$4
  NAME=$name PORT=$port LOG_DIR="$BASE/$name" STARTUP_GRACE=$grace WARMUP_CMD="$WARM" \
    setsid sh /home/remote_supervise.sh sh "$BASE/$script" < /dev/null >> "$KLOG" 2>&1 &
  klog "started $name supervisor (pid $!)"
}
healthy() { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$1/health")" = "200" ]; }

klog "keeper $$ up; big=$BIG_MODEL small=${SMALL_MODEL:-<missing>}"
while true; do
  sup_alive big || start_sup big 8000 big.sh 1800
  if [ -n "$SMALL_MODEL" ] && ! sup_alive small; then
    # never race the big tier for memory: it must hold its pools before vllm measures free hbm
    if healthy 8000; then start_sup small 8001 small.sh 1200; else klog "small waits for big to be healthy"; fi
  fi
  # interruptible: a TERM trap runs as soon as `wait` returns, not after a 30 s sleep
  sleep 30 & wait $!
done
