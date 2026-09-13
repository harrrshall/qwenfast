#!/bin/sh
# One benchmark config end-to-end on the GPU host:
#   TAG=<tag> VLLM_EXTRA_ARGS="<extra vllm flags>" [CONC=..] [ENV="K=V K=V"] sh remote_run_config.sh
# Kills any running vLLM server, starts a new one with the extra args, waits for readiness, runs the standard sweep,
# then leaves the server running (next config kills it). Server log: /home/qwenfast-results/server-$TAG.log
set -u
TAG=${TAG:?}; mkdir -p /home/qwenfast-results
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 5; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 3
export HF_HOME=/home/hf HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}
for kv in ${ENV:-}; do export "$kv"; done
. /home/venv_vllm/bin/activate
LOG=/home/qwenfast-results/server-$TAG.log
nohup python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL:-Qwen/Qwen3.8-27B-FP8} --served-model-name qwen3.8-27b \
  --max-model-len ${MAX_MODEL_LEN:-65536} --max-num-seqs ${MAX_NUM_SEQS:-256} --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 --port 8000 ${VLLM_EXTRA_ARGS:-} > "$LOG" 2>&1 &
echo "server pid $! log $LOG"
for i in $(seq 1 120); do
  if curl -s -m 3 localhost:8000/v1/models >/dev/null 2>&1; then echo "READY after ~$((i*10))s"; break; fi
  if grep -qE "Engine core initialization failed|Traceback" "$LOG"; then echo "SERVER FAILED"; grep -E "Error|raise" "$LOG" | tail -5; exit 1; fi
  sleep 10
done
curl -s -m 3 localhost:8000/v1/models >/dev/null || { echo "SERVER TIMEOUT"; exit 1; }
grep -E "GDN decode kernel|attention backend|KV cache size|Graph capturing finished|spec" "$LOG" | cut -c1-200
# quick correctness probe
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":20,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c 'import json,sys;print("probe:",json.load(sys.stdin)["choices"][0]["message"]["content"])'
TAG=$TAG CONC=${CONC:-1,8,32,64,128,256,512} sh /home/remote_bench_sweep.sh ${BENCH_ARGS:-}
cp "$LOG" /home/qwenfast-results/server-$TAG.log 2>/dev/null; echo DONE
