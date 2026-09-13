#!/bin/sh
# Baseline vLLM server: FP8 weights, text-only, otherwise defaults.
export HF_HOME=/home/hf HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=INFO VLLM_USE_DEEP_GEMM=0
. /home/venv_vllm/bin/activate
exec python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
  --max-model-len 65536 --max-num-seqs 256 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 --port 8000 ${VLLM_EXTRA_ARGS}
