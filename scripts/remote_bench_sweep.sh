#!/bin/sh
# Standard benchmark sweep against the running local server. Usage: TAG=<tag> sh remote_bench_sweep.sh [extra bench args]
export HF_HOME=/home/hf HF_HUB_OFFLINE=1
. /home/venv_vllm/bin/activate
TAG=${TAG:-untagged}; mkdir -p /home/qwenfast-results
cd /home/benchmarks && python bench_serve.py --base-url http://localhost:8000/v1 --model qwen3.8-27b \
  --tokenizer /home/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ \
  --concurrency ${CONC:-1,8,32,64,128,256,512} --input-len 2000 --output-len 500 --dataset random --nvidia-smi \
  --tag "$TAG" --out /home/qwenfast-results/bench-$TAG.json "$@"
