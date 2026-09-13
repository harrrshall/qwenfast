#!/bin/sh
# Persistent qwenfast demo server (foreground; keep the session alive). OpenAI-compatible on :8000.
. /home/venv_vllm/bin/activate; export PYTHONPATH=/home/engine HF_HOME=/home/hf HF_HUB_OFFLINE=1
export CUDA_HOME=/home/venv_vllm/lib/python3.10/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FP8=$(ls -d /home/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1); cd /home
sh /home/remote_gpu_free.sh
echo "===== $(date -u) DEMO SERVER starting"
exec python -m qwenfast.runtime.serve --model "$FP8" --served-model-name qwen3.8-27b --preset fastest \
  --spec-k 3 --spec-max-batch 16 --mixed-forward --prefill-chunk-tokens 1024 \
  --max-num-seqs 64 --max-model-len 8192 --host 0.0.0.0 --port 8000 ${DEMO_EXTRA:-}
