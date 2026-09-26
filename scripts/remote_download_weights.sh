#!/bin/sh
# Download checkpoints into the persistent /home/hf cache. MODELS overrides the list
# (default: Qwen3.8-27B FP8 then BF16).
set -e
export HF_HOME=/home/hf HF_HUB_ENABLE_HF_TRANSFER=1
python3 -m pip install -q -U "huggingface_hub[hf_transfer]" 
for m in ${MODELS:-Qwen/Qwen3.8-27B-FP8 Qwen/Qwen3.8-27B}; do
  echo "=== $(date -u) downloading $m"
  hf download "$m" --exclude "*.gguf" || huggingface-cli download "$m"
  echo "=== $(date -u) done $m"; du -sh /home/hf/hub
done
