#!/bin/sh
# Kill our own servers/benches and wait until the GPU is (nearly) free. Usage: sh /home/remote_gpu_free.sh
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; pkill -f "qwenfast.runtime.serve" 2>/dev/null; pkill -f "qwenfast.server" 2>/dev/null; pkill -f bench_serve 2>/dev/null; sleep 5; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 4000 ] && { echo "GPU free (used ${used} MiB) after $((i*3))s"; exit 0; }
  sleep 3
done
echo "WARNING: GPU still has ${used} MiB used after 180s"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
