#!/bin/sh
# Reference evals on vLLM's best config (FP8 + DeepGEMM), same evals/run_eval.py settings as the qwenfast eval run.
. /home/venv_vllm/bin/activate; export HF_HOME=/home/hf HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=1 CUDA_HOME=/home/venv_vllm/lib/python3.10/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$PATH
R=/home/qwenfast-results; cd /home
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 5; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 3
nohup python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b --max-model-len 8192 --max-num-seqs 256 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes --host 0.0.0.0 --port 8000 > $R/server-vllm-eval.log 2>&1 &
for i in $(seq 1 90); do sleep 10; curl -s -m 3 localhost:8000/v1/models >/dev/null 2>&1 && { echo "READY after ~$((i*10))s"; break; }; done
curl -s -m 3 localhost:8000/v1/models >/dev/null || { echo "SERVER TIMEOUT"; exit 1; }
python /home/evals/run_eval.py --base-url http://localhost:8000/v1 --model qwen3.8-27b --concurrency 32 --enable-thinking false --max-tokens 512 --temperature 0 --tag vllm-deepgemm --out $R/eval-vllm-deepgemm.json 2>&1 | tail -15
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null; sleep 5; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
echo "VLLM EVAL DONE"
