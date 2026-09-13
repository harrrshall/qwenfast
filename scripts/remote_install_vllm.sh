#!/bin/sh
# Create /home/venv_vllm with the latest vLLM (baseline engine for the benchmarks).
set -e
cd /home
python3 -m venv venv_vllm
. venv_vllm/bin/activate
pip install -q -U pip
pip install -U vllm
pip install -U "flash-linear-attention" causal-conv1d 2>&1 | tail -3 || true
python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,torch.version.cuda)"
pip list | grep -iE "vllm|fla|flash|triton|torch|transformers|flashinfer|causal"
