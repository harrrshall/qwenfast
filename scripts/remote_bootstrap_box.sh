#!/bin/sh
# Idempotent bring-up for a fresh H200 host (a PyTorch image) to the engine's development
# environment. Safe to re-run at any point — every step checks whether it already succeeded
# before doing work.
#
# What this script does NOT do: mirror /home/engine, /home/benchmarks, /home/evals and the
# scripts/remote_*.sh files. Those live in the local repo and must be copied to the host
# from the operator's machine BEFORE or AFTER running this script. This script only handles
# what the host itself needs to build: HF weights, the vLLM venv, and the CUDA 13.0 toolchain
# fix inside it. It also creates /home/qwenfast-results if missing and runs the download
# script if present.
#
# Usage: sh /home/remote_bootstrap_box.sh
set -e

log() { echo "===== $(date -u +%H:%M:%S) $1"; }

# ---------------------------------------------------------------------------
# 0. Results dir
# ---------------------------------------------------------------------------
mkdir -p /home/qwenfast-results
log "results dir ready"

# ---------------------------------------------------------------------------
# 1. HF weights (idempotent: skip if a snapshot with safetensors already exists)
# ---------------------------------------------------------------------------
have_weights() {
  d=$(ls -d "/home/hf/hub/models--Qwen--Qwen3.8-27B$1/snapshots"/*/ 2>/dev/null | head -1)
  [ -n "$d" ] && ls "$d"*.safetensors >/dev/null 2>&1
}
if have_weights "-FP8" && have_weights ""; then
  log "weights already present, skipping download"
else
  if [ -f /home/remote_download_weights.sh ]; then
    log "downloading weights via remote_download_weights.sh"
    sh /home/remote_download_weights.sh
  else
    log "WARNING: /home/remote_download_weights.sh not found — upload it first, weights not fetched"
  fi
fi

# ---------------------------------------------------------------------------
# 2. venv_vllm with pinned, known-good versions
# ---------------------------------------------------------------------------
VENV=/home/venv_vllm
need_venv=1
if [ -x "$VENV/bin/python" ]; then
  cur=$("$VENV/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
  [ "$cur" = "0.28.0" ] && need_venv=0
fi

if [ "$need_venv" = "1" ]; then
  log "building venv_vllm"
  python3 -m venv "$VENV"
  . "$VENV/bin/activate"
  pip install -q -U pip
  # Pin the exact versions the engine was built and measured against.
  pip install -q \
    "vllm==0.28.0" \
    "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0" \
    "flashinfer-python==0.6.16.post3" \
    "fla-core==0.5.2" \
    "triton==3.7.1" \
    "transformers==5.16.1" \
    "accelerate==1.14.0" \
    "pytest==9.1.1" \
    "huggingface_hub[hf_transfer]==1.28.0" \
    "safetensors==0.8.0"
  python -c "import vllm,torch,flashinfer,fla,triton; print('vllm',vllm.__version__,'torch',torch.__version__,torch.version.cuda,'flashinfer',flashinfer.__version__,'fla',fla.__version__,'triton',triton.__version__)"
else
  log "venv_vllm already at vllm==0.28.0, skipping install"
  . "$VENV/bin/activate"
fi

# ---------------------------------------------------------------------------
# 3. CUDA 13.0 toolchain consistency fix
#    Bug: pip's default resolver pulls nvidia-nvvm (cicc) at a newer minor (e.g. 13.3) than
#    nvidia-cuda-nvcc/crt/runtime/cccl (13.0), so ptxas 13.0 rejects PTX .version 9.3 emitted
#    by cicc 13.3 -> every FlashInfer/DeepGEMM JIT compile fails. Fix: force these five
#    packages to the SAME 13.0.* build, --no-deps so pip doesn't drag torch/etc back in.
# ---------------------------------------------------------------------------
. "$VENV/bin/activate"
CU="$VENV/lib/python3.10/site-packages/nvidia/cu13"

log "pinning CUDA 13.0 toolchain (nvcc/nvvm/crt/runtime/cccl)"
pip install -q --no-deps --force-reinstall \
  "nvidia-cuda-nvcc==13.0.88" \
  "nvidia-nvvm==13.0.88" \
  "nvidia-cuda-crt==13.0.88" \
  "nvidia-cuda-runtime==13.0.96" \
  "nvidia-cuda-cccl==13.0.85"

log "toolchain versions now:"
pip freeze | grep -E "^(nvidia-cuda-nvcc|nvidia-nvvm|nvidia-cuda-crt|nvidia-cuda-runtime|nvidia-cuda-cccl)=="

# lib64 -> lib and the libcudart/libcublas/libcublasLt symlinks FlashInfer's GEMM JIT needs
# to link against at runtime.
mkdir -p "$CU/lib"
ln -sf lib "$CU/lib64"
cd "$CU/lib"
for base in libcudart libcublas libcublasLt; do
  tgt=$(ls "$base".so.* 2>/dev/null | sort -V | tail -1)
  if [ -n "$tgt" ]; then
    ln -sf "$tgt" "$base.so"
  else
    log "WARNING: no $base.so.* found to symlink"
  fi
done
cd /
log "cu13 symlinks:"
ls -la "$CU/lib64" "$CU"/lib/libcudart.so "$CU"/lib/libcublas.so "$CU"/lib/libcublasLt.so 2>&1

export CUDA_HOME="$CU"
export PATH="$CUDA_HOME/bin:$PATH"
log "nvcc check: $($CUDA_HOME/bin/nvcc --version 2>&1 | tail -1)"

log "BOOTSTRAP DONE"
