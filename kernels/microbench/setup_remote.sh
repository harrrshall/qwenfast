#!/bin/sh
# Install the optional GPU-kernel dependencies used by the microbenchmark
# suite into the existing vLLM venv (/home/venv_vllm, per
# scripts/remote_install_vllm.sh). Every install is best-effort: a failure is
# logged and the script continues, since every benchmark script in this
# directory already degrades gracefully (records the error, keeps going) when
# a given optional package isn't importable.
#
# Installs, in order:
#   1. flash-linear-attention (fla)   -- pure-Python + Triton, should always work
#   2. causal-conv1d                  -- CUDA extension, MAY NEED A BUILD; skip on failure
#   3. cuLA (inclusionAI)             -- CUDA/CuTe extension, early-stage; skip on failure
#
# Known environment wrinkle: a GPU host's system nvcc may be CUDA 12.6 while
# torch is built for cu13 (`torch.version.cuda` reports 13.x). A DeepGEMM JIT
# build fails with "NVCC compilation failed" for exactly this reason, and
# VLLM_USE_DEEP_GEMM=0 only sidesteps the JIT rather than fixing the mismatch.
# This script instead ATTEMPTS the real fix for any package that needs to be
# built from source here: `pip install nvidia-cuda-nvcc-cu13` (a wheel-packaged
# CUDA-13 nvcc) and point CUDA_HOME at its install directory, so `nvcc`
# matches torch's CUDA version for any `setup.py build_ext` / ninja build.
# If that package isn't available or doesn't produce a working nvcc, we fall
# back to the system CUDA Toolkit (12.6) and log a clear warning that
# extension builds may fail or mis-link -- this is exactly the DeepGEMM
# failure mode, expected to also hit causal-conv1d/cuLA source builds.
#
# Usage (on the GPU host):
#   ./setup_remote.sh
#   VENV=/home/venv_vllm ./setup_remote.sh
#
# Never run this locally (macOS, no CUDA) -- there is nothing for it to do.

set -u

VENV="${VENV:-/home/venv_vllm}"
LOG_DIR="${LOG_DIR:-/home/qwenfast-results}"
CULA_SRC_DIR="${CULA_SRC_DIR:-$(dirname "$VENV")/cuLA_src}"
MAX_JOBS="${MAX_JOBS:-4}"
export MAX_JOBS

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/setup_remote-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "Logging full output to $LOG"

run_logged() {
  echo "+ $*"
  echo "+ $*" >> "$LOG"
  # shellcheck disable=SC2068
  "$@" >>"$LOG" 2>&1
  rc=$?
  tail -n 20 "$LOG" | sed 's/^/    /'
  return $rc
}

echo "=== [0/5] venv ==="
if [ ! -d "$VENV" ]; then
  echo "venv $VENV not found; creating it (expected to already exist per scripts/remote_install_vllm.sh)"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python3 -c "import sys; print('python', sys.version)"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
  || echo "WARNING: torch not importable in $VENV -- run scripts/remote_install_vllm.sh first"

echo ""
echo "=== [1/5] nvcc / CUDA_HOME situation ==="
if command -v nvcc >/dev/null 2>&1; then
  echo "system nvcc found: $(command -v nvcc)"
  nvcc --version | tail -4
else
  echo "no system nvcc on PATH"
fi

echo "--- attempting pip install nvidia-cuda-nvcc-cu13 (matches torch's cu13 build) ---"
run_logged pip install -q nvidia-cuda-nvcc-cu13
NVCC_CU13_DIR=$(python3 -c "
try:
    import nvidia.cuda_nvcc as m, os
    print(os.path.dirname(m.__file__))
except Exception:
    pass
" 2>/dev/null)

if [ -n "$NVCC_CU13_DIR" ] && [ -x "$NVCC_CU13_DIR/bin/nvcc" ]; then
  export CUDA_HOME="$NVCC_CU13_DIR"
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "CUDA_HOME set to $CUDA_HOME (nvidia-cuda-nvcc-cu13 wheel; matches torch cu13)"
  "$CUDA_HOME/bin/nvcc" --version | tail -4
else
  echo "nvidia-cuda-nvcc-cu13 unavailable or produced no working nvcc binary."
  if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH="$CUDA_HOME/bin:$PATH"
    echo "Falling back to CUDA_HOME=$CUDA_HOME (system CUDA Toolkit, likely 12.6)."
  fi
  echo "WARNING: system nvcc (12.6) vs. torch's cu13 build is a KNOWN mismatch" \
       "(DeepGEMM JIT fails with this exact" \
       "combination). CUDA-extension builds below (causal-conv1d, cuLA) may fail or mis-link" \
       "for the same reason -- failures are logged and skipped, not fatal to this script."
fi

echo ""
echo "=== [2/5] flash-linear-attention (fla) -- pure Python + Triton, should not need nvcc ==="
run_logged pip install -q -U flash-linear-attention
if python3 -c "import fla" 2>/dev/null; then
  python3 -c "import fla; print('fla OK, version', getattr(fla, '__version__', '?'))"
else
  echo "fla import FAILED after install attempt -- gdn_decode_bench.py / gdn_prefill_bench.py " \
       "will record this as an error per-variant and fall back to the torch reference paths."
fi

echo ""
echo "=== [3/5] causal-conv1d -- CUDA extension, MAY NEED A BUILD ==="
run_logged pip install -q causal-conv1d --no-build-isolation
if python3 -c "import causal_conv1d" 2>/dev/null; then
  python3 -c "import causal_conv1d; print('causal_conv1d OK, version', getattr(causal_conv1d, '__version__', '?'))"
else
  echo "causal-conv1d install/import FAILED -- SKIPPING (allowed failure)." \
       "gdn_decode_bench.py's causal_conv1d_update benchmark automatically falls back to a" \
       "pure-torch F.conv1d implementation (same math, ported from the HF reference) and labels" \
       "the result impl='torch_fallback' so this is visible in the output, not silently wrong."
fi

echo ""
echo "=== [4/5] cuLA (inclusionAI) -- early-stage CUDA/CuTe extension, BEST EFFORT ==="
echo "--- try 1: pip install cula ---"
run_logged pip install -q cula
python3 -c "import cula" 2>/dev/null
CULA_OK=$?

if [ $CULA_OK -ne 0 ]; then
  echo "--- try 2: pip install cuda-linear-attention (the pypi distribution name in cuLA's README) ---"
  run_logged pip install -q cuda-linear-attention
  python3 -c "import cula" 2>/dev/null
  CULA_OK=$?
fi

if [ $CULA_OK -ne 0 ]; then
  echo "--- try 3: source build (git clone + submodules + pip install -e, per cuLA's README) ---"
  echo "Requires CUDA Toolkit/nvcc >= 12.9 and torch >= 2.9.1 with a MATCHING system CUDA Toolkit" \
       "version -- likely to fail on this host given the nvcc/torch mismatch noted in step [1/5]." \
       "That is an ALLOWED failure: logged below, then skipped."
  rm -rf "$CULA_SRC_DIR"
  run_logged git clone --depth 1 https://github.com/inclusionAI/cuLA.git "$CULA_SRC_DIR"
  if [ -d "$CULA_SRC_DIR" ]; then
    (
      cd "$CULA_SRC_DIR" || exit 1
      run_logged git submodule update --init --recursive
      run_logged pip install -q -e third_party/flash-linear-attention
      run_logged pip install -q -e . --no-build-isolation
    )
    python3 -c "import cula" 2>/dev/null
    CULA_OK=$?
  fi
fi

if [ $CULA_OK -eq 0 ]; then
  python3 -c "import cula; print('cula OK, version', getattr(cula, '__version__', '?'))"
else
  echo "cuLA install FAILED (all 3 methods) -- SKIPPING (allowed failure, logged above and in $LOG)." \
       "gdn_decode_bench.py's cuLA probe will record status=unavailable and the rest of the suite" \
       "runs unaffected. Note cuLA's public API (at the time of writing)" \
       "exposes KDA (Kimi Delta Attention) and Lightning Attention, not a confirmed drop-in for" \
       "fla's gated_delta_rule -- even a successful build may not change gdn_decode_bench's cuLA" \
       "row from 'unavailable' to 'ok'. See gdn_decode_bench.py's _CULA_CANDIDATES for exactly" \
       "which entrypoints are probed."
fi

echo ""
echo "=== [5/5] flashinfer / vllm sanity check (should already be present per scripts/remote_install_vllm.sh) ==="
python3 -c "import flashinfer; print('flashinfer OK, version', flashinfer.__version__)" \
  || { echo "flashinfer missing -- installing flashinfer-python"; run_logged pip install -q flashinfer-python; }
python3 -c "import vllm; print('vllm OK, version', vllm.__version__)" \
  || echo "vllm missing -- run scripts/remote_install_vllm.sh first (gemm_bench.py's fp8_block128" \
          "and fp8_cutlass variants need it; bf16/fp8_pertensor still work without it)."

echo ""
echo "=== summary ==="
for pkg in torch fla causal_conv1d cula flashinfer vllm; do
  python3 -c "
import importlib
try:
    m = importlib.import_module('$pkg')
    print('$pkg'.ljust(16), 'OK', getattr(m, '__version__', '?'))
except Exception as e:
    print('$pkg'.ljust(16), 'MISSING:', type(e).__name__, str(e)[:120])
"
done
echo ""
echo "Full log: $LOG"
echo "Next: cd $(dirname "$0") && ./run_all.sh"
