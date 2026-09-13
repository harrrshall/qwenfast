#!/bin/sh
# Run the full Gated-DeltaNet / Qwen3.8-27B kernel microbenchmark suite and
# write combined results to /home/qwenfast-results/microbench-<ts>.json + .md
# (plus the four individual per-script JSON/MD files alongside it).
#
# Non-interactive, safe to run unattended. Each underlying python script
# never hard-crashes on a missing optional dependency (fla / causal_conv1d /
# flashinfer / vllm / cula) -- it records the error string for that variant
# in the JSON and keeps going, so this script always produces *some* output
# as long as CUDA itself is available.
#
# Usage (remote, inside /home/venv_vllm):
#   ./run_all.sh
#   MICROBENCH_OUT_DIR=/tmp/mb WARMUP=5 ITERS=20 ./run_all.sh   # quick smoke run
#
# Env vars (all optional):
#   MICROBENCH_OUT_DIR   default /home/qwenfast-results
#   WARMUP                default 20  (gdn_decode / attn_decode / gemm)
#   ITERS                  default 100 (gdn_decode / attn_decode / gemm)
#   PREFILL_WARMUP         default 10
#   PREFILL_ITERS           default 30
#   VENV                    default /home/venv_vllm

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR="${MICROBENCH_OUT_DIR:-/home/qwenfast-results}"
VENV="${VENV:-/home/venv_vllm}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
PREFILL_WARMUP="${PREFILL_WARMUP:-10}"
PREFILL_ITERS="${PREFILL_ITERS:-30}"

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_TAG="microbench-${TS}"

mkdir -p "$OUT_DIR"

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  echo "=== activated venv: $VENV ==="
else
  echo "=== WARNING: $VENV/bin/activate not found, using whatever python is on PATH ($(command -v python3)) ==="
fi

cd "$SCRIPT_DIR" || exit 1

echo "=== env ==="
python3 -c "import sys; print('python', sys.version)"
for pkg in torch fla causal_conv1d flashinfer vllm cula; do
  python3 -c "import importlib; m=importlib.import_module('$pkg'); print('$pkg', getattr(m,'__version__','?'))" 2>&1 | tail -1
done

OVERALL_STATUS=0

echo "=== [1/4] gdn_decode_bench.py ==="
python3 gdn_decode_bench.py --out "$OUT_DIR/${RUN_TAG}-gdn_decode_bench" --warmup "$WARMUP" --iters "$ITERS"
[ $? -ne 0 ] && { echo "!!! gdn_decode_bench.py exited non-zero"; OVERALL_STATUS=1; }

echo "=== [2/4] gdn_prefill_bench.py ==="
python3 gdn_prefill_bench.py --out "$OUT_DIR/${RUN_TAG}-gdn_prefill_bench" --warmup "$PREFILL_WARMUP" --iters "$PREFILL_ITERS"
[ $? -ne 0 ] && { echo "!!! gdn_prefill_bench.py exited non-zero"; OVERALL_STATUS=1; }

echo "=== [3/4] attn_decode_bench.py ==="
python3 attn_decode_bench.py --out "$OUT_DIR/${RUN_TAG}-attn_decode_bench" --warmup "$WARMUP" --iters "$ITERS"
[ $? -ne 0 ] && { echo "!!! attn_decode_bench.py exited non-zero"; OVERALL_STATUS=1; }

echo "=== [4/4] gemm_bench.py ==="
python3 gemm_bench.py --out "$OUT_DIR/${RUN_TAG}-gemm_bench" --warmup "$WARMUP" --iters "$ITERS"
[ $? -ne 0 ] && { echo "!!! gemm_bench.py exited non-zero"; OVERALL_STATUS=1; }

echo "=== merging ==="
python3 merge_results.py \
  --gdn-decode "$OUT_DIR/${RUN_TAG}-gdn_decode_bench.json" \
  --gdn-prefill "$OUT_DIR/${RUN_TAG}-gdn_prefill_bench.json" \
  --attn-decode "$OUT_DIR/${RUN_TAG}-attn_decode_bench.json" \
  --gemm "$OUT_DIR/${RUN_TAG}-gemm_bench.json" \
  --out "$OUT_DIR/${RUN_TAG}"

echo "=== done ==="
echo "Combined:   $OUT_DIR/${RUN_TAG}.json"
echo "            $OUT_DIR/${RUN_TAG}.md"
echo "Per-script: $OUT_DIR/${RUN_TAG}-{gdn_decode_bench,gdn_prefill_bench,attn_decode_bench,gemm_bench}.{json,md}"

exit $OVERALL_STATUS
