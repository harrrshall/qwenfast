# qwenfast

a custom inference engine for `qwen3.8-27b`: 48 gated-deltanet layers, 16 gated gqa layers, an mtp head, and the fp8 block-128 checkpoint. it serves an openai-compatible api from a single gpu with cuda-graphed decode, continuous batching with chunked prefill, and mtp speculative decoding. the design is described in [docs/architecture.md](../../docs/architecture.md).

## layout

```
engine/qwenfast/
├── model.py          pure-pytorch reference model: the correctness oracle
├── weights.py        config parsing, sharded safetensors loading, block-128 fp8 dequant
├── verify_vs_hf.py   teacher-forced parity gate against hugging face transformers
├── bench_m0.py       reference decode-step benchmark against the bandwidth ceiling
├── kernels_gdn/      gated-deltanet kernels: pool-indexed decode, causal conv, verify-and-commit
├── gemm/             fused fp8 weights, per-m-bucket gemm dispatch, autotune
├── attn/             paged kv pool, flashinfer decode/prefill wrappers, rope
├── runtime/          fused model, cuda graphs, scheduler, speculative decoding, `serve`
├── server/           fastapi app, openai protocol, tokenization, metrics, auth
└── tests/            cpu unit tests for the reference model
```

each subpackage has its own readme: [kernels_gdn](kernels_gdn/README.md), [gemm](gemm/README.md), [attn](attn/README.md), [runtime](runtime/README.md), [server](server/README.md).

the package is imported as `qwenfast`, so every command below runs with `PYTHONPATH=engine` from the repository root.

## environment

- python 3.10 or newer
- `torch` with cuda, `triton`, `safetensors`, `transformers`, `huggingface_hub`
- `flashinfer-python` for paged attention (required by the cuda-graphed decode path)
- `fla-core` (flash-linear-attention), optional: the chunked prefill kernel and the reference decode fallback
- `vllm`, optional: provides the marlin, cutlass, deepgemm and machete gemm backends; without it the dispatcher falls through to `flashinfer_fp8_blockscale` and the torch fallbacks
- `fastapi`, `uvicorn`, `pydantic` for the server; `pyzmq` for `--engine-process`
- `pytest` and `pytest-asyncio` for the test suites

check the stack:

```bash
python -c "import torch, triton, flashinfer; print(torch.__version__, torch.version.cuda, triton.__version__, flashinfer.__version__)"
python -c "import fla; print('fla', fla.__version__)" || echo "fla not installed: torch gdn fallback"
```

flashinfer and deepgemm jit-compile kernels with the cuda toolkit on `PATH`; the nvcc and ptxas versions must match the toolkit torch was built against.

## checkpoint

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.8-27B-FP8', local_dir='/path/to/Qwen3.8-27B-FP8')
"
```

every `--model` argument accepts a snapshot directory, a `models--*` hub cache directory (the newest snapshot is picked), or a glob. the fp8 repository shards by layer name; the loader reads `model.safetensors.index.json` and falls back to globbing, so shard names are never hardcoded.

## run the server

```bash
PYTHONPATH=engine python -m qwenfast.runtime.serve \
  --model /path/to/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b \
  --preset fastest --spec-k 3 --spec-max-batch 16 \
  --host 0.0.0.0 --port 8000
```

`--preset fastest` fills every knob left at its default from `runtime/preset.py::CANONICAL_FAST` (fp16 ssm state, bf16 kv, triton norms and fused ops, per-m-bucket gemm dispatch, cuda graphs on); explicit flags always win. the flags that matter most:

| flag | effect |
|---|---|
| `--spec-k {0,1,2,3}` | mtp speculative decoding draft length (0 = off) |
| `--spec-max-batch N` | largest decode batch the speculative step is used at |
| `--max-num-seqs N` | ssm slots, which is the concurrency cap |
| `--max-model-len N` | prompt plus completion bound; sizes the kv pool |
| `--max-num-batched-tokens N` | chunked prefill budget per step |
| `--prefill-chunk-tokens N` | cap a chunk below the budget to bound tail latency |
| `--mixed-forward` / `--mixed-graphs` | one forward per step over prefill chunk and decode rows, optionally cuda-graphed |
| `--async-scheduling` | schedule step n+1 on the host while step n runs |
| `--engine-process` | run the engine loop in its own process over zeromq |
| `--ssm-state-dtype {fp32,fp16}` | stored recurrent state precision |
| `--kv-cache-dtype {bf16,fp8}` | kv pool precision |
| `--gemm-priority {v9,v8,v7,v4}` | gemm cold-start priority table |
| `--gemm-accuracy {fast,strict}` | restrict gemm dispatch to w8a16 backends |
| `--fused-cache DIR` | boot from a saved `fused_weights.safetensors` |
| `--gpu-memory-utilization F` | fraction of free hbm the memory plan may use |
| `--no-graphs` | eager decode, debug only |
| `--api-key KEY` | require `Authorization: Bearer` on `/v1/*` |

`python -m qwenfast.runtime.serve --help` lists everything. the server prints its memory plan at startup and refuses to start when the plan exceeds the budget. endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` (prometheus), `/status`; see [docs/api.md](../../docs/api.md).

`python -m qwenfast.server --engine mock` runs the same app against a fake engine with no gpu, for client and protocol work.

## parity check

`verify_vs_hf.py` compares the reference model against hugging face `transformers`, teacher-forced: our model greedily generates a fixed-length continuation for five chat prompts, the identical `prompt + continuation` is scored by both stacks in one forward, and per-position logit differences and top-1 agreement decide the verdict. free-running text is printed for information only.

```bash
PYTHONPATH=engine python engine/qwenfast/verify_vs_hf.py \
  --model /path/to/Qwen3.8-27B-FP8 --device cuda:0 --json-out parity.json
```

exit code 0 is pass. thresholds are `--logit-tol` (default 0.35) and `--agree-tol` (default 0.97). `--hf-model` points transformers at a different snapshot, `--no-fla` forces the pure-torch gdn path, `--kv-cache-dtype fp8` exercises the fp8 kv path, `--residency both` keeps both models resident (two bf16 copies need about 108 gib).

## reference decode benchmark

`bench_m0.py` times the eager reference model's decode step and prints it next to the analytic bandwidth ceiling (`efficiency = ceiling / measured`).

```bash
PYTHONPATH=engine python engine/qwenfast/bench_m0.py \
  --model /path/to/Qwen3.8-27B-FP8 --batch 1 32 --prompt-len 512 --steps 32 --json-out bench_m0.json
```

flags: `--state-dtype {fp32,fp16,bf16}`, `--bandwidth-tbs` (4.8 for h200), `--no-fla`, `--warmup`. the reference model is one kernel launch per op, so single-digit efficiency is expected; the runtime benchmarks in `runtime/bench_runtime.py` and `runtime/bench_spec.py` measure the real engine.

## tests

cpu suites, no gpu needed:

```bash
python engine/qwenfast/tests/test_ops.py                     # reference ops vs the hf modeling code
python engine/qwenfast/kernels_gdn/tests/test_kernels_gdn.py # torch backend vs the oracle, pool layout, verify-and-commit
python engine/qwenfast/gemm/tests/test_gemm.py               # fusion math, dispatch order, save/load
python engine/qwenfast/attn/tests/test_attn.py               # page allocator, kv pool, rope, attention parity
pytest engine/qwenfast/runtime/tests engine/qwenfast/server/tests
```

`tests/test_ops.py` pins the conventions that are easy to get subtly wrong: `RMSNorm` scales by `(1 + weight)` and `RMSNormGated` by plain `weight`; text-only mrope equals standard rope over 64 dims; the chunked gated delta rule equals the recurrent one and threads `initial_state` across a split; left padding cannot leak into gdn state; a synthetic checkpoint round-trips through `weights.py` in bf16 and block-128 fp8; and the teacher-forced slicing in `verify_vs_hf.py` is aligned on both sides.

gpu-only tests in every suite skip themselves when cuda is unavailable and run for real on a gpu box with the same commands.

## library use

the reference model:

```python
import torch
from qwenfast import QwenFastForCausalLM, Generator

model = QwenFastForCausalLM.from_pretrained(
    "/path/to/Qwen3.8-27B-FP8", device="cuda:0", dtype=torch.bfloat16, with_mtp=False,
)
gen = Generator(model)
out = gen.generate([[1, 2, 3], [4, 5]], max_new_tokens=32)   # list[list[int]]
```

for chat, build the prompt with the hugging face tokenizer:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/path/to/Qwen3.8-27B-FP8")
ids = tok.apply_chat_template(
    [{"role": "user", "content": "hello"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
)
print(tok.decode(gen.generate([ids], max_new_tokens=64)[0]))
```

the fused runtime, without the server:

```python
from qwenfast.runtime import RuntimeConfig, SpecConfig, build_engine

comps = build_engine("/path/to/Qwen3.8-27B-FP8", RuntimeConfig(ssm_state_dtype="fp16"),
                     fused_cache="/path/to/fused-cache", spec=SpecConfig(k=2))
comps.decoder.warmup(); comps.decoder.capture()
```

`build_engine` returns `EngineComponents` (`model`, `buf`, `decoder`, `rt`, optional `spec`), the pieces the benchmarks drive directly.

## conventions worth knowing

- `q_proj` already contains the attention output gate: it is `[12288, 5120]`, laid out `[h0_q | h0_gate | h1_q | h1_gate | ...]`.
- two rmsnorms: `RMSNorm` scales by `(1 + weight)`, the gdn output norm `RMSNormGated` by plain `weight`.
- gdn has 48 value heads and 16 key heads; the kernels take q/k in 16-head form and map value head `hv` to key head `hv // 3`.
- the recurrent state is `[48, 128, 128]` per layer per sequence, 144 mib per sequence in fp32 across the 48 layers. that, and the kv cache second, bounds concurrency.
- `in_proj_a`, `in_proj_b`, `embed_tokens`, `lm_head` and `mtp.fc` are bf16 in the fp8 checkpoint; `weight_scale_inv` is bf16 and is upcast on load.
- the mtp head is a full-attention layer with no gdn state, shares `embed_tokens` and `lm_head`, and uses kv layer index 16 of the 17-layer pool. its `fc` input order is `[embedding ; hidden]`; `mtp_hidden_first=True` (`--mtp-hidden-first`) flips it.
