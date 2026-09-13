# evals

quality gate for changes to the engine, the quantization path or speculative decoding. it runs two small fixed suites against any openai-compatible chat-completions server and writes a result json with every raw response, so two runs can be diffed item by item. the parity numbers in [`docs/benchmarks.md`](../docs/benchmarks.md) come from this directory.

## files

| path | purpose |
|---|---|
| `run_eval.py` | runs gsm8k-200 and ifeval-50 against a live server; diffs two saved runs |
| `ifeval_checkers.py` | the programmatic checkers for the ifeval prompts (stdlib only) |
| `logit_check.py` | offline comparison of two engines' saved next-token logits |
| `data/gsm8k_200.jsonl` | 200 fixed gsm8k test problems |
| `data/ifeval_50.jsonl` | 50 instruction-following prompts with their checker spec |
| `tests/mock_server.py` | deterministic mock server for testing the harness without a gpu |
| `requirements.txt` | `aiohttp` for server calls, `numpy` for `logit_check.py` |

## data provenance

`data/gsm8k_200.jsonl` was built by downloading the official gsm8k test set (`https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`, 1319 rows) and taking the first 200 rows in file order (index 0 to 199, ids `gsm8k_0000` to `gsm8k_0199`). each row has `id`, `question`, `answer` (the gold numeric answer, comma-free) and `full_solution` (the original chain of thought with its `####` marker, kept for reference and unused in grading). re-running the same download and truncation reproduces the file exactly.

`data/ifeval_50.jsonl` is hand-authored and is a small stand-in for google's ifeval. each row has `id`, `prompt`, `checker` (a function name in `ifeval_checkers.py`) and `checker_args`. five checker families with about ten prompts each: `bullet_count`, `all_caps`, `all_lower`, `word_count_at_least`, `json_keys`.

## running

```bash
pip install -r evals/requirements.txt

python evals/run_eval.py \
  --base-url http://localhost:8000/v1 --model qwen3.8-27b \
  --concurrency 32 --enable-thinking false --max-tokens 512 --temperature 0 \
  --out result.json --tag <tag>
```

flags:

| flag | default | meaning |
|---|---|---|
| `--base-url` | required | server base url ending in `/v1` |
| `--model` | required | model name as registered on the server |
| `--api-key` | `$OPENAI_API_KEY` | bearer token if the server checks one |
| `--concurrency` | `32` | requests in flight, shared by both suites |
| `--enable-thinking` | server default | `true` or `false`, sets `chat_template_kwargs.enable_thinking` |
| `--max-tokens` | `512` | completion budget per request |
| `--temperature` | `0.0` | sampling temperature |
| `--timeout` | `120` | per-request timeout in seconds |
| `--retries` | `2` | retries per request |
| `--gsm8k-n`, `--ifeval-n` | all | use only the first n items of a suite, for smoke tests |
| `--reference PATH` | none | after the run, print per-item agreement against a saved result |
| `--compare-only A B` | none | skip the server and diff two saved results |
| `--out` | `result.json` | result path |
| `--tag` | empty | label stored in the result |

both suites run concurrently under one semaphore. the console summary looks like:

```
  model=qwen3.8-27b  base_url=http://localhost:8000/v1  enable_thinking=false  temperature=0.0
  gsm8k    n= 200  accuracy= 92.5%  mean_output_tokens=317.1  errors=0
  ifeval   n=  50  accuracy= 98.0%  mean_output_tokens=37.9  errors=0
  wall_time_s=73.4
```

## grading

gsm8k: each problem is sent as one user turn asking for a final line of the form `#### <number>`. grading strips any `<think>...</think>` block, takes the last `#### <number>` match, and falls back to the last standalone number in the text. prediction and gold are number-normalized (commas and `$` stripped, `3.0` becomes `3`) before an exact-match comparison.

ifeval: each response, with `<think>` blocks stripped, is passed to the checker named in its row. checkers are strict: `json_keys` requires exactly the given keys, `bullet_count` requires exactly n non-empty lines with the given marker. the gate exists to catch regressions in instruction following, so leniency is not a goal.

## output format

```json
{
  "meta": {"tag": "...", "base_url": "...", "model": "...", "enable_thinking": "false",
           "max_tokens": 512, "temperature": 0.0, "concurrency": 32,
           "wall_time_s": 73.4, "timestamp_utc": "..."},
  "gsm8k":  {"n": 200, "accuracy": 0.925, "mean_output_tokens": 317.1,
             "mean_output_words": 173.1, "errors": 0, "items": [...]},
  "ifeval": {"n": 50, "accuracy": 0.98, "mean_output_tokens": 37.9,
             "mean_output_words": 23.1, "errors": 0, "items": [...]}
}
```

each gsm8k item records `id`, `question`, `gold`, `pred`, `correct`, `output_tokens`, `latency_s`, `raw_output` and `error`; each ifeval item records `id`, `prompt`, `checker`, `passed` and the same trailing fields. `mean_output_tokens` comes from the server's `usage.completion_tokens` and `mean_output_words` is a fallback proxy that is always available.

## diffing two runs

`--reference` and `--compare-only` report, per suite, `exact_text_agreement_pct` (byte-identical raw responses) and `answer_agreement_pct` (gsm8k) or `pass_agreement_pct` (ifeval), plus `overall_exact_text_agreement_pct` across both suites. with `--reference` the report is also stored in the result under `agreement_vs_reference`.

suggested thresholds, with the previous known-good result as the baseline:

| check | threshold |
|---|---|
| gsm8k accuracy | at least baseline minus 2 points |
| ifeval accuracy | at least baseline minus 4 points |
| exact text agreement, speculative decoding on versus off, same engine, temperature 0 | at least 95 percent |
| exact text agreement, engine change against a bf16 reference | at least 99 percent for a bf16 to bf16 swap, at least 90 percent for fp8 weights or fp8 kv cache |

treat the accuracy thresholds as the hard gate and the agreement numbers as diagnostics that explain why a change passed or failed.

## logit_check.py

an offline check for numerical drift that text-level agreement can miss. dump next-token logits from two engines for the same fixed prompt set and compare:

```bash
python evals/logit_check.py --a logits_reference.npy --b logits_candidate.npy \
  --out logit_check_result.json --topk 5
```

input files are `float32` arrays of shape `(N, V)`: one row per prompt, raw pre-softmax logits over the vocabulary for the next token, produced with each engine in teacher-forced prefill mode. a shape mismatch is a hard error. the report contains `top1_agreement`, `topk_agreement`, kl divergence in both directions (mean, median, max), `max_abs_logit_diff` (overall max, mean and median of the per-row max), `n_mismatched_rows` and a capped sample of mismatched row indices. a same-engine rerun should show `top1_agreement` of at least 99 percent.

## self-test without a gpu

`tests/mock_server.py` mimics `/v1/chat/completions` deterministically: gsm8k prompts get `#### 42`, ifeval prompts get responses that mostly pass their checkers.

```bash
MOCK_MODE=good python3 evals/tests/mock_server.py --port 8123 &

python3 evals/run_eval.py --base-url http://127.0.0.1:8123/v1 --model mock-model \
  --out /tmp/run1.json --tag selftest1 --enable-thinking false --temperature 0
python3 evals/run_eval.py --base-url http://127.0.0.1:8123/v1 --model mock-model \
  --out /tmp/run2.json --tag selftest2 --reference /tmp/run1.json
python3 evals/run_eval.py --compare-only /tmp/run1.json /tmp/run2.json
```

the mock is deterministic, so the second run reports 100 percent agreement. `MOCK_MODE=nohash` answers without a `####` marker to exercise the fallback extraction, and `MOCK_MODE=badjson` returns malformed json for the `json_keys` prompts to confirm the checker marks them failed.
