#!/usr/bin/env python3
"""Merge the four individual benchmark JSON/Markdown outputs into one
combined `microbench-<ts>.json` + `.md`, as used by `run_all.sh`.

Not itself a GPU-touching script -- pure file I/O, safe to run anywhere
(including this Mac, for testing the merge logic against fixture JSON).

Usage:
    python merge_results.py \\
        --gdn-decode gdn_decode.json --gdn-prefill gdn_prefill.json \\
        --attn-decode attn_decode.json --gemm gemm.json \\
        --out /home/qwenfast-results/microbench-<timestamp>
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys


def _load(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_md(json_path: str | None) -> str:
    if not json_path:
        return ""
    md_path = json_path[: -len(".json")] + ".md" if json_path.endswith(".json") else json_path + ".md"
    if not os.path.exists(md_path):
        return f"_(no markdown found at {md_path})_\n"
    with open(md_path) as f:
        return f.read()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gdn-decode", type=str, default=None)
    p.add_argument("--gdn-prefill", type=str, default=None)
    p.add_argument("--attn-decode", type=str, default=None)
    p.add_argument("--gemm", type=str, default=None)
    p.add_argument("--out", type=str, required=True,
                    help="Base output path (no extension); writes <out>.json and <out>.md.")
    args = p.parse_args(argv)

    sections = {
        "gdn_decode_bench": args.gdn_decode,
        "gdn_prefill_bench": args.gdn_prefill,
        "attn_decode_bench": args.attn_decode,
        "gemm_bench": args.gemm,
    }

    combined = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sources": {k: v for k, v in sections.items() if v},
    }
    missing = []
    for key, path in sections.items():
        data = _load(path)
        if data is None:
            missing.append(key)
            combined[key] = {"status": "missing", "expected_path": path}
        else:
            combined[key] = data
    combined["missing_sections"] = missing

    md_parts = [
        "# Qwen3.8-27B Gated-DeltaNet kernel microbenchmark suite -- combined results",
        f"Generated: {combined['generated_utc']}",
    ]
    if missing:
        md_parts.append(f"**Missing sections (not run or failed before writing output): {', '.join(missing)}**")
    for key, path in sections.items():
        md_parts.append("\n---\n\n" + _load_md(path))

    json_path = args.out if args.out.endswith(".json") else args.out + ".json"
    md_path = (args.out[: -len(".json")] if args.out.endswith(".json") else args.out) + ".md"
    os.makedirs(os.path.dirname(os.path.abspath(json_path)) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    with open(md_path, "w") as f:
        f.write("\n".join(md_parts) + "\n")

    print(f"[merge_results] wrote {json_path} and {md_path}")
    if missing:
        print(f"[merge_results] WARNING: missing sections: {missing}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
