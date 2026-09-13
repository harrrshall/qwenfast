#!/bin/sh
# Quick correctness + single-stream speed check against a local OpenAI-compatible server.
BASE=${BASE:-http://localhost:8000/v1}; MODEL=${MODEL:-qwen3.8-27b}
curl -s $BASE/models | head -c 300; echo
for think in false true; do
  echo "--- enable_thinking=$think"
  start=$(date +%s.%N)
  curl -s $BASE/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Then write one sentence about Chennai.\"}],\"max_tokens\":300,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":$think}}" > /tmp/resp.json
  end=$(date +%s.%N)
  python3 - "$start" "$end" <<'PY'
import json,sys
d=json.load(open('/tmp/resp.json')); u=d.get('usage',{}); m=d['choices'][0]['message']
dt=float(sys.argv[2])-float(sys.argv[1])
print('reasoning:',(m.get('reasoning_content') or m.get('reasoning') or '')[:200].replace('\n',' '))
print('content:',(m.get('content') or '')[:300].replace('\n',' '))
print(f"usage={u} wall={dt:.2f}s ~{u.get('completion_tokens',0)/dt:.1f} tok/s (incl. prefill)")
PY
done
