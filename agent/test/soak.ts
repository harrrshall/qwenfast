// soak test: keeps a running daemon busy for hours with verifiable tasks of every difficulty and
// reports success rate and latency per routed tier.
//
//   node test/soak.ts [--hours 2] [--every 5] [--batch 3]
//
// every task gets a fresh workspace under $QFA_HOME/workspace/soak/ and a verify command, so "done"
// means the check passed. progress lines go to stdout; the summary is written to
// $QFA_HOME/logs/soak-<start>.json as well.

import { mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const BASE = process.env.QFA_URL ?? "http://127.0.0.1:7788";
const HOME = process.env.QFA_HOME ?? join(homedir(), ".qwenfast-code", "agent");
const arg = (k: string, d: number) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > 0 ? Number(process.argv[i + 1]) : d;
};
const HOURS = arg("hours", 2);
const EVERY_MIN = arg("every", 5);
const BATCH = arg("batch", 3);

interface Spec {
  kind: "easy" | "medium" | "hard";
  files: Record<string, string>;
  prompt: string;
  verify: string;
}

function rnd(n: number): number {
  return Math.floor(Math.random() * n);
}

const SPECS: Array<() => Spec> = [
  () => {
    const n = 20 + rnd(80);
    return {
      kind: "easy",
      files: { "data.txt": Array.from({ length: n }, (_, i) => `row ${i} ${rnd(1000)}`).join("\n") + "\n" },
      prompt: "count the lines in data.txt and write just the number into count.txt",
      verify: `[ "$(tr -d ' \\n' < count.txt)" = "${n}" ]`,
    };
  },
  () => {
    const words = ["alpha", "beta", "gamma", "delta"];
    const w = words[rnd(words.length)];
    return {
      kind: "easy",
      files: { "notes.md": `# notes\nthe secret word is ${w}.\n` },
      prompt: "read notes.md and write the secret word, lowercase, alone, into answer.txt",
      verify: `grep -qx "${w}" answer.txt`,
    };
  },
  () => ({
    kind: "medium",
    files: { "mathx.py": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n" },
    prompt:
      "In mathx.py add is_prime(n) (False for n < 2), lcm(a, b) using gcd, and primes_below(n) returning a sorted list. " +
      "Write unittest tests for all of them in test_mathx.py and make sure they pass.",
    verify:
      "python3 -m unittest -q test_mathx && python3 -c \"from mathx import *; assert primes_below(20)==[2,3,5,7,11,13,17,19]; assert lcm(4,6)==12; assert not is_prime(1) and is_prime(97)\"",
  }),
  () => ({
    kind: "medium",
    files: {
      "inventory.py": "ITEMS = []\n\n\ndef add(name, qty):\n    ITEMS.append((name, qty))\n",
    },
    prompt:
      "Turn inventory.py into a small module with a class Inventory: add(name, qty) merges quantities for the same name, " +
      "remove(name, qty) raises ValueError when there is not enough, total() returns the sum of all quantities, and " +
      "to_json() returns a json string of {name: qty} sorted by name. Add unittest tests in test_inventory.py.",
    verify:
      "python3 -m unittest -q test_inventory && python3 -c \"import json; from inventory import Inventory as I; i=I(); i.add('b',2); i.add('a',1); i.add('b',3); assert i.total()==6; assert json.loads(i.to_json())=={'a':1,'b':5}\"",
  }),
  () => ({
    kind: "hard",
    files: {
      "search.py":
        "def first_at_least(xs, target):\n    \"\"\"index of the first element >= target in sorted xs, len(xs) if none\"\"\"\n" +
        "    lo, hi = 0, len(xs) - 1\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if xs[mid] <= target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo\n",
      "test_search.py":
        "import random, unittest\nfrom search import first_at_least\n\nclass T(unittest.TestCase):\n    def test_random(self):\n        for _ in range(2000):\n" +
        "            xs = sorted(random.randint(0, 20) for _ in range(random.randint(0, 12)))\n            t = random.randint(-2, 22)\n" +
        "            want = next((i for i, x in enumerate(xs) if x >= t), len(xs))\n            self.assertEqual(first_at_least(xs, t), want, (xs, t))\n\nif __name__ == '__main__':\n    unittest.main()\n",
    },
    prompt:
      "test_search.py fails intermittently. Find the root cause of every bug in search.py and fix the function so the " +
      "randomized test always passes. Do not change the test.",
    verify: "for i in 1 2 3; do python3 -m unittest -q test_search || exit 1; done",
  }),
  () => ({
    kind: "hard",
    files: {
      "ratelimit.py":
        "import time\n\nclass TokenBucket:\n    def __init__(self, rate, capacity, clock=time.monotonic):\n        self.rate = rate\n        self.capacity = capacity\n" +
        "        self.clock = clock\n        self.tokens = 0\n        self.last = clock()\n\n    def allow(self, cost=1):\n        now = self.clock()\n" +
        "        self.tokens = self.tokens + (now - self.last) * self.rate\n        if self.tokens >= cost:\n            self.tokens -= cost\n            return True\n        return False\n",
    },
    prompt:
      "ratelimit.py has a token bucket with several bugs: it starts empty instead of full, never caps tokens at capacity, " +
      "and misbehaves if the clock goes backwards. Fix them all, make cost larger than capacity always rejected, and write " +
      "deterministic unittest tests with a fake clock in test_ratelimit.py covering each bug.",
    verify:
      "python3 -m unittest -q test_ratelimit && python3 -c \"from ratelimit import TokenBucket as T\nt=[0.0]\nb=T(1,3,clock=lambda:t[0])\nassert all(b.allow() for _ in range(3)) and not b.allow()\nt[0]=100\nassert sum(b.allow() for _ in range(10))==3\nt[0]=50\nassert not b.allow(5)\"",
  }),
];

async function api(method: string, path: string, body?: unknown): Promise<any> {
  const r = await fetch(`${BASE}${path}`, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  return r.json();
}

const started = new Date();
const stamp = started.toISOString().replace(/[:.]/g, "-");
const submitted: Array<{ id: string; kind: string }> = [];

async function submitBatch(round: number): Promise<void> {
  for (let i = 0; i < BATCH; i++) {
    const spec = SPECS[rnd(SPECS.length)]();
    const dir = join(HOME, "workspace", "soak", `${stamp}-${round}-${i}`);
    mkdirSync(dir, { recursive: true });
    for (const [f, text] of Object.entries(spec.files)) writeFileSync(join(dir, f), text);
    const t = await api("POST", "/tasks", { prompt: spec.prompt, cwd: dir, verify: spec.verify, timeoutMinutes: 45 });
    submitted.push({ id: t.id, kind: spec.kind });
    console.log(`${new Date().toISOString()} submitted ${t.id} (${spec.kind})`);
  }
}

async function summary(final: boolean) {
  const rows = await Promise.all(submitted.map(async (s) => ({ ...s, t: await api("GET", `/tasks/${s.id}`) })));
  const by: Record<string, { n: number; done: number; failed: number; open: number; secs: number[]; tiers: Record<string, number>; attempts: number }> = {};
  for (const r of rows) {
    const b = (by[r.kind] ??= { n: 0, done: 0, failed: 0, open: 0, secs: [], tiers: {}, attempts: 0 });
    b.n++;
    if (r.t.status === "done") b.done++;
    else if (r.t.status === "failed") b.failed++;
    else b.open++;
    b.attempts += r.t.attempts.length;
    const first = r.t.attempts[0]?.tier ?? r.t.route?.tier;
    if (first) b.tiers[first] = (b.tiers[first] ?? 0) + 1;
    if (r.t.status === "done") b.secs.push((Date.parse(r.t.updatedAt) - Date.parse(r.t.createdAt)) / 1000);
  }
  const out = Object.fromEntries(
    Object.entries(by).map(([k, b]) => {
      const s = b.secs.sort((x, y) => x - y);
      return [k, { tasks: b.n, done: b.done, failed: b.failed, open: b.open, firstTier: b.tiers, attemptsPerTask: +(b.attempts / b.n).toFixed(2), p50s: s[Math.floor(s.length / 2)] ?? null, p90s: s[Math.floor(s.length * 0.9)] ?? null }];
    }),
  );
  console.log(`${new Date().toISOString()} ${final ? "FINAL" : "progress"} ${JSON.stringify(out)}`);
  if (final) writeFileSync(join(HOME, "logs", `soak-${stamp}.json`), JSON.stringify({ started, hours: HOURS, summary: out, tasks: rows.map((r) => ({ id: r.id, kind: r.kind, status: r.t.status, attempts: r.t.attempts })) }, null, 2));
}

const end = Date.now() + HOURS * 3600_000;
let round = 0;
while (Date.now() < end) {
  await submitBatch(round++);
  await summary(false);
  await new Promise((r) => setTimeout(r, EVERY_MIN * 60_000));
}
// let the last tasks finish
for (let i = 0; i < 120; i++) {
  const open = (await Promise.all(submitted.map((s) => api("GET", `/tasks/${s.id}`)))).filter((t) => ["queued", "running"].includes(t.status)).length;
  if (!open) break;
  await new Promise((r) => setTimeout(r, 30_000));
}
await summary(true);
