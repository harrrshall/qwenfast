/**
 * Unit tests for the client-side history trimming (`lib/trim.ts`).
 *
 *   node --test demo/tests/
 *
 * These are the half of the long-conversation fix that runs in the browser; the server half
 * (clamping `max_tokens` into the room the prompt leaves) is tested in
 * `engine/qwenfast/server/tests/test_reliability.py`.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  CHARS_PER_TOKEN,
  PER_MESSAGE_TOKENS,
  estimatePromptTokens,
  estimateTokens,
  promptBudget,
  trimHistory,
  type ChatMsg,
  type Limits,
} from "../lib/trim.ts";

const LIMITS: Limits = {
  max_context_length: 4096,
  max_prompt_tokens: null,
  min_completion_tokens: 256,
  max_messages: 128,
};

function convo(turns: number, chars = 300): ChatMsg[] {
  const out: ChatMsg[] = [];
  for (let i = 0; i < turns; i++) {
    out.push({ role: "user", content: `u${i} ` + "x".repeat(chars) });
    out.push({ role: "assistant", content: `a${i} ` + "y".repeat(chars) });
  }
  return out;
}

test("the token estimate over-counts rather than under-counts", () => {
  // Qwen BPE is ~3.7 chars/token on prose; the estimate must be at least as large as the
  // real count, or trimming stops short of the budget and the server refuses the request.
  assert.ok(CHARS_PER_TOKEN < 3.7);
  assert.equal(estimateTokens("x".repeat(300)), 100);
  assert.equal(estimateTokens(""), 0);
});

test("promptBudget prefers an explicit max_prompt_tokens", () => {
  assert.equal(promptBudget({ ...LIMITS, max_prompt_tokens: 3072 }), 3072);
});

test("promptBudget otherwise reserves min_completion_tokens out of the context", () => {
  assert.equal(promptBudget(LIMITS), 4096 - 256);
  // And has a sane answer when the page has no limits yet (first paint).
  assert.equal(promptBudget(null), 4096 - 256);
});

test("a short conversation is sent untouched", () => {
  const messages = convo(2);
  const out = trimHistory(messages, LIMITS);
  assert.equal(out.dropped, 0);
  assert.deepEqual(out.messages, messages);
});

test("a long conversation is trimmed from the front, and keeps the newest turn", () => {
  const messages = convo(60); // 120 messages, ~12k estimated tokens
  const out = trimHistory(messages, LIMITS);

  assert.ok(out.dropped > 0, "should have dropped something");
  assert.ok(estimatePromptTokens(out.messages) <= promptBudget(LIMITS));
  // The message the user just typed is always the last one sent.
  assert.deepEqual(out.messages.at(-1), messages.at(-1));
  // And what survives is a contiguous suffix of the original.
  assert.deepEqual(out.messages, messages.slice(out.dropped));
});

test("a very long conversation that no longer fits is trimmed to fit", () => {
  // A history of about 23,000 prompt tokens, re-sent in full. Untrimmed it is
  // far over budget; trimmed it fits, which is the whole point.
  const messages = convo(120, 600);
  assert.ok(estimatePromptTokens(messages) > 22_000);
  const out = trimHistory(messages, LIMITS);
  assert.ok(estimatePromptTokens(out.messages) <= promptBudget(LIMITS));
});

test("the message-count cap is respected as well as the token budget", () => {
  // Many tiny messages: nowhere near the token budget, but over --max-messages.
  const messages: ChatMsg[] = Array.from({ length: 200 }, (_, i) => ({
    role: i % 2 ? "assistant" : "user",
    content: "hi",
  }));
  const out = trimHistory(messages, { ...LIMITS, max_messages: 20 });
  assert.ok(out.messages.length <= 19, `sent ${out.messages.length}`);
  assert.deepEqual(out.messages.at(-1), messages.at(-1));
});

test("a system message survives trimming", () => {
  const messages: ChatMsg[] = [{ role: "system", content: "be terse" }, ...convo(60)];
  const out = trimHistory(messages, LIMITS);
  assert.equal(out.messages[0].role, "system");
  assert.equal(out.messages[0].content, "be terse");
  assert.ok(estimatePromptTokens(out.messages) <= promptBudget(LIMITS));
});

test("a single over-budget message is sent anyway, for the server to explain", () => {
  // Trimming cannot help here — there is nothing older to drop — and the server's 400 names
  // the real token count, which is more useful than silently truncating what was typed.
  const messages: ChatMsg[] = [{ role: "user", content: "x".repeat(50_000) }];
  const out = trimHistory(messages, LIMITS);
  assert.equal(out.dropped, 0);
  assert.equal(out.messages.length, 1);
});

test("per-message template overhead is counted, not just the text", () => {
  const one: ChatMsg[] = [{ role: "user", content: "" }];
  assert.equal(estimatePromptTokens(one), PER_MESSAGE_TOKENS * 2);
});
