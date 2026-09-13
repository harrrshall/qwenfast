/**
 * Keep a conversation inside the engine's prompt budget by dropping its oldest turns.
 *
 * Why the client does this at all. The endpoint serves a 4,096-token context; a chat page that
 * posts its entire history grows past that after a dozen exchanges and then fails *every*
 * subsequent message, permanently, with no way out but "clear": one conversation re-sent in
 * full, again and again, after it has already stopped fitting.
 *
 * The server-side half of the fix (clamping `max_tokens` into the room the prompt leaves) makes
 * long conversations *degrade* instead of failing; this is the other half, which keeps them
 * working at all once the prompt alone no longer fits.
 *
 * Token counting is an estimate, because the browser has no tokenizer and shipping one for this
 * would cost more than the whole page. It is deliberately biased to *over*-count: guessing high
 * trims one extra turn, guessing low reproduces the bug.
 */

export type Role = "user" | "assistant" | "system";
export type ChatMsg = { role: Role; content: string };

export type Limits = {
  max_context_length?: number | null;
  max_prompt_tokens?: number | null;
  min_completion_tokens?: number | null;
  max_messages?: number | null;
};

/**
 * Characters per token, for the estimate.
 *
 * Qwen's BPE averages ~3.7 chars/token on English prose and rather fewer on code, markdown and
 * CJK. 3.0 is below all of those, so the estimate lands high on every realistic input.
 */
export const CHARS_PER_TOKEN = 3;

/**
 * Chat-template overhead per message: the role markers and separators the server adds around
 * each turn (`<|im_start|>role\n` … `<|im_end|>\n`), which the raw text does not include.
 */
export const PER_MESSAGE_TOKENS = 8;

export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

export function estimateMessageTokens(m: ChatMsg): number {
  return estimateTokens(m.content) + PER_MESSAGE_TOKENS;
}

export function estimatePromptTokens(messages: ChatMsg[]): number {
  // A trailing generation prompt is added by the template on top of the messages themselves.
  return messages.reduce((n, m) => n + estimateMessageTokens(m), PER_MESSAGE_TOKENS);
}

/**
 * How many prompt tokens the server will accept.
 *
 * `max_prompt_tokens` is authoritative when the deployment sets one. Otherwise the budget is the
 * context minus whatever the server insists on leaving for a reply (`min_completion_tokens`),
 * which is the same arithmetic `fit_to_context` does on the other side.
 */
export function promptBudget(limits: Limits | null | undefined): number {
  const ctx = limits?.max_context_length ?? 4096;
  const explicit = limits?.max_prompt_tokens;
  if (explicit && explicit > 0) return explicit;
  const floor = limits?.min_completion_tokens ?? 256;
  return Math.max(256, ctx - floor);
}

export type TrimResult = {
  /** What to send. Always contains at least the final message. */
  messages: ChatMsg[];
  /** How many leading messages were dropped. 0 means the history fitted as-is. */
  dropped: number;
};

/**
 * Drop the oldest messages until the rest fit the prompt budget and the message cap.
 *
 * The newest turn is never dropped: it is what the user just typed, and sending the
 * conversation without it would answer the wrong question. If that message alone is over
 * budget, it is sent anyway — the server's 400 for that case names the actual number of tokens
 * and what to do about it, which is a better answer than this function silently truncating the
 * user's text.
 *
 * A leading `system` message is preserved wherever it appears in the kept window, because
 * dropping it changes the model's behaviour rather than just its memory.
 */
export function trimHistory(messages: ChatMsg[], limits: Limits | null | undefined): TrimResult {
  if (messages.length <= 1) return { messages, dropped: 0 };

  const budget = promptBudget(limits);
  const maxMessages = Math.max(2, (limits?.max_messages ?? 128) - 1);

  const system = messages[0]?.role === "system" ? messages[0] : null;
  const body = system ? messages.slice(1) : messages;

  let start = 0;
  const systemCost = system ? estimateMessageTokens(system) : 0;

  const fits = (from: number) =>
    estimatePromptTokens(body.slice(from)) + systemCost <= budget &&
    body.length - from + (system ? 1 : 0) <= maxMessages;

  // Walk forward one message at a time. The history is at most a few hundred entries and this
  // runs once per send, so the simple loop is cheaper than being clever about it.
  while (start < body.length - 1 && !fits(start)) start += 1;

  const kept = body.slice(start);
  return {
    messages: system ? [system, ...kept] : kept,
    dropped: start,
  };
}
