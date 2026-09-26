// pi model definitions for the three tiers.
//
// both servers speak openai chat completions. thinking is switched through chat_template_kwargs
// exactly as the qwen chat templates read it: `enable_thinking`, plus `reasoning_effort` on
// qwen3.8 whose template knows three levels (low, medium, xhigh). pi thinking levels map onto
// those; a level the template does not know would make it raise, so every level is mapped.

import type { Model } from "@earendil-works/pi-ai";
import type { TierConfig, TierName } from "./config.ts";

export const PROVIDER_BIG = "qwenfast";
export const PROVIDER_SMALL = "qwenfast-small";

export function tierModel(name: TierName, tier: TierConfig, baseUrl: string): Model<"openai-completions"> {
  const big = tier.endpoint === "big";
  return {
    id: tier.model,
    name: `${tier.model} (${name})`,
    api: "openai-completions",
    provider: big ? PROVIDER_BIG : PROVIDER_SMALL,
    baseUrl: `${baseUrl.replace(/\/+$/, "")}/v1`,
    reasoning: true,
    thinkingLevelMap: big
      ? { off: null, minimal: "low", low: "low", medium: "medium", high: "xhigh", xhigh: "xhigh", max: "xhigh" }
      : undefined,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: tier.contextWindow,
    maxTokens: tier.maxTokens,
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsUsageInStreaming: true,
      maxTokensField: "max_tokens",
      supportsStrictMode: false,
      supportsMidConvoSystemMessages: false,
      thinkingFormat: "chat-template",
      chatTemplateKwargs: big
        ? {
            enable_thinking: { $var: "thinking.enabled" },
            reasoning_effort: { $var: "thinking.effort", omitWhenOff: true },
          }
        : { enable_thinking: { $var: "thinking.enabled" } },
    },
  };
}
