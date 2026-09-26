// scripted openai compatible chat server for daemon tests. streams sse like qwenfast and vllm.
//
// behaviour per request (looked up by the model name on the wire):
//   judge calls (max_tokens <= 8)          -> the word in `judgeReply`
//   last message is a tool result           -> final text from `finalText(model)`
//   otherwise                               -> one bash tool call running `command(model)`
// `delayMs` holds every agent reply, so a test can kill the daemon mid-attempt.

import { createServer, type Server } from "node:http";

export interface MockScript {
  judgeReply: string;
  command: (model: string) => string;
  finalText: (model: string) => string;
  delayMs?: number;
  /** delay only replies that follow a tool result (the session file exists by then) */
  delayAfterToolMs?: number;
}

export interface MockServer {
  url: string;
  requests: Array<{ model: string; body: any }>;
  close: () => Promise<void>;
  script: MockScript;
}

export async function startMock(script: MockScript): Promise<MockServer> {
  const requests: Array<{ model: string; body: any }> = [];
  const server: Server = createServer(async (req, res) => {
    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200).end("ok");
      return;
    }
    if (req.method === "GET") {
      res.writeHead(404).end("not found");
      return;
    }
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const body = JSON.parse(Buffer.concat(chunks).toString() || "{}");
    const model = String(body.model);
    requests.push({ model, body });
    const msgs = body.messages as Array<{ role: string }>;
    if ((body.max_tokens ?? 999) <= 8 && !body.stream) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: script.judgeReply } }] }));
      return;
    }
    if (script.delayMs) await new Promise((r) => setTimeout(r, script.delayMs));
    if (script.delayAfterToolMs && msgs.at(-1)?.role === "tool") await new Promise((r) => setTimeout(r, script.delayAfterToolMs));
    const base = { id: "c1", object: "chat.completion.chunk", created: 1, model };
    const send = (delta: object, finish: string | null = null) =>
      res.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`);
    res.writeHead(200, { "content-type": "text/event-stream" });
    send({ role: "assistant", content: "" });
    if (msgs.at(-1)?.role === "tool") {
      send({ reasoning_content: "checking the result. " });
      send({ content: script.finalText(model) });
      send({}, "stop");
    } else {
      send({ reasoning_content: "I will run a command. " });
      send({ tool_calls: [{ index: 0, id: "call_0", type: "function", function: { name: "bash", arguments: "" } }] });
      send({ tool_calls: [{ index: 0, function: { arguments: JSON.stringify({ command: script.command(model) }) } }] });
      send({}, "tool_calls");
    }
    res.write(`data: ${JSON.stringify({ ...base, choices: [], usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 } })}\n\n`);
    res.end("data: [DONE]\n\n");
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const addr = server.address() as { port: number };
  return {
    url: `http://127.0.0.1:${addr.port}`,
    requests,
    script,
    close: () => new Promise((r) => { server.closeAllConnections(); server.close(() => r()); }),
  };
}
