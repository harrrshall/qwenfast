"""harbor agent: the qwenfast pi agent (same harness, models and router as agent/) on terminal-bench.

environment on the harbor host:

    QFA_TIER        small | medium | large | auto   (auto = the agent's own router)
    QFA_BIG_URL     qwenfast base url (no /v1)
    QFA_SMALL_URL   vllm base url (no /v1)
    QFA_API_KEY     key for both servers
    QFA_AGENT_DIR   path to agent/ (its node cli emits the models.json and routes)

run:

    harbor run -d terminal-bench@2.0 --agent-import-path qwenfast_tbench:QwenfastPiAgent \
        -m qwenfast/qwen3.8-27b --ak version=0.87.1 -n 8
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from typing import Any, override

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_PI_DIR = "/tmp/harbor-pi-agent"

AUTONOMY = (
    "You are running as an autonomous agent. No human is watching and nobody will answer questions. "
    "Never ask for confirmation: pick the most reasonable interpretation and continue. Work until the "
    "task is actually complete, and check your work with the tools (run the code or a command that "
    "proves the result) before you finish. Keep tool output small (use head, tail, grep)."
)

# tier -> (pi provider, model id, thinking level); mirrors agent/src/config.ts defaults
TIERS: dict[str, tuple[str, str, str]] = {
    "small": ("qwenfast-small", "qwen3.6-35b-a3b", "off"),
    "medium": ("qwenfast", "qwen3.8-27b", "medium"),
    "large": ("qwenfast", "qwen3.8-27b", "high"),
}


def _agent_cli(*args: str) -> str:
    agent_dir = os.environ["QFA_AGENT_DIR"]
    out = subprocess.run(
        ["node", os.path.join(agent_dir, "src", "cli.ts"), *args],
        check=True, capture_output=True, text=True, timeout=120, env=os.environ,
    )
    return out.stdout


def route(instruction: str) -> dict[str, Any]:
    tier = os.environ.get("QFA_TIER", "auto")
    if tier != "auto":
        return {"tier": tier, "source": "forced"}
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(instruction)
        path = f.name
    try:
        return json.loads(_agent_cli("route", "--judge", "--file", path))
    except Exception as exc:  # the router must never cost a trial: fall back to the middle tier
        return {"tier": "medium", "source": f"route-error: {exc}"[:200]}
    finally:
        os.unlink(path)


class QwenfastPiAgent(Pi):
    @staticmethod
    @override
    def name() -> str:
        return "qwenfast-pi"

    @override
    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        decision = route(instruction)
        tier = decision.get("tier", "medium")
        provider, model_id, thinking = TIERS[tier]
        (self.logs_dir / "qfa-route.json").write_text(json.dumps(decision) + "\n")

        models = json.loads(
            _agent_cli("models-json", "--big", os.environ["QFA_BIG_URL"], "--small", os.environ["QFA_SMALL_URL"])
        )
        await self._write_custom_models_json(environment, models)

        env = {"QFA_PI_KEY": os.environ["QFA_API_KEY"]}
        resume = "--continue " if self._resume else ""
        await self.exec_as_agent(
            environment,
            command=(
                ". ~/.nvm/nvm.sh; "
                f"PI_CODING_AGENT_DIR={_PI_DIR} pi --print --mode json "
                "--session-dir /logs/agent/pi/sessions "
                f"{resume}--provider {provider} --model {model_id} --thinking {thinking} "
                f"--append-system-prompt {shlex.quote(AUTONOMY)} "
                f"{shlex.quote(instruction)} "
                "2>&1 </dev/null | grep -v '\"type\":\"message_update\"' | stdbuf -oL tee /logs/agent/pi.txt"
            ),
            env=env,
        )
