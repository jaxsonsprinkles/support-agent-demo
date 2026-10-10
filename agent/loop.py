from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.prompt import SYSTEM_PROMPT
from agent.provider import Provider
from agent.tracing import current_trace_id, traced
from tools import ToolBox

MAX_STEPS = 6  # model calls per run


@dataclass
class RunResult:
    reply: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {name, args, result}
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    trace_id: str = ""
    hit_step_limit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_agent(message: str, provider: Provider, toolbox: ToolBox | None = None,
              max_steps: int | None = None) -> RunResult:
    """Run one customer message through the agent. `max_steps` defaults to MAX_STEPS."""
    return _run(message, provider, toolbox or ToolBox(), max_steps or MAX_STEPS)


@traced
def _run(message: str, provider: Provider, toolbox: ToolBox, max_steps: int) -> RunResult:
    res = RunResult(reply="", trace_id=current_trace_id())
    messages: list[dict[str, Any]] = [{"role": "user", "content": message}]
    for _ in range(max_steps):
        r = provider.create(system=SYSTEM_PROMPT, messages=messages, tools=toolbox.schemas)
        res.steps += 1
        res.input_tokens += r.input_tokens
        res.output_tokens += r.output_tokens
        if not r.tool_uses:
            res.reply = r.text
            return res
        messages.append({"role": "assistant", "content": r.content})
        results = []
        for tu in r.tool_uses:
            out = toolbox.call(tu["name"], tu["input"])
            res.tool_calls.append({"name": tu["name"], "args": tu["input"], "result": out})
            results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
        res.reply = r.text
    res.hit_step_limit = True
    if not res.reply:
        res.reply = "Sorry, I could not finish handling this request. A human agent will follow up."
    return res
