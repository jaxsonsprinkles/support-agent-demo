from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.logs import log_tool_call
from agent.prompt import SYSTEM_PROMPT
from agent.provider import Provider
from agent.tracing import current_trace_id, traced
from tools import ToolBox

MAX_STEPS = 12  # model calls per run; leaves room for verification searches and retries
_UNKNOWN_ORDER = re.compile(r"^order '(.*)' not found$")


def unknown_order_reply(order_id: str) -> str:
    return (f"Sorry, I couldn't find an order with the number {order_id}. Could you double-check the order "
            "number and send it again? Is there anything else we can help with?")


def _guard_unknown_order(reply: str, tool_calls: list[dict[str, Any]]) -> str:
    """Replace a reply that echoes a raw 'order ... not found' tool error with a customer-friendly apology."""
    for tc in tool_calls:
        err = tc["result"].get("error", "")
        m = _UNKNOWN_ORDER.match(err)
        if m and err.lower() in reply.lower():
            return unknown_order_reply(m.group(1))
    return reply


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
            res.reply = _guard_unknown_order(r.text, res.tool_calls)
            return res
        messages.append({"role": "assistant", "content": r.content})
        results = []
        for tu in r.tool_uses:
            t0 = time.perf_counter()
            out = toolbox.call(tu["name"], tu["input"])
            log_tool_call(tu["name"], tu["input"], (time.perf_counter() - t0) * 1000)
            res.tool_calls.append({"name": tu["name"], "args": tu["input"], "result": out})
            results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
        res.reply = r.text
    res.hit_step_limit = True
    if not res.reply:
        res.reply = "Sorry, I could not finish handling this request. A human agent will follow up."
    return res
