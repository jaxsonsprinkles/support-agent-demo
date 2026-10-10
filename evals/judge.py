"""Optional LLM-judge rubric hook. OFF by default and strictly secondary: it only runs on cases that have a
`rubric` AND whose deterministic checks already passed, and only when a judge object is passed to run_case.

A judge is any object with `judge(case, run) -> {"passed": bool, "reason": str}`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

JUDGE_SYSTEM = (
    "You grade a customer-support agent's reply against a rubric. Respond with only JSON: "
    '{"passed": true|false, "reason": "<one sentence>"}'
)


class Judge(Protocol):
    def judge(self, case: Any, run: Any) -> dict[str, Any]: ...


class LLMJudge:
    """Judge backed by an agent.provider.Provider (no tools)."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def judge(self, case: Any, run: Any) -> dict[str, Any]:
        prompt = f"Customer message: {case.input}\nAgent reply: {run.reply}\nRubric: {case.rubric}"
        r = self.provider.create(system=JUDGE_SYSTEM, messages=[{"role": "user", "content": prompt}], tools=[])
        m = re.search(r"\{.*\}", r.text, re.S)
        try:
            v = json.loads(m.group(0)) if m else {}
            return {"passed": bool(v["passed"]), "reason": str(v.get("reason", ""))}
        except (ValueError, KeyError):
            return {"passed": False, "reason": f"unparseable judge output: {r.text[:120]!r}"}
