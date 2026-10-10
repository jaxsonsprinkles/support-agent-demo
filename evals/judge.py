"""Optional LLM-judge rubric hook. OFF by default and strictly secondary: it only runs on cases that have a
`rubric` AND whose deterministic checks already passed, and only when a judge object is passed to run_case.

The judge is PINNED (JUDGE_MODEL, default claude-sonnet-5-5), must be a smarter tier than the support agent
(see evals/models.py; LLMJudge fails loudly otherwise), runs at temperature 0, follows the rubric strictly and
returns a structured verdict:

    {"passed": bool, "reason": str, "criteria": [{"criterion", "passed", "reason"}],
     "rubric_sha256": str, "judge_model": str}

`passed` is computed in code as "every criterion passed"; the model's own overall claim is ignored.
A judge is any object with `judge(case, run) -> dict` containing at least `passed` and `reason`.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Protocol

from evals import models

JUDGE_SYSTEM = (
    "You are a strict grader of a customer-support agent's reply. Grade ONLY against the rubric; do not add "
    "criteria of your own and do not be lenient. Split the rubric into its individual criteria and judge each "
    "one independently as pass or fail with a one-sentence reason. Respond with only JSON: "
    '{"criteria": [{"criterion": "<text>", "passed": true|false, "reason": "<one sentence>"}]}'
)


def rubric_hash(rubric: str) -> str:
    return hashlib.sha256(rubric.encode("utf-8")).hexdigest()


class Judge(Protocol):
    def judge(self, case: Any, run: Any) -> dict[str, Any]: ...


class LLMJudge:
    """Judge backed by an agent.provider.Provider (no tools). The provider's `model` is the pinned judge id."""

    def __init__(self, provider: Any, *, model: str | None = None, agent_model: str | None = None,
                 enforce_guard: bool = True) -> None:
        self.provider = provider
        self.model = model or getattr(provider, "model", None) or models.judge_model()
        self.agent_model = agent_model or models.agent_model()
        if enforce_guard:
            models.check_judge_model(self.model, self.agent_model)

    def judge(self, case: Any, run: Any) -> dict[str, Any]:
        prompt = f"Customer message: {case.input}\nAgent reply: {run.reply}\nRubric: {case.rubric}"
        r = self.provider.create(system=JUDGE_SYSTEM, messages=[{"role": "user", "content": prompt}], tools=[])
        base = {"rubric_sha256": rubric_hash(case.rubric), "judge_model": self.model}
        m = re.search(r"\{.*\}", r.text, re.S)
        try:
            raw = json.loads(m.group(0))["criteria"] if m else []
            criteria = [{"criterion": str(c.get("criterion", "")), "passed": c["passed"] is True,
                         "reason": str(c.get("reason", ""))} for c in raw]
            if not criteria:
                raise ValueError("no criteria")
        except (ValueError, KeyError, TypeError, AttributeError):
            return {"passed": False, "reason": f"unparseable judge output: {r.text[:120]!r}", "criteria": [], **base}
        failed = [c for c in criteria if not c["passed"]]
        reason = "; ".join(c["reason"] for c in failed) if failed else "; ".join(c["reason"] for c in criteria)
        return {"passed": not failed, "reason": reason, "criteria": criteria, **base}
