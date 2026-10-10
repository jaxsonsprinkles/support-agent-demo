"""Eval case model and loader (evals/cases.jsonl)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evals.checks import validate_checks

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"
CATEGORY_COUNTS = {
    "order_status": 5, "refund_within_policy": 4, "refund_outside_policy": 3, "cancel_request": 3,
    "policy_question": 4, "escalation": 3, "multi_step": 2,
}
# Ground truth ("breaks" R1/R2/R3, "intent_for") is deliberately NOT part of a case: it would leak the planted
# regressions to anything that reads this repo. It lives in agentbisect/evidence/ground_truth.json.
FORBIDDEN_KEYS = {"breaks", "intent_for"}
SUITES = ("main", "held_out", "all")


@dataclass
class Case:
    id: str
    category: str
    input: str
    checks: dict[str, Any]
    held_out: bool = False
    requires_tool: str | None = None
    rubric: str | None = None      # optional LLM-judge rubric (judge is off by default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases(path: Path | str | None = None, suite: str = "all", ids: list[str] | None = None) -> list[Case]:
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {SUITES}")
    cases: list[Case] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path or CASES_PATH).read_text().splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if leaked := FORBIDDEN_KEYS & raw.keys():
            raise ValueError(f"line {n}: ground-truth keys {sorted(leaked)} must not appear in cases.jsonl")
        c = Case(**raw)
        if c.id in seen:
            raise ValueError(f"line {n}: duplicate case id {c.id}")
        seen.add(c.id)
        if c.category not in CATEGORY_COUNTS:
            raise ValueError(f"{c.id}: unknown category {c.category}")
        validate_checks(c.checks)
        cases.append(c)
    if suite == "main":
        cases = [c for c in cases if not c.held_out]
    elif suite == "held_out":
        cases = [c for c in cases if c.held_out]
    if ids:
        unknown = set(ids) - {c.id for c in cases}
        if unknown:
            raise ValueError(f"unknown case ids in suite '{suite}': {sorted(unknown)}")
        cases = [c for c in cases if c.id in ids]
    return cases
