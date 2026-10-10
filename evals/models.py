"""Pinned model identities for the eval pipeline.

Roles: the support agent (cheap, Haiku tier) is the lab rat; the judge must be a strictly smarter model.
Both ids are recorded in every eval result so a model never silently drifts between commits, and results
produced under different pinned models refuse to be compared.
"""
from __future__ import annotations

import os
from typing import Any

DEFAULT_AGENT_MODEL = "claude-haiku-5-5"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"

TIERS = {"haiku": 1, "sonnet": 2, "opus": 3, "fable": 3}  # haiku < sonnet < opus/fable


class ModelConfigError(ValueError):
    """Raised when pinned models are misconfigured or results from different pins are mixed."""


def tier(model: str) -> int:
    """Tier rank of a model id by substring ('claude-haiku-5-5' -> 1). Unknown families raise."""
    m = model.lower()
    for name, rank in TIERS.items():
        if name in m:
            return rank
    raise ModelConfigError(f"cannot determine tier of model '{model}' (expected haiku/sonnet/opus/fable in id)")


def agent_model() -> str:
    return os.environ.get("AGENT_MODEL") or DEFAULT_AGENT_MODEL


def judge_model() -> str:
    return os.environ.get("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


def check_judge_model(judge: str, agent: str) -> None:
    """Fail loudly unless the judge is a different model, not haiku-tier, and a higher tier than the agent."""
    if judge == agent:
        raise ModelConfigError(f"judge model must differ from agent model (both are '{judge}')")
    jt, at = tier(judge), tier(agent)
    if jt <= TIERS["haiku"]:
        raise ModelConfigError(f"judge model '{judge}' is haiku-tier; the judge must be smarter than the agent")
    if jt <= at:
        raise ModelConfigError(f"judge model '{judge}' must be a higher tier than agent model '{agent}'")


def pinned(agent: str | None = None, judge: str | None = None) -> dict[str, str]:
    return {"agent_model": agent or agent_model(), "judge_model": judge or judge_model()}


def assert_comparable(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Refuse to compare two result/baseline dicts unless both agent_model and judge_model are present and equal."""
    for key in ("agent_model", "judge_model"):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            raise ModelConfigError(f"cannot compare: '{key}' missing from a result (unpinned results are not comparable)")
        if va != vb:
            raise ModelConfigError(f"cannot compare results across pinned models: {key} {va!r} != {vb!r}")
