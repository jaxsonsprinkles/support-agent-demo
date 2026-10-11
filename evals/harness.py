"""Run eval cases against the agent. Deterministic checks first, then budget, then the optional judge."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evals import SCHEMA_VERSION, models
from evals.cases import Case
from evals.checks import run_checks

MAX_STEPS_BUDGET = 6
TOKEN_FACTOR = 1.5
BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- baseline
class Baseline:
    """Per-case median tokens/steps from a good-commit run (written by `python -m evals calibrate`)."""

    def __init__(self, data: dict[str, Any], path: str | None = None) -> None:
        self.data, self.path = data, path
        self.cases: dict[str, dict[str, float]] = data.get("cases", {})

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Baseline | None":
        p = Path(path or BASELINE_PATH)
        return cls(json.loads(p.read_text()), str(p)) if p.exists() else None

    def token_limit(self, case_id: str) -> float | None:
        c = self.cases.get(case_id)
        return TOKEN_FACTOR * c["median_tokens"] if c else None


def build_baseline(results: list["CaseResult"], meta: dict[str, Any]) -> dict[str, Any]:
    """Medians over runs whose deterministic checks passed (budget ignored)."""
    by: dict[str, list[CaseResult]] = {}
    for r in results:
        if r.status in ("pass", "fail") and r.checks_passed:
            by.setdefault(r.case_id, []).append(r)
    return {
        "schema_version": SCHEMA_VERSION, "commit": meta.get("commit"), "created_at": _now(),
        "provider": meta.get("provider"),
        "agent_model": meta.get("agent_model"), "judge_model": meta.get("judge_model"), "token_factor": TOKEN_FACTOR, "max_steps": MAX_STEPS_BUDGET,
        "cases": {cid: {"median_tokens": statistics.median(r.tokens for r in rs),
                        "median_steps": statistics.median(r.steps for r in rs), "n": len(rs)}
                  for cid, rs in sorted(by.items())},
    }


# ---------------------------------------------------------------- results
@dataclass
class CaseResult:
    case_id: str
    category: str
    held_out: bool
    repeat: int = 0
    agent_model: str | None = None  # pinned ids, recorded in every result (see evals/models.py)
    judge_model: str | None = None
    status: str = "pass"  # pass | fail | skipped | error
    checks: list[dict[str, Any]] = field(default_factory=list)
    checks_passed: bool | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] | None = None
    reply: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens: int = 0
    trace_id: str = ""
    duration_s: float = 0.0
    reason: str | None = None  # skip / error explanation

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaseResult":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def budget_check(steps: int, tokens: int, case_id: str, baseline: Baseline | None) -> dict[str, Any]:
    """steps <= 6 always; tokens <= 1.5x baseline median when a baseline exists for the case."""
    b: dict[str, Any] = {"steps": steps, "max_steps": MAX_STEPS_BUDGET, "tokens": tokens,
                         "token_limit": None, "baseline_median_tokens": None}
    reasons = []
    if steps > MAX_STEPS_BUDGET:
        reasons.append(f"{steps} steps > {MAX_STEPS_BUDGET}")
    if baseline is None:
        b["token_check"] = "skipped: no baseline"
    elif (limit := baseline.token_limit(case_id)) is None:
        b["token_check"] = "skipped: case not in baseline"
    else:
        b["token_check"] = "checked"
        b["token_limit"], b["baseline_median_tokens"] = limit, baseline.cases[case_id]["median_tokens"]
        if tokens > limit:
            reasons.append(f"{tokens} tokens > limit {limit:.0f} ({TOKEN_FACTOR}x baseline median)")
    b["status"] = "fail" if reasons else ("pass" if b["token_check"] == "checked" else "partial")
    b["reason"] = "; ".join(reasons) or ("within budget" if b["status"] == "pass" else "steps ok; " + b["token_check"])
    return b


# ---------------------------------------------------------------- running
def run_case(case: Case, provider: Any, *, toolbox_factory: Callable[[], Any] | None = None,
             baseline: Baseline | None = None, judge: Any | None = None, repeat: int = 0,
             max_steps: int | None = None, agent_model: str | None = None,
             judge_model: str | None = None) -> CaseResult:
    """Run one case once. `provider` is a Provider. Never raises for agent/provider failures (status 'error').

    Tracing tags (commit/run/case) are process-wide, see evals/README.md; use run_suite(isolate=True) for
    per-case tags."""
    from agent.loop import run_agent
    from agent.tracing import CASE_SPAN, span
    from tools import ToolBox

    res = CaseResult(case.id, case.category, case.held_out, repeat=repeat,
                     agent_model=agent_model or getattr(provider, "model", None) or models.agent_model(),
                     judge_model=judge_model or getattr(judge, "model", None) or models.judge_model())
    # flaky tool is seeded per (case, repeat) so reruns reproduce the same timeouts (see DEMO_FLAKE_SEED / DEMO_FLAKE)
    flake_seed = f"{os.environ.get('DEMO_FLAKE_SEED', '')}:{case.id}:{repeat}"
    toolbox = toolbox_factory() if toolbox_factory else ToolBox(flake_seed=flake_seed)
    if case.requires_tool and case.requires_tool not in {s["name"] for s in toolbox.schemas}:
        res.status, res.reason = "skipped", f"requires tool '{case.requires_tool}' which this commit does not have"
        return res
    t0 = time.time()
    try:
        # one parent span per case run: every model call and tool call below becomes a child of one trace
        with span(CASE_SPAN, "WORKFLOW", **{"case.id": case.id, "case.category": case.category, "case.repeat": repeat}):
            run = run_agent(case.input, provider, toolbox, max_steps)
    except Exception as e:  # provider/network/agent crash: infra problem, not a behavioral verdict
        res.status, res.reason, res.duration_s = "error", f"{type(e).__name__}: {e}", round(time.time() - t0, 3)
        return res
    res.duration_s = round(time.time() - t0, 3)
    res.reply, res.tool_calls, res.steps, res.trace_id = run.reply, run.tool_calls, run.steps, run.trace_id
    res.input_tokens, res.output_tokens = run.input_tokens, run.output_tokens
    res.tokens = run.input_tokens + run.output_tokens
    res.checks = run_checks(case.checks, run)
    res.checks_passed = all(c["passed"] for c in res.checks)
    res.budget = budget_check(res.steps, res.tokens, case.id, baseline)
    ok = res.checks_passed and res.budget["status"] != "fail"
    if judge is not None and case.rubric and res.checks_passed:  # judge only after deterministic checks pass
        try:
            res.judge = judge.judge(case, run)
        except Exception as e:
            res.judge = {"passed": False, "reason": f"judge error: {e}"}
        ok = ok and res.judge["passed"]
    res.status = "pass" if ok else "fail"
    return res


def _make_provider(provider: Any, case: Case) -> Any:
    """`provider` is a Provider, or a callable(case) -> Provider (needed for ScriptedProvider, which is consumed)."""
    return provider if hasattr(provider, "create") else provider(case)


def run_suite(cases: list[Case], provider: Any, *, repeats: int = 1, isolate: bool = False,
              toolbox_factory: Callable[[], Any] | None = None, baseline: Baseline | None = None,
              judge: Any | None = None, meta: dict[str, Any] | None = None,
              child_args: list[str] | None = None, progress: Callable[[CaseResult], None] | None = None
              ) -> dict[str, Any]:
    """Run every case `repeats` times and return the report dict (see evals/README.md)."""
    meta = {**models.pinned(), **{k: v for k, v in (meta or {}).items() if v is not None}}
    if baseline is not None:  # never judge a run against a baseline made under different pinned models
        models.assert_comparable(baseline.data, meta)
    results: list[CaseResult] = []
    for case in cases:
        for rep in range(repeats):
            if isolate:
                r = run_case_isolated(case, rep, child_args or [], meta or {})
            else:
                r = run_case(case, _make_provider(provider, case), toolbox_factory=toolbox_factory,
                             baseline=baseline, judge=judge, repeat=rep,
                             agent_model=meta["agent_model"], judge_model=meta["judge_model"])
            results.append(r)
            if progress:
                progress(r)
    return build_report(results, cases, repeats=repeats, baseline=baseline, isolated=isolate, meta=meta)


def run_case_isolated(case: Case, repeat: int, child_args: list[str], meta: dict[str, Any]) -> CaseResult:
    """Run one case in a fresh `python -m evals` subprocess so neatlogs tags (process-wide) are per case."""
    env = {**os.environ, "AGENTBISECT_CASE": case.id}
    for key, name in (("commit", "AGENTBISECT_COMMIT"), ("run_id", "AGENTBISECT_RUN_ID")):
        if meta.get(key):
            env[name] = str(meta[key])
    cmd = [sys.executable, "-m", "evals", "run", "--suite", "all", "--cases", case.id, "--repeats", "1",
           "--json", *child_args]
    try:
        p = subprocess.run(cmd, env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
        out = json.loads(p.stdout)
        r = CaseResult.from_dict(out["results"][0])
    except Exception as e:
        stderr = locals().get("p") and p.stderr[-300:]
        r = CaseResult(case.id, case.category, case.held_out, status="error",
                       agent_model=meta.get("agent_model"), judge_model=meta.get("judge_model"),
                       reason=f"isolated run failed: {type(e).__name__}: {e} {stderr or ''}".strip())
    r.repeat = repeat
    return r


# ---------------------------------------------------------------- report
def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def build_report(results: list[CaseResult], cases: list[Case], *, repeats: int, baseline: Baseline | None,
                 isolated: bool, meta: dict[str, Any]) -> dict[str, Any]:
    ran = [r for r in results if r.status in ("pass", "fail")]
    bud = [r for r in ran if r.budget.get("status") in ("pass", "fail")]
    per_case = []
    for c in cases:
        rs = [r for r in results if r.case_id == c.id]
        ex = [r for r in rs if r.status in ("pass", "fail")]
        eb = [r for r in ex if r.budget.get("status") in ("pass", "fail")]
        n_pass = sum(r.passed for r in rs)
        if rs and all(r.status == "skipped" for r in rs):
            st = "skipped"
        elif rs and all(r.status == "error" for r in rs):
            st = "error"
        elif ex and n_pass == len(ex):
            st = "pass"
        elif n_pass == 0:
            st = "fail"
        else:
            st = "flaky"
        per_case.append({
            "case_id": c.id, "category": c.category, "held_out": c.held_out, "status": st,
            "runs": len(rs), "passed": n_pass, "failed": sum(r.status == "fail" for r in rs),
            "skipped": sum(r.status == "skipped" for r in rs), "errors": sum(r.status == "error" for r in rs),
            "pass_rate": _rate(n_pass, len(ex)),
            "checks_pass_rate": _rate(sum(bool(r.checks_passed) for r in ex), len(ex)),
            "budget_pass_rate": _rate(sum(r.budget["status"] == "pass" for r in eb), len(eb)),
            "mean_tokens": round(statistics.mean(r.tokens for r in ex), 1) if ex else None,
            "mean_steps": round(statistics.mean(r.steps for r in ex), 2) if ex else None,
        })
    n_pass = sum(r.passed for r in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "commit": meta.get("commit"), "run_id": meta.get("run_id"), "suite": meta.get("suite"),
        "provider": meta.get("provider"),
        "agent_model": meta.get("agent_model"), "judge_model": meta.get("judge_model"), "repeats": repeats, "isolated": isolated,
        "started_at": meta.get("started_at") or _now(), "finished_at": _now(),
        "baseline": {"loaded": baseline is not None, "path": baseline.path if baseline else None,
                     "commit": baseline.data.get("commit") if baseline else None,
                     "budget_check": ("active" if baseline else "token budget skipped: no baseline (steps still checked)")},
        "summary": {
            "total_runs": len(results), "executed": len(ran), "passed": n_pass,
            "failed": sum(r.status == "fail" for r in results), "skipped": sum(r.status == "skipped" for r in results),
            "errors": sum(r.status == "error" for r in results),
            "pass_rate": _rate(n_pass, len(ran)),
            "checks_pass_rate": _rate(sum(bool(r.checks_passed) for r in ran), len(ran)),
            "budget_pass_rate": _rate(sum(r.budget["status"] == "pass" for r in bud), len(bud)),
            "total_tokens": sum(r.tokens for r in ran),
        },
        "cases": per_case,
        "results": [r.to_dict() for r in results],
    }


def compare_reports(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Per-case status/pass-rate diff between two reports. Raises ModelConfigError if the reports were produced
    under different pinned agent/judge models (or either lacks them): the comparison would be meaningless."""
    models.assert_comparable(a, b)
    ca, cb = {c["case_id"]: c for c in a["cases"]}, {c["case_id"]: c for c in b["cases"]}
    return {cid: {"from": ca[cid]["status"], "to": cb[cid]["status"], "pass_rate": [ca[cid]["pass_rate"], cb[cid]["pass_rate"]]}
            for cid in ca.keys() & cb.keys() if ca[cid]["status"] != cb[cid]["status"]}
