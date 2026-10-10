"""CLI: python -m evals run|calibrate|list. See evals/README.md."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from evals import models
from evals.cases import CATEGORY_COUNTS, load_cases
from evals.harness import BASELINE_PATH, REPO_ROOT, Baseline, build_baseline, run_suite, _now, CaseResult


def _import_attr(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evals")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "calibrate"):
        s = sub.add_parser(name)
        s.add_argument("--suite", choices=["main", "held_out", "all"], default="all" if name == "calibrate" else "main")
        s.add_argument("--repeats", type=int, default=3 if name == "calibrate" else 1)
        s.add_argument("--cases", help="comma-separated case ids")
        s.add_argument("--json", action="store_true", help="print the JSON report to stdout")
        s.add_argument("--out", help="also write the report (run) / baseline (calibrate) to this path")
        s.add_argument("--isolate", action="store_true", help="one subprocess per case run (per-case trace tags)")
        s.add_argument("--provider", help="'module:callable(case)->Provider'; default: agent.provider.get_provider()")
        s.add_argument("--toolbox", help="'module:Class' ToolBox factory (default tools.ToolBox)")
        s.add_argument("--commit", help="sets AGENTBISECT_COMMIT (default: git HEAD)")
        s.add_argument("--run-id", help="sets AGENTBISECT_RUN_ID")
        s.add_argument("--baseline", help=f"baseline file (default {BASELINE_PATH.name})")
    sub.choices["run"].add_argument("--judge", action="store_true", help="enable the optional LLM judge rubric")
    sub.choices["run"].add_argument("--strict", action="store_true", help="exit 1 if any run did not pass")
    sub.add_parser("list")
    return p


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    if a.cmd == "list":
        for c in load_cases():
            print(f"{c.id}\t{c.category}\t{'held_out' if c.held_out else 'main'}\t{c.input}")
        return 0
    if a.commit:
        os.environ["AGENTBISECT_COMMIT"] = a.commit
    if a.run_id:
        os.environ["AGENTBISECT_RUN_ID"] = a.run_id
    commit = os.environ.get("AGENTBISECT_COMMIT") or _git_commit()
    ids = [x for x in (a.cases or "").split(",") if x] or None
    cases = load_cases(suite=a.suite, ids=ids)
    pins = models.pinned()
    models.check_judge_model(pins["judge_model"], pins["agent_model"])  # fail loudly on a bad pin, even without --judge
    baseline = None if a.cmd == "calibrate" else Baseline.load(a.baseline)
    meta = {"commit": commit, "run_id": os.environ.get("AGENTBISECT_RUN_ID"), "suite": a.suite,
            "provider": a.provider or os.environ.get("AGENT_PROVIDER", "anthropic"), "started_at": _now(), **pins}

    child_args = [x for flag, val in (("--provider", a.provider), ("--toolbox", a.toolbox), ("--baseline", a.baseline))
                  if val for x in (flag, val)]
    if a.cmd == "run" and a.judge:
        child_args.append("--judge")
    provider = judge = None
    if not a.isolate:
        if a.provider:
            provider = _import_attr(a.provider)
        else:
            from agent.tracing import init_tracing
            init_tracing()  # before the client is created so it can be wrapped
            from agent.provider import get_provider
            provider = get_provider()
        if a.cmd == "run" and a.judge:
            from agent.provider import AnthropicProvider
            from evals.judge import LLMJudge
            judge = LLMJudge(AnthropicProvider(model=pins["judge_model"]), agent_model=pins["agent_model"])
    toolbox_factory = _import_attr(a.toolbox) if a.toolbox else None

    def progress(r: CaseResult) -> None:
        print(f"{r.case_id:5} #{r.repeat} {r.status:7} steps={r.steps} tokens={r.tokens} {r.reason or ''}", file=sys.stderr)

    try:
        report = run_suite(cases, provider, repeats=a.repeats, isolate=a.isolate, toolbox_factory=toolbox_factory,
                           baseline=baseline, judge=judge, meta=meta, child_args=child_args, progress=progress)
    finally:
        if not a.isolate and not a.provider:
            from agent.tracing import shutdown_tracing
            shutdown_tracing()

    if a.cmd == "calibrate":
        base = build_baseline([CaseResult.from_dict(r) for r in report["results"]], meta)
        out = Path(a.out or BASELINE_PATH)
        out.write_text(json.dumps(base, indent=2) + "\n")
        print(f"wrote baseline for {len(base['cases'])} cases to {out}", file=sys.stderr)
        missing = [r["case_id"] for r in report["cases"] if r["case_id"] not in base["cases"]]
        if missing:
            print(f"WARNING: no passing runs for {missing}; they have no token baseline", file=sys.stderr)
        return 0
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    s = report["summary"]
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"pass_rate={s['pass_rate']} passed={s['passed']}/{s['executed']} skipped={s['skipped']} "
              f"errors={s['errors']} budget={report['baseline']['budget_check']}")
        for c in report["cases"]:
            print(f"  {c['case_id']:5} {c['category']:22} {c['status']}")
    return 1 if a.strict and (s["failed"] or s["errors"]) else 0
