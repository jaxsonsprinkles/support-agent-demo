import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.provider import Response, ScriptedProvider
from evals import checks as ck
from evals.cases import CATEGORY_COUNTS, Case, load_cases
from evals.harness import Baseline, budget_check, build_baseline, run_case, run_suite
from evals.judge import LLMJudge
from evals import reference as ref
from tools import ToolBox

ROOT = Path(__file__).resolve().parent.parent
RTB = ref.ReferenceToolBox


def run(reply="", calls=(), hit=False):
    return SimpleNamespace(reply=reply, tool_calls=[{"name": n, "args": a, "result": {}} for n, a in calls],
                           hit_step_limit=hit)


def one(name, spec, r):
    return ck.CHECKS[name](spec, r)[0]


# ---- cases file
def test_split_sizes_and_categories():
    allc, main, held = load_cases(), load_cases(suite="main"), load_cases(suite="held_out")
    assert (len(allc), len(main), len(held)) == (24, 18, 6)
    assert Counter(c.category for c in allc) == CATEGORY_COUNTS
    assert Counter(c.category for c in held) == {k: 1 for k in CATEGORY_COUNTS if k != "multi_step"}
    assert len({c.id for c in allc}) == 24


def test_breaks_tags_and_requires_tool():
    by = {c.id: c for c in load_cases()}
    for c in by.values():
        assert c.breaks == {"order_status": "R1", "multi_step": "R1", "refund_within_policy": "R3",
                            "policy_question": "R2", "escalation": "R2"}.get(c.category, "control")
        assert (c.requires_tool == "cancel_order") == (c.category in ("cancel_request", "multi_step"))


def test_load_filters_and_validation(tmp_path):
    assert [c.id for c in load_cases(ids=["os1", "ro1"])] == ["os1", "ro1"]
    with pytest.raises(ValueError):
        load_cases(suite="main", ids=["os5"])  # held out
    bad = tmp_path / "c.jsonl"
    bad.write_text(json.dumps(dict(id="x", category="order_status", input="i", breaks="R1", checks={"nope": 1})))
    with pytest.raises(ValueError, match="unknown check"):
        load_cases(bad)


# ---- each check type, pass and fail
@pytest.mark.parametrize("name,spec,r,expected", [
    ("must_call", ["lookup_order"], run(calls=[("lookup_order", {})]), True),
    ("must_call", ["lookup_order"], run(calls=[("search_help", {})]), False),
    ("must_not_call", ["issue_refund"], run(calls=[("lookup_order", {})]), True),
    ("must_not_call", ["issue_refund"], run(calls=[("issue_refund", {})]), False),
    ("only_calls", ["lookup_order"], run(calls=[("lookup_order", {})] * 2), True),
    ("only_calls", ["lookup_order"], run(calls=[("lookup_order", {}), ("search_help", {})]), False),
    ("first_call", "lookup_order", run(calls=[("lookup_order", {}), ("issue_refund", {})]), True),
    ("first_call", "lookup_order", run(calls=[("issue_refund", {}), ("lookup_order", {})]), False),
    ("first_call", "lookup_order", run(), False),
    ("call_sequence", ["lookup_order", "cancel_order"], run(calls=[("lookup_order", {}), ("x", {}), ("cancel_order", {})]), True),
    ("call_sequence", ["lookup_order", "cancel_order"], run(calls=[("cancel_order", {}), ("lookup_order", {})]), False),
    ("call_counts", {"cancel_order": 1}, run(calls=[("cancel_order", {})]), True),
    ("call_counts", {"cancel_order": 1}, run(calls=[("cancel_order", {})] * 2), False),
    ("call_counts", {"cancel_order": 0}, run(), True),
    ("max_calls", 2, run(calls=[("a", {})] * 2), True),
    ("max_calls", 2, run(calls=[("a", {})] * 3), False),
    ("reply_contains", ["TRK-1", "Shipped"], run("shipped, TRK-1"), True),
    ("reply_contains", ["TRK-1"], run("nothing"), False),
    ("reply_contains_any", ["a", "b"], run("has B"), True),
    ("reply_contains_any", ["a", "b"], run("none"), False),
    ("reply_not_contains", ["sorry"], run("fine"), True),
    ("reply_not_contains", ["sorry"], run("SORRY"), False),
    ("reply_regex", [r"\$64(\.00)?"], run("refunded $64.00"), True),
    ("reply_regex", [r"\$64(\.00)?(?![\d,])"], run("refunded $6,400.00"), False),
    ("reply_not_regex", [r"6,?400"], run("$64"), True),
    ("reply_not_regex", [r"6,?400"], run("$6,400"), False),
    ("finished", True, run(hit=False), True),
    ("finished", True, run(hit=True), False),
])
def test_check_types(name, spec, r, expected):
    assert one(name, spec, r) is expected


@pytest.mark.parametrize("matcher,value,expected", [
    (64.0, 64, True), (64.0, 6400.0, False), (64.0, "64", False), ("1007", "#1007", True), ("1007", "1008", False),
    ({"in": ["high", "urgent"]}, "urgent", True), ({"in": ["high", "urgent"]}, "low", False),
    ({"regex": "^TKT"}, "TKT-1", True), ({"contains": "manager"}, "Wants MANAGER", True),
    ({"gte": 10}, 12, True), ({"gte": 10}, 3, False), ({"lte": 10}, 12, False), ({"gte": 1}, "x", False),
])
def test_call_args_matchers(matcher, value, expected):
    r = run(calls=[("t", {"k": value})])
    assert one("call_args", [{"tool": "t", "args": {"k": matcher}}], r) is expected


def test_call_args_missing_tool_and_run_checks_adds_finished():
    assert not one("call_args", [{"tool": "t", "args": {"k": 1}}], run())
    res = ck.run_checks({"must_call": ["a"]}, run(calls=[("a", {})]))
    assert [c["check"] for c in res] == ["finished", "must_call"] and all(c["passed"] for c in res)
    with pytest.raises(ValueError):
        ck.validate_checks({"bogus": 1})


# ---- run_case
def case(id):
    return next(c for c in load_cases() if c.id == id)


def go(id, factory=ref.good, **kw):
    kw.setdefault("toolbox_factory", RTB)
    c = case(id)
    return run_case(c, factory(c), **kw)


def test_run_case_pass_records_everything():
    r = go("os1")
    assert r.status == "pass" and r.passed and r.checks_passed
    assert r.steps == 2 and r.tokens == 460 and r.trace_id and r.reply.startswith("Order 1042")
    assert r.tool_calls[0]["result"]["orders"][0]["tracking_number"] == "TRK-884201"
    assert r.budget["status"] == "partial" and "no baseline" in r.budget["token_check"]
    d = r.to_dict()
    assert d["passed"] is True and d["category"] == "order_status"


def test_run_case_check_failure_has_reasons():
    r = go("os1", ref.r1)
    assert r.status == "fail" and not r.checks_passed
    failed = {c["check"]: c["reason"] for c in r.checks if not c["passed"]}
    assert {"first_call", "must_call", "must_not_call", "call_args", "reply_contains"} <= set(failed)
    assert "cancel_order" in failed["must_not_call"]


def test_run_case_skips_when_tool_missing():
    c = case("ca1")
    r = run_case(c, ref.good(c))  # baseline ToolBox has no cancel_order
    assert r.status == "skipped" and "cancel_order" in r.reason and r.steps == 0 and not r.passed
    assert "cancel_order" not in {s["name"] for s in ToolBox().schemas}


def test_run_case_error_on_provider_failure():
    r = run_case(case("os1"), ScriptedProvider([]), toolbox_factory=RTB)
    assert r.status == "error" and "exhausted" in r.reason


def test_step_limit_marks_unfinished():
    c = case("os1")
    loop = ScriptedProvider([Response([{"type": "tool_use", "id": f"t{i}", "name": "lookup_order",
                                        "input": {"order_id": "1042"}}], "tool_use", 1, 1) for i in range(9)])
    r = run_case(c, loop, toolbox_factory=RTB)
    assert r.status == "fail" and r.steps == 6
    assert not next(x for x in r.checks if x["check"] == "finished")["passed"]


# ---- budget
def test_budget_check_rules():
    base = Baseline({"cases": {"os1": {"median_tokens": 100, "median_steps": 2}}})
    assert budget_check(2, 150, "os1", base)["status"] == "pass"     # exactly 1.5x
    b = budget_check(2, 151, "os1", base)
    assert b["status"] == "fail" and "151 tokens" in b["reason"] and b["token_limit"] == 150
    assert budget_check(7, 10, "os1", base)["status"] == "fail"      # steps
    assert budget_check(6, 10, "os1", base)["status"] == "pass"
    assert budget_check(2, 10**6, "os1", None)["status"] == "partial"   # skipped, reported
    assert budget_check(7, 1, "os1", None)["status"] == "fail"           # steps still enforced
    assert "not in baseline" in budget_check(2, 5, "zz", base)["token_check"]


def good_baseline():
    results = [go(c.id) for c in load_cases()]
    return Baseline(build_baseline(results, {"commit": "good", "provider": "scripted"}))


def test_calibrate_median_and_budget_regression():
    b = good_baseline()
    assert b.cases["os1"]["median_tokens"] == 460 and b.cases["rw1"]["median_steps"] == 3
    assert go("os1", baseline=b).status == "pass" and go("os1", baseline=b).budget["status"] == "pass"
    r = go("os1", ref.r2, baseline=b)  # same answer, 3x tokens
    assert r.checks_passed and r.budget["status"] == "fail" and r.status == "fail"


def test_baseline_load_absent_and_present(tmp_path):
    assert Baseline.load(tmp_path / "nope.json") is None
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"commit": "abc", "cases": {"os1": {"median_tokens": 10}}}))
    assert Baseline.load(p).token_limit("os1") == 15


# ---- suite aggregation
def suite(factory, **kw):
    kw.setdefault("toolbox_factory", RTB)
    return run_suite(load_cases(), factory, **kw)


def test_suite_good_all_pass_and_schema():
    rep = suite(ref.good, repeats=2, meta={"commit": "c1", "run_id": "r1", "suite": "all"})
    s = rep["summary"]
    assert (s["total_runs"], s["passed"], s["pass_rate"]) == (48, 48, 1.0)
    assert rep["schema_version"] == 1 and rep["commit"] == "c1" and rep["repeats"] == 2
    assert rep["baseline"]["loaded"] is False and "skipped" in rep["baseline"]["budget_check"]
    assert len(rep["cases"]) == 24 and all(c["runs"] == 2 and c["status"] == "pass" for c in rep["cases"])
    assert {"case_id", "status", "checks", "budget", "trace_id", "tokens", "repeat"} <= set(rep["results"][0])
    json.dumps(rep)


def test_suite_baseline_toolbox_skips_cancel_and_multistep():
    rep = run_suite(load_cases(), ref.good)
    skipped = {c["case_id"] for c in rep["cases"] if c["status"] == "skipped"}
    assert skipped == {"ca1", "ca2", "ca3", "ms1", "ms2"}
    assert rep["summary"]["skipped"] == 5 and rep["summary"]["executed"] == 19 and rep["summary"]["pass_rate"] == 1.0


def failing(rep, key="status"):
    return {c["case_id"] for c in rep["cases"] if c["status"] == "fail"}


def test_expected_failing_sets_per_regression():
    base = good_baseline()
    os_ids, ms_ids = {f"os{i}" for i in range(1, 6)}, {"ms1", "ms2"}
    rw_ids = {f"rw{i}" for i in range(1, 5)}
    assert failing(suite(ref.r1, baseline=base)) == os_ids | ms_ids
    assert failing(suite(ref.r3, baseline=base)) == rw_ids
    r2 = suite(ref.r2, baseline=base)
    assert failing(r2) == {c.id for c in load_cases()}            # budget on every case
    assert r2["summary"]["checks_pass_rate"] == 1.0 and r2["summary"]["budget_pass_rate"] == 0.0


def test_flaky_status_and_rates():
    cases = [case("os1")]
    plays = iter([ref.good, ref.r1])
    rep = run_suite(cases, lambda c: next(plays)(c), repeats=2, toolbox_factory=RTB)
    c = rep["cases"][0]
    assert c["status"] == "flaky" and c["pass_rate"] == 0.5 and rep["summary"]["pass_rate"] == 0.5


def test_progress_callback_and_errors_counted():
    seen = []
    rep = run_suite([case("os1")], lambda c: ScriptedProvider([]), toolbox_factory=RTB, progress=seen.append)
    assert len(seen) == 1 and rep["summary"]["errors"] == 1 and rep["summary"]["pass_rate"] is None
    assert rep["cases"][0]["status"] == "error"


# ---- judge (off by default, after deterministic checks)
def text(t):
    return Response([{"type": "text", "text": t}], "end_turn", 5, 5)


def test_judge_only_on_rubric_cases_after_checks():
    c = case("es3")
    assert c.rubric
    assert go("es3").judge is None  # off by default
    j = LLMJudge(ScriptedProvider([text('{"passed": false, "reason": "promised refund"}')]))
    r = go("es3", judge=j)
    assert r.status == "fail" and r.judge == {"passed": False, "reason": "promised refund"}
    j = LLMJudge(ScriptedProvider([text('ok {"passed": true, "reason": "good"}')]))
    assert go("es3", judge=j).status == "pass"
    assert go("os1", judge=LLMJudge(ScriptedProvider([]))).judge is None  # no rubric: judge not called
    broken = LLMJudge(ScriptedProvider([text("???")]))
    assert go("es3", judge=broken).judge["passed"] is False
    skipped_judge = LLMJudge(ScriptedProvider([]))  # would raise if called: checks failed, so judge must not run
    r = run_case(case("es3"), ScriptedProvider([text("hi")]), toolbox_factory=RTB, judge=skipped_judge)
    assert r.status == "fail" and r.judge is None


# ---- CLI & isolate
def cli(*args, env=None):
    import os
    p = subprocess.run([sys.executable, "-m", "evals", *args], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    return p


REF = ["--provider", "evals.reference:good", "--toolbox", "evals.reference:ReferenceToolBox"]


def test_cli_json_and_cases_filter():
    p = cli("run", "--suite", "all", "--cases", "os1,po1", "--repeats", "2", "--json", *REF, "--commit", "abc")
    rep = json.loads(p.stdout)
    assert p.returncode == 0 and rep["commit"] == "abc" and rep["summary"]["total_runs"] == 4
    assert cli("run", "--suite", "main", "--cases", "os5", *REF).returncode != 0  # held-out id in main suite


def test_cli_isolate_runs_subprocess_per_case_with_tags(tmp_path):
    p = cli("run", "--suite", "all", "--cases", "os1,rw1", "--repeats", "2", "--json", "--isolate", *REF,
            "--commit", "deadbeef", "--run-id", "run9")
    rep = json.loads(p.stdout)
    assert rep["isolated"] is True and rep["summary"]["passed"] == 4
    assert sorted(r["repeat"] for r in rep["results"]) == [0, 0, 1, 1]


def test_isolate_failure_becomes_error_result():
    from evals.harness import run_case_isolated
    r = run_case_isolated(case("os1"), 0, ["--provider", "nope.missing:x"], {"commit": "c"})
    assert r.status == "error" and "isolated run failed" in r.reason


def test_isolate_env_tags(monkeypatch):
    from evals import harness
    seen = {}

    def fake(cmd, env, **kw):
        seen.update(env=env, cmd=cmd)
        return SimpleNamespace(stdout=json.dumps({"results": [go("os1").to_dict()]}), stderr="")
    monkeypatch.setattr(harness.subprocess, "run", fake)
    r = harness.run_case_isolated(case("os1"), 3, [], {"commit": "abc", "run_id": "r7"})
    e = seen["env"]
    assert (e["AGENTBISECT_CASE"], e["AGENTBISECT_COMMIT"], e["AGENTBISECT_RUN_ID"]) == ("os1", "abc", "r7")
    assert r.repeat == 3 and r.passed


def test_calibrate_cli_writes_baseline(tmp_path):
    out = tmp_path / "baseline.json"
    p = cli("calibrate", "--repeats", "2", "--out", str(out), *REF)
    assert p.returncode == 0
    b = json.loads(out.read_text())
    assert len(b["cases"]) == 24 and b["cases"]["os1"]["median_tokens"] == 460 and b["cases"]["os1"]["n"] == 2
    p = cli("run", "--suite", "all", "--baseline", str(out), "--json", *REF)
    assert json.loads(p.stdout)["baseline"]["budget_check"] == "active"
