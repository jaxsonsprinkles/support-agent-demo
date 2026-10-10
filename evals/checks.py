"""Declarative deterministic checks. Each check name maps to a function (spec, run) -> (passed, reason).

`run` is an agent.loop.RunResult (or anything with .reply, .tool_calls, .hit_step_limit).
"""
from __future__ import annotations

import re
from typing import Any, Callable

WRITE_TOOLS = ("issue_refund", "escalate_to_human", "cancel_order")


def _names(run) -> list[str]:
    return [c["name"] for c in run.tool_calls]


def _norm(v: Any) -> Any:
    return v.lstrip("#").strip() if isinstance(v, str) else v


def _match_value(actual: Any, matcher: Any) -> bool:
    """Literal (numbers compared within half a cent, strings exactly after stripping '#'), or one of
    {"in": [...]}, {"regex": p}, {"contains": s}, {"gte": n}, {"lte": n}."""
    if isinstance(matcher, dict):
        (op, arg), = matcher.items()
        if op == "in":
            return any(_match_value(actual, a) for a in arg)
        if op == "regex":
            return actual is not None and re.search(arg, str(actual), re.I) is not None
        if op == "contains":
            return actual is not None and str(arg).lower() in str(actual).lower()
        if op in ("gte", "lte"):
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                return False
            return actual >= arg if op == "gte" else actual <= arg
        raise ValueError(f"unknown arg matcher '{op}'")
    if isinstance(matcher, (int, float)) and not isinstance(matcher, bool):
        return isinstance(actual, (int, float)) and abs(float(actual) - float(matcher)) < 0.005
    return _norm(actual) == _norm(matcher)


def _c_must_call(spec, run):
    missing = [n for n in spec if n not in _names(run)]
    return not missing, f"missing calls: {missing}" if missing else f"called {spec}"


def _c_must_not_call(spec, run):
    bad = [n for n in spec if n in _names(run)]
    return not bad, f"forbidden calls made: {bad}" if bad else f"none of {spec} called"


def _c_only_calls(spec, run):
    extra = sorted({n for n in _names(run) if n not in spec})
    return not extra, f"unexpected tools used: {extra}" if extra else f"only {spec} used"


def _c_first_call(spec, run):
    names = _names(run)
    return bool(names) and names[0] == spec, f"first call was {names[0] if names else None}, expected {spec}"


def _c_call_sequence(spec, run):
    it = iter(_names(run))
    ok = all(any(n == want for n in it) for want in spec)
    return ok, f"calls {_names(run)} {'contain' if ok else 'do not contain'} ordered sequence {spec}"


def _c_call_counts(spec, run):
    names = _names(run)
    bad = {k: names.count(k) for k, v in spec.items() if names.count(k) != v}
    return not bad, f"wrong call counts (actual): {bad}, expected {spec}" if bad else f"counts ok {spec}"


def _c_max_calls(spec, run):
    n = len(run.tool_calls)
    return n <= spec, f"{n} tool calls (max {spec})"


def _c_call_args(spec, run):
    """spec: list of {"tool": name, "args": {k: matcher}}; each entry needs at least one matching call."""
    problems = []
    for want in spec:
        cands = [c for c in run.tool_calls if c["name"] == want["tool"]]
        if not any(all(_match_value(c["args"].get(k), m) for k, m in want["args"].items()) for c in cands):
            problems.append(f"no {want['tool']} call matching {want['args']} (saw {[c['args'] for c in cands]})")
    return not problems, "; ".join(problems) or "arguments match"


def _c_reply_contains(spec, run):
    miss = [s for s in spec if s.lower() not in run.reply.lower()]
    return not miss, f"reply missing {miss}" if miss else "reply contains all"


def _c_reply_contains_any(spec, run):
    ok = any(s.lower() in run.reply.lower() for s in spec)
    return ok, f"reply contains one of {spec}" if ok else f"reply contains none of {spec}"


def _c_reply_not_contains(spec, run):
    bad = [s for s in spec if s.lower() in run.reply.lower()]
    return not bad, f"reply contains forbidden {bad}" if bad else "reply clean"


def _c_reply_regex(spec, run):
    miss = [p for p in spec if not re.search(p, run.reply, re.I)]
    return not miss, f"reply does not match {miss}" if miss else "reply matches all patterns"


def _c_reply_not_regex(spec, run):
    bad = [p for p in spec if re.search(p, run.reply, re.I)]
    return not bad, f"reply matches forbidden {bad}" if bad else "reply avoids forbidden patterns"


def _c_finished(spec, run):
    return (not run.hit_step_limit) == bool(spec), "hit the step limit" if run.hit_step_limit else "finished"


CHECKS: dict[str, Callable[[Any, Any], tuple[bool, str]]] = {
    "must_call": _c_must_call, "must_not_call": _c_must_not_call, "only_calls": _c_only_calls,
    "first_call": _c_first_call, "call_sequence": _c_call_sequence, "call_counts": _c_call_counts,
    "max_calls": _c_max_calls, "call_args": _c_call_args,
    "reply_contains": _c_reply_contains, "reply_contains_any": _c_reply_contains_any,
    "reply_not_contains": _c_reply_not_contains, "reply_regex": _c_reply_regex,
    "reply_not_regex": _c_reply_not_regex, "finished": _c_finished,
}


def validate_checks(checks: dict[str, Any]) -> None:
    for name in checks:
        if name not in CHECKS:
            raise ValueError(f"unknown check '{name}' (known: {sorted(CHECKS)})")


def run_checks(checks: dict[str, Any], run) -> list[dict[str, Any]]:
    """Run every declared check (plus the implicit `finished`). Returns [{check, passed, reason}]."""
    spec = {"finished": True, **checks}
    out = []
    for name, s in spec.items():
        passed, reason = CHECKS[name](s, run)
        out.append({"check": name, "passed": bool(passed), "reason": reason})
    return out
