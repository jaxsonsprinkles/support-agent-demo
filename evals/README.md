# evals

24 deterministic eval cases (`cases.jsonl`: 18 main, 6 `held_out`) plus a harness that runs them against the
agent and emits a stable JSON report for AgentBisect. Only `ScriptedProvider` has been used to build and test
this; no live model run has been made.

## Cases

Fields: `id`, `category`, `input`, `checks`, `breaks` (`R1`|`R2`|`R3`|`control`, the primary regression expected
to break it), `held_out`, `requires_tool` (case is reported `skipped` if the commit's ToolBox lacks the tool;
`cancel_order` does not exist at baseline), `intent_for` (regression whose fix must keep it passing), `rubric`
(optional LLM-judge text).

| Category | Ids | breaks | Held out |
| --- | --- | --- | --- |
| order_status (5) | os1-os5 | R1 | os5 |
| refund_within_policy (4) | rw1-rw4 | R3 | rw4 |
| refund_outside_policy (3) | ro1-ro3 | control | ro3 |
| cancel_request (3, needs cancel_order) | ca1-ca3 | control, `intent_for: R1` | ca3 |
| policy_question (4) | po1-po4 | R2 (budget) | po4 |
| escalation (3) | es1-es3 | R2 (budget) | es3 |
| multi_step (2, needs cancel_order) | ms1-ms2 | R1 | none |

Checks (all declarative, in `checks.py`): `must_call`, `must_not_call`, `only_calls`, `first_call`,
`call_sequence` (ordered subsequence), `call_counts`, `max_calls`, `call_args` (list of `{tool, args}`; arg
values are literals, numbers within half a cent, or `{in|regex|contains|gte|lte}`), `reply_contains`,
`reply_contains_any`, `reply_not_contains`, `reply_regex`, `reply_not_regex` (case-insensitive). `finished`
(did not hit the step limit) is always added. Order-status cases forbid write tools rather than allowing only
`lookup_order`, so an extra read-only `search_help` (R2) shows up as a budget failure, not a check failure.

## Budget

A run's budget fails if `steps > 6` or `tokens > 1.5 x` the case's median in `evals/baseline.json`. The case
fails if its checks fail or its budget fails; `checks_passed` and `budget.status` are reported separately so
a budget-only regression (R2) is distinguishable. With no baseline file the token check is skipped and
reported (`baseline.budget_check`, `budget.token_check`, `budget.status = "partial"`); steps are still
enforced. Write the baseline from a known-good commit:

```
python -m evals calibrate --repeats 3 [--out evals/baseline.json]
```

It uses only runs whose deterministic checks passed.

## CLI

```
python -m evals run --suite main|held_out|all --repeats N [--cases os1,po1] [--json] [--out report.json]
                    [--isolate] [--commit SHA] [--run-id ID] [--baseline path] [--judge] [--strict]
python -m evals calibrate ...      # writes baseline.json
python -m evals list
```

`--json` prints only the report on stdout (progress goes to stderr). Exit code is 0 unless `--strict` and
something failed. `--provider module:callable` (callable(case) -> Provider) and `--toolbox module:Class`
override the live provider and the default `ToolBox`. `evals/reference.py` has scripted `good`/`r1`/`r2`/`r3`
providers and a `ReferenceToolBox` with `cancel_order` for self-testing (not a real model).

Tracing tags: `AGENTBISECT_COMMIT`, `AGENTBISECT_RUN_ID`, `AGENTBISECT_CASE` are read once by
`neatlogs.init`, so they are process-wide. An in-process suite run carries one `case:` tag at most. Use
`--isolate` to run every (case, repeat) in its own subprocess with `AGENTBISECT_CASE` set to that case id and
commit/run id inherited. A crashed or timed-out child becomes a `status: "error"` result.

## Result contract (`schema_version: 1`)

```
{
  "schema_version": 1, "commit": "sha|null", "run_id": "str|null", "suite": "main|held_out|all",
  "provider": "str", "repeats": 3, "isolated": false, "started_at": "iso8601", "finished_at": "iso8601",
  "baseline": {"loaded": bool, "path": "str|null", "commit": "str|null", "budget_check": "active|<skipped note>"},
  "summary": {"total_runs", "executed", "passed", "failed", "skipped", "errors",
              "pass_rate", "checks_pass_rate", "budget_pass_rate", "total_tokens"},   // rates: null if no denominator
  "cases": [{"case_id", "category", "held_out", "breaks",
             "status": "pass|fail|flaky|skipped|error", "runs", "passed", "failed", "skipped", "errors",
             "pass_rate", "checks_pass_rate", "budget_pass_rate", "mean_tokens", "mean_steps"}],
  "results": [{"case_id", "category", "held_out", "breaks", "repeat",
               "status": "pass|fail|skipped|error", "passed": bool,
               "checks": [{"check", "passed", "reason"}], "checks_passed": bool|null,
               "budget": {"status": "pass|fail|partial", "steps", "max_steps", "tokens", "token_limit",
                          "baseline_median_tokens", "token_check", "reason"},
               "judge": {"passed", "reason"}|null, "reply", "tool_calls": [{"name","args","result"}],
               "steps", "input_tokens", "output_tokens", "tokens", "trace_id", "duration_s", "reason"}]
}
```

`pass_rate` is passed / executed (skipped and error runs are excluded from the denominator). `error` means an
infrastructure failure (provider exception, crashed child): treat it as "no verdict", not "bad commit".
`flaky` means mixed pass/fail across repeats. `baseline.json` holds `{commit, created_at, provider,
token_factor, max_steps, cases: {id: {median_tokens, median_steps, n}}}`.

## Judge

`judge.py` is off by default. With `--judge` (or `judge=` in `run_case`), cases that have a `rubric` and whose
deterministic checks already passed are graded by an LLM; a failed judge verdict fails the case. Only es3 has a
rubric.
