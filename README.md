# support-agent-demo

Demo target agent for [AgentBisect](https://github.com/jaxsonsprinkles/agentbisect) (neatHack, Oct 2026): a tool-using customer-support agent for a small online store ("Northwind Goods").

**The regressions in this repo's history are seeded.** The baseline commit is clean. Later commits, written by a coding agent with Entire checkpoints, add real-looking features that carry three planted regressions so AgentBisect has something to find.

## Layout

- `agent/` loop (`loop.py`, max 6 model calls), system prompt (`prompt.py`), model providers (`provider.py`), neatlogs tracing (`tracing.py`), CLI (`__main__.py`)
- `tools/` four mocked tools with Anthropic tool-use schemas: `lookup_order`, `search_help`, `issue_refund`, `escalate_to_human`
- `data/` seed orders (12) and help articles. Dates are fixed relative to `PINNED_TODAY = 2026-10-01`, so runs are repeatable
- `evals/` 24 eval cases, harness and `python -m evals` CLI (see `evals/README.md`)
- `tests/` pytest suite using `ScriptedProvider` (no network, no API key)

## Run

```
uv sync
export ANTHROPIC_API_KEY=...        # required for real runs
uv run python -m agent "Where is order 1042?"   # prints reply, tool calls, steps, tokens, trace id as JSON
uv run pytest
```

Env vars: `AGENT_PROVIDER` (default `anthropic`; extension point in `agent/provider.py` for Bedrock or the `claude` CLI, not implemented), `AGENT_MODEL` (pinned, default `claude-haiku-5-5`), `JUDGE_MODEL` (eval judge, default `claude-sonnet-5-5`, must outrank the agent), `NEATLOGS_API_KEY` (tracing is a no-op without it), `LOG_LEVEL` (`INFO` or `DEBUG` writes one JSON line per tool call to stderr with tool name, args and `duration_ms`; unset, `WARNING` or higher is silent), and run tags `AGENTBISECT_COMMIT`, `AGENTBISECT_RUN_ID`, `AGENTBISECT_CASE`, attached as `commit:`, `run:`, `case:` tags (process-wide, so run one case per process).

Refund policy (enforced in the tool and stated in the prompt): delivered within 30 days, not already refunded, amount at most the order total; amounts are US dollars; refunds over $500 and angry customers go to `escalate_to_human`.

## Roles and model pinning (short)

Full version: `ROLES.md` in the agentbisect repo. Five separate roles: (1) the history-builder session writes this repo's commits (Entire on, never sees AgentBisect); (2) this support agent, the lab rat, pinned to Haiku (`AGENT_MODEL`, default `claude-haiku-5-5`); (3) the eval judge, pinned to a smarter model (`JUDGE_MODEL`, default `claude-sonnet-5-5`; code guard rejects same-model or haiku-tier judges), temperature 0, structured per-criterion verdict with rubric hash; (4) AgentBisect; (5) the Fixer, which sees only the culprit diff, intent and failing evidence, never history-builder briefs or ground truth. `agent_model`/`judge_model` are written into every eval result and results from different pins are not comparable. The `breaks` ground truth is not in this repo (`agentbisect/evidence/ground_truth.json`).
