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

Env vars: `AGENT_PROVIDER` (default `anthropic`; extension point in `agent/provider.py` for Bedrock or the `claude` CLI, not implemented), `AGENT_MODEL` (default `claude-haiku-4-5`), `NEATLOGS_API_KEY` (tracing is a no-op without it), and run tags `AGENTBISECT_COMMIT`, `AGENTBISECT_RUN_ID`, `AGENTBISECT_CASE`, attached as `commit:`, `run:`, `case:` tags (process-wide, so run one case per process).

Refund policy (enforced in the tool and stated in the prompt): delivered within 30 days, not already refunded, amount at most the order total; amounts are US dollars; refunds over $500 and angry customers go to `escalate_to_human`.
