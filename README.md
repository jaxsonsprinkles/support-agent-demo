# support-agent-demo

Demo target agent for [AgentBisect](https://github.com/jaxsonsprinkles/agentbisect) (neatHack, Oct 2026): a tool-using customer-support agent for a small online store ("Northwind Goods").

**The regressions in this repo's history are seeded.** The baseline commit is clean. Later commits, written by a coding agent with Entire checkpoints, add real-looking features that carry three planted regressions so AgentBisect has something to find.

## Layout

- `agent/` loop (`loop.py`, max 12 model calls, leaving room to verify each reply with `search_help` and retry uncertain tool calls), system prompt (`prompt.py`), model providers (`provider.py`), neatlogs tracing (`tracing.py`), CLI (`__main__.py`)
- `tools/` five mocked tools with Anthropic tool-use schemas: `lookup_order`, `search_help`, `issue_refund`, `cancel_order`, `escalate_to_human`
- `data/` seed orders (12) and help articles. Dates are fixed relative to `PINNED_TODAY = 2026-10-01`, so runs are repeatable
- `evals/` 24 eval cases, harness and `python -m evals` CLI (see `evals/README.md`)
- `tests/` pytest suite using `ScriptedProvider` (no network, no API key)

## Run

```
uv sync
export ANTHROPIC_API_KEY=...        # required for real runs
uv run python -m agent "Where is order 1042?"   # prints reply, tool calls, steps, tokens, trace id, total wall-clock `latency_ms` as JSON
uv run python -m agent --verbose "Where is order 1042?"   # also lists each tool call + result on stderr
uv run pytest
```

Env vars: `AGENT_PROVIDER` (default `anthropic`; extension point in `agent/provider.py` for Bedrock or the `claude` CLI, not implemented), `AGENT_MODEL` (pinned, default `claude-haiku-5-5`), `JUDGE_MODEL` (eval judge, default `claude-sonnet-5-5`, must outrank the agent), `NEATLOGS_API_KEY` (tracing is a no-op without it), `LOG_LEVEL` (`INFO` or `DEBUG` writes one JSON line per tool call to stderr with tool name, args and `duration_ms`; unset, `WARNING` or higher is silent), and run tags `AGENTBISECT_COMMIT`, `AGENTBISECT_RUN_ID`, `AGENTBISECT_CASE`, attached as `commit:`, `run:`, `case:` tags (process-wide, so run one case per process).

Refund policy (enforced in the tool and stated in the prompt): delivered within 30 days, not already refunded, amount at most the order total; `issue_refund` takes and returns integer US cents (order totals are `total_usd` dollars); refunds over $500 and angry customers go to `escalate_to_human`.

## Tools reference

Schemas and mock implementations live in `tools/impl.py`; every run gets a fresh `ToolBox`, so refunds and tickets never leak between runs.

| Tool | Input | Returns | Notes |
|---|---|---|---|
| `lookup_order` | `order_id` or `email` | `orders`: status, items, `total_usd`, ship/delivered dates, tracking, `estimated_delivery_date` | Read-only. Accepts `#1042` / `order 1042`; email match ignores case and whitespace. ETA is ship date + 5 days for shipped orders, else `null` |
| `search_help` | `query` | top 3 `articles` (id, title, body) | Read-only. Covers returns and refunds, shipping times, warranty, payments, gift cards, address changes, international shipping and customs |
| `issue_refund` | `order_id`, `amount` (integer cents), `reason` | `refund_id`, `amount` (cents), `currency: "usd"` | Delivered within 30 days, not already refunded, amount ≤ order total. Floats and non-positive amounts are rejected. Over $500 (50000 cents): escalate instead |
| `cancel_order` | `order_id` | `confirmation` (`CXL-<id>`), `status: "cancelled"` | Only `processing` orders; shipped or delivered orders get a return instead. Irreversible |
| `escalate_to_human` | `summary`, `priority` (`low`/`normal`/`high`/`urgent`), optional `order_id` | `ticket_id`, `priority`, `summary` | Prefixes the summary with `Order <id>:` if the id isn't already in it |

Errors come back as `{"error": "..."}` rather than exceptions, so the model can read and recover from them.

## Changelog

Recent features, newest first (see `git log` for the full history):

- **Help center**: international shipping and customs, address changes and gift cards articles.
- **`escalate_to_human`**: optional `order_id`; ticket summaries always name the order.
- **`issue_refund`**: amounts are integer US cents in and out, matching Stripe; new tests for policy limits and escalation tickets.
- **Refund replies**: the agent tells customers refunds typically appear within 5 to 7 business days.
- **CLI**: total wall-clock `latency_ms` in the JSON output; `--verbose` prints each tool call and result on stderr.
- **`lookup_order`**: `estimated_delivery_date` for shipped orders; order ids normalised (`#1042`, `order 1042`); email lookup tolerant of case and whitespace.
- **Agent loop**: customer messages capped at 2000 characters (truncation is logged); the agent verifies replies with `search_help` and retries uncertain tool calls (max 12 model calls).
- **`cancel_order`**: new tool for unshipped (processing) orders, with edge-case tests (unknown, shipped, delivered, already cancelled).
- **Tone**: friendly opener and "anything else" closer; apologises and asks the customer to double-check unknown order ids.
- **Observability**: structured JSON logging for tool calls, gated by `LOG_LEVEL`.
- **Evals**: pinned agent and judge models with a strict structured judge, on top of the 24-case eval suite.

## Roles and model pinning (short)

Full version: `ROLES.md` in the agentbisect repo. Five separate roles: (1) the history-builder session writes this repo's commits (Entire on, never sees AgentBisect); (2) this support agent, the lab rat, pinned to Haiku (`AGENT_MODEL`, default `claude-haiku-5-5`); (3) the eval judge, pinned to a smarter model (`JUDGE_MODEL`, default `claude-sonnet-5-5`; code guard rejects same-model or haiku-tier judges), temperature 0, structured per-criterion verdict with rubric hash; (4) AgentBisect; (5) the Fixer, which sees only the culprit diff, intent and failing evidence, never history-builder briefs or ground truth. `agent_model`/`judge_model` are written into every eval result and results from different pins are not comparable. The `breaks` ground truth is not in this repo (`agentbisect/evidence/ground_truth.json`).
