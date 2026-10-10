"""neatlogs instrumentation. Every function no-ops if NEATLOGS_API_KEY is missing or neatlogs fails."""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import uuid
from typing import Any, Callable, Iterator

WORKFLOW_NAME = "support-agent"
CASE_SPAN = "support_case"  # name of the per-case parent span opened by the harness / CLI
_enabled = False
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def run_tags() -> list[str]:
    """Tags from env: AGENTBISECT_COMMIT, AGENTBISECT_RUN_ID, AGENTBISECT_CASE."""
    out = []
    for env, key in (("AGENTBISECT_COMMIT", "commit"), ("AGENTBISECT_RUN_ID", "run"), ("AGENTBISECT_CASE", "case")):
        if os.environ.get(env):
            out.append(f"{key}:{os.environ[env]}")
    return out


def init_tracing() -> bool:
    """Call once per process, before creating the model client. Tags are process-wide in neatlogs."""
    global _enabled
    if _enabled:
        return True
    if not os.environ.get("NEATLOGS_API_KEY"):
        return False
    try:
        import neatlogs

        neatlogs.init(api_key=os.environ["NEATLOGS_API_KEY"], workflow_name=WORKFLOW_NAME, tags=run_tags())
        _enabled = True
    except Exception:  # tracing must never break the agent
        _enabled = False
    return _enabled


def wrap_client(client: Any) -> Any:
    if not _enabled:
        return client
    import neatlogs

    return neatlogs.wrap(client)


class _NullSpan:
    def set_attribute(self, *a: Any, **k: Any) -> None: ...
    def record_exception(self, *a: Any, **k: Any) -> None: ...


@contextlib.contextmanager
def span(name: str, kind: str = "CHAIN", **attributes: Any) -> Iterator[Any]:
    """Public neatlogs `trace()` context manager: a child span of the active trace, or a new root trace.

    Yields an OTel span (or a no-op stand-in when tracing is off). Attribute values must be primitives."""
    if not _enabled:
        yield _NullSpan()
        return
    import neatlogs

    with neatlogs.trace(name, kind=kind, **attributes) as sp:
        token = None
        try:  # neatlogs uses its own tracer provider, so the OTel global current span is not set: remember ours
            ctx = sp.get_span_context()
            if ctx.is_valid:
                token = _trace_id.set(format(ctx.trace_id, "032x"))
        except Exception:
            pass
        try:
            yield sp
        finally:
            if token is not None:
                _trace_id.reset(token)


def set_json(sp: Any, key: str, value: Any) -> None:
    """Attach a JSON-serialised (truncated) attribute to a span; never raises."""
    try:
        sp.set_attribute(key, json.dumps(value, default=str)[:4000])
    except Exception:
        pass


def traced(fn: Callable) -> Callable:
    """Run an agent run inside an `agent.run` span (kind AGENT). It nests under the caller's `support_case`
    span (eval harness / CLI) so everything lands in one trace; with no parent it becomes the root."""
    def inner(*a, **kw):
        with span("agent.run", "AGENT"):
            return fn(*a, **kw)
    inner.__name__ = fn.__name__
    return inner


def current_trace_id() -> str:
    """Hex OpenTelemetry trace id of the active neatlogs span, or a local uuid when tracing is off."""
    if _trace_id.get():
        return _trace_id.get()
    try:
        from opentelemetry import trace

        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return "local-" + uuid.uuid4().hex[:16]


def shutdown_tracing() -> None:
    global _enabled
    if not _enabled:
        return
    try:
        import neatlogs

        neatlogs.flush()
        neatlogs.shutdown()
    except Exception:
        pass
    _enabled = False
