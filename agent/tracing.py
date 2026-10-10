"""neatlogs instrumentation. Every function no-ops if NEATLOGS_API_KEY is missing or neatlogs fails."""
from __future__ import annotations

import os
import uuid
from typing import Any, Callable

WORKFLOW_NAME = "support-agent"
_enabled = False


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


def traced(fn: Callable) -> Callable:
    """Group a whole agent run into one root span (one trace) when tracing is on.

    Uses neatlogs' internal tracer because @neatlogs.span did not expose a recordable OTel span context
    (so the trace id could not be read). Falls back to untraced if that private API changes."""
    def inner(*a, **kw):
        if not _enabled:
            return fn(*a, **kw)
        try:
            from neatlogs._wrap_utils import get_tracer

            cm = get_tracer().start_as_current_span(
                "support-agent.run", attributes={"openinference.span.kind": "AGENT"})
        except Exception:
            return fn(*a, **kw)
        with cm:
            return fn(*a, **kw)
    inner.__name__ = fn.__name__
    return inner


def current_trace_id() -> str:
    """Hex OpenTelemetry trace id of the active span, or a local uuid when tracing is off."""
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
