"""Structured JSON logs to stderr. Off unless LOG_LEVEL is INFO or lower (LOG_LEVEL=INFO or DEBUG)."""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any


def _level() -> int:
    """LOG_LEVEL from env (name like INFO/debug, or a number). Unset or invalid means WARNING, i.e. no tool logs."""
    raw = os.environ.get("LOG_LEVEL", "").strip()
    if raw.isdigit():
        return int(raw)
    lvl = logging.getLevelName(raw.upper()) if raw else logging.WARNING
    return lvl if isinstance(lvl, int) else logging.WARNING


def log_tool_call(name: str, args: dict[str, Any], duration_ms: float) -> None:
    """Emit one JSON line per tool call at INFO. Logging must never break the agent."""
    try:
        if _level() > logging.INFO:
            return
        line = json.dumps({"level": "INFO", "event": "tool_call", "tool": name, "args": args,
                           "duration_ms": round(duration_ms, 3)}, default=str)
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
