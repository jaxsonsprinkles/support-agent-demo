"""Structured JSON logs to stderr. Tool calls log at INFO (off unless LOG_LEVEL=INFO or DEBUG);
message truncation logs at WARNING (on by default, off at LOG_LEVEL=ERROR or above)."""
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


def log_message_truncated(original_len: int, max_len: int) -> None:
    """Emit one JSON line at WARNING when a customer message is cut to max_len. Never logs the message itself."""
    try:
        if _level() > logging.WARNING:
            return
        line = json.dumps({"level": "WARNING", "event": "message_truncated",
                           "original_length": original_len, "max_length": max_len})
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass


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
