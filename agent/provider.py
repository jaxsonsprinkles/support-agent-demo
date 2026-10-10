"""Model provider layer.

A Provider turns (system, messages, tools) into a normalized Response. Select with env AGENT_PROVIDER
("anthropic" is the default; "scripted" is for tests and is constructed directly).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-haiku-5-5"  # the support agent is PINNED to Haiku tier; see evals/models.py


@dataclass
class Response:
    """Normalized model response. `content` is a list of Anthropic-style blocks:
    {"type": "text", "text": ...} or {"type": "tool_use", "id": ..., "name": ..., "input": {...}}."""

    content: list[dict[str, Any]]
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b["type"] == "text")

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b["type"] == "tool_use"]


class Provider(Protocol):
    def create(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
               max_tokens: int = 1024) -> Response: ...


class AnthropicProvider:
    """Anthropic Messages API. Key from ANTHROPIC_API_KEY, model from AGENT_MODEL, temperature 0."""

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        if client is None:
            from anthropic import Anthropic

            from agent.tracing import wrap_client

            client = wrap_client(Anthropic())  # reads ANTHROPIC_API_KEY
        self.client = client
        self.model = model or os.environ.get("AGENT_MODEL", DEFAULT_MODEL)

    def create(self, *, system, messages, tools, max_tokens=1024) -> Response:
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens,
            extra_body={"temperature": 0},  # SDK 1.x dropped the typed kwarg; the API still accepts it
            system=system, messages=messages, tools=tools,
        )
        blocks: list[dict[str, Any]] = []
        for b in r.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
        return Response(blocks, r.stop_reason or "end_turn", r.usage.input_tokens, r.usage.output_tokens)


@dataclass
class ScriptedProvider:
    """Replays a fixed list of Responses (for tests). Records every call in `calls`."""

    script: list[Response]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, *, system, messages, tools, max_tokens=1024) -> Response:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        if not self.script:
            raise RuntimeError("ScriptedProvider script exhausted")
        return self.script.pop(0)


# EXTENSION POINT: add other backends (AWS Bedrock, `claude` CLI, ...) as classes implementing
# Provider.create and register them in _FACTORIES. Not implemented on purpose.
_FACTORIES = {"anthropic": AnthropicProvider}


def get_provider(name: str | None = None) -> Provider:
    name = (name or os.environ.get("AGENT_PROVIDER", "anthropic")).lower()
    if name not in _FACTORIES:
        raise ValueError(f"unknown AGENT_PROVIDER '{name}'; available: {sorted(_FACTORIES)}")
    return _FACTORIES[name]()
