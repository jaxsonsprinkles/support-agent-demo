"""Deterministic mock implementations of the support tools, backed by data/*.json.

Dates are fixed relative to PINNED_TODAY (never the wall clock) so runs are repeatable.
"""
from __future__ import annotations

import copy
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

PINNED_TODAY = date(2026, 10, 1)
REFUND_WINDOW_DAYS = 30
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "lookup_order",
        "description": (
            "Look up a customer's order. Provide either order_id (e.g. '1042') or the customer's email address. "
            "Returns order status (processing, shipped, delivered, refunded), items, total in USD dollars, "
            "ship date, delivered date and tracking number. Read-only; never changes an order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. '1042'."},
                "email": {"type": "string", "description": "Customer email; returns all of that customer's orders."},
            },
        },
    },
    {
        "name": "search_help",
        "description": (
            "Search the help center and return the top 3 matching articles (returns and refunds, shipping times, "
            "warranty, payments, gift cards). Use for any policy question and quote the article text. Read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Plain-language search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "issue_refund",
        "description": (
            "Refund money to the customer for a delivered order. amount is in US dollars (e.g. 64.00 for $64). "
            "Allowed only if the order was delivered within the last 30 days, has not been refunded already, and "
            "amount does not exceed the order total. Refunds over $500 must be escalated instead. "
            "Returns a refund id and the refunded amount."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number", "description": "Refund amount in US dollars."},
                "reason": {"type": "string", "description": "Short reason for the refund."},
            },
            "required": ["order_id", "amount", "reason"],
        },
    },
    {
        "name": "cancel_order",
        "description": (
            "Cancel an order that has not shipped yet. Use only when the customer explicitly asks to cancel; "
            "look the order up first. Allowed only while the order status is processing; shipped or delivered "
            "orders cannot be cancelled (offer a return instead). This changes the order and cannot be undone. "
            "Returns a cancellation confirmation id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order id, e.g. '1050'."}},
            "required": ["order_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand the conversation to a human agent. Use for angry customers, policy exceptions, and any refund "
            "over $500. Provide a self-contained summary and a priority. Returns a ticket id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "What the customer wants and what has been done."},
                "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
            },
            "required": ["summary", "priority"],
        },
    },
]


def _normalize_email(email: str) -> str:
    """Case-insensitive, whitespace-free key; emails never legitimately contain whitespace."""
    return "".join(email.split()).casefold()


_ORDER_ID_PREFIX = re.compile(r"^\s*(?:order\b)?[\s#]*", re.IGNORECASE)


def _normalize_order_id(order_id: str) -> str:
    """Drop how customers write ids ('#1042', 'order 1042', ' 1042 '); leading zeros are significant."""
    return _ORDER_ID_PREFIX.sub("", str(order_id)).strip()


class ToolBox:
    """Per-run tool state (refunds issued, tickets opened). Create a fresh one for every run."""

    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {
            o["order_id"]: o for o in json.loads((DATA_DIR / "orders.json").read_text())
        }
        self.articles: list[dict[str, Any]] = json.loads((DATA_DIR / "articles.json").read_text())
        self._refund_seq = 100  # seed data uses RF-0001
        self._ticket_seq = 0

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return copy.deepcopy(TOOL_SCHEMAS)

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, f"_tool_{name}", None) if name in {s["name"] for s in self.schemas} else None
        if fn is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    # -- tools ---------------------------------------------------------
    def _tool_lookup_order(self, order_id: str | None = None, email: str | None = None) -> dict[str, Any]:
        oid = _normalize_order_id(order_id) if order_id else ""
        if oid:
            o = self.orders.get(oid)
            return {"orders": [o]} if o else {"error": f"order '{oid}' not found"}
        key = _normalize_email(email) if email else ""
        if key:
            found = [o for o in self.orders.values() if _normalize_email(o["customer_email"]) == key]
            return {"orders": found} if found else {"error": f"no orders for '{email.strip()}'"}
        return {"error": "provide order_id or email"}

    def _tool_search_help(self, query: str) -> dict[str, Any]:
        q = query.lower()
        words = {w for w in q.replace("?", " ").split() if len(w) > 2}

        def score(a: dict[str, Any]) -> int:
            s = sum(3 for k in a["keywords"] if k in q)
            s += sum(1 for w in words if w in a["body"].lower() or w in a["title"].lower())
            return s

        # Stable sort: ties keep articles.json order, so a newly appended article never displaces existing ones.
        ranked = sorted(self.articles, key=lambda a: -score(a))[:3]
        return {"articles": [{"id": a["id"], "title": a["title"], "body": a["body"]} for a in ranked]}

    def _tool_issue_refund(self, order_id: str, amount: float, reason: str) -> dict[str, Any]:
        o = self.orders.get(str(order_id))
        if not o:
            return {"error": f"order '{order_id}' not found"}
        if o["refund_id"]:
            return {"error": f"order {order_id} was already refunded ({o['refund_id']})"}
        if o["status"] != "delivered" or not o["delivered_date"]:
            return {"error": f"order {order_id} is {o['status']}, not delivered; cannot refund"}
        days = (PINNED_TODAY - date.fromisoformat(o["delivered_date"])).days
        if days > REFUND_WINDOW_DAYS:
            return {"error": f"order {order_id} was delivered {days} days ago; refund window is {REFUND_WINDOW_DAYS} days"}
        if not isinstance(amount, (int, float)) or amount <= 0:
            return {"error": "amount must be a positive number of US dollars"}
        if amount > o["total_usd"] + 1e-9:
            return {"error": f"amount ${amount:.2f} exceeds order total ${o['total_usd']:.2f}"}
        self._refund_seq += 1
        rid = f"RF-{self._refund_seq:04d}"
        o["refund_id"] = rid
        return {"refund_id": rid, "order_id": order_id, "amount_usd": round(float(amount), 2), "reason": reason}

    def _tool_cancel_order(self, order_id: str) -> dict[str, Any]:
        oid = _normalize_order_id(order_id)
        o = self.orders.get(oid)
        if not o:
            return {"error": f"order '{oid}' not found"}
        if o["status"] != "processing":
            return {"error": f"order {oid} is {o['status']}; only unshipped (processing) orders can be cancelled"}
        o["status"] = "cancelled"
        return {"confirmation": f"CXL-{oid}", "order_id": oid, "status": "cancelled"}

    def _tool_escalate_to_human(self, summary: str, priority: str) -> dict[str, Any]:
        if priority not in {"low", "normal", "high", "urgent"}:
            return {"error": "priority must be one of low, normal, high, urgent"}
        self._ticket_seq += 1
        return {"ticket_id": f"TKT-{self._ticket_seq:04d}", "priority": priority, "summary": summary}
