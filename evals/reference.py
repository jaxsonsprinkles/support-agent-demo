"""Scripted reference behaviours for every case. NOT a real model: this is a self-test for the harness and a
sanity oracle for the planted regressions (what a good agent, and agents with R1/R2/R3, would do).

Factories `good`, `r1`, `r2`, `r3` are callable(case) -> ScriptedProvider, usable as
`python -m evals run --provider evals.reference:good --toolbox evals.reference:ReferenceToolBox`.
"""
from __future__ import annotations

import re
from typing import Any

from agent.provider import Response, ScriptedProvider
from tools import ToolBox, TOOL_SCHEMAS

CANCEL_SCHEMA = {
    "name": "cancel_order",
    "description": "Cancel an order that has not shipped yet.",
    "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
}


class ReferenceToolBox(ToolBox):
    """Baseline toolbox plus cancel_order, as the R1 commit will add it."""

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return super().schemas + [CANCEL_SCHEMA]

    def _tool_cancel_order(self, order_id: str) -> dict[str, Any]:
        o = self.orders.get(str(order_id))
        if not o:
            return {"error": f"order '{order_id}' not found"}
        if o["status"] != "processing":
            return {"error": f"order {order_id} is {o['status']}; only unshipped orders can be cancelled"}
        o["status"] = "cancelled"
        return {"confirmation": f"CXL-{order_id}", "order_id": order_id, "status": "cancelled"}


def L(oid): return ("lookup_order", {"order_id": oid})
def S(q): return ("search_help", {"query": q})


# id -> (tool calls, final reply, order id used by the R1 side effect)
REFERENCE: dict[str, tuple[list[tuple[str, dict]], str, str | None]] = {
    "os1": ([L("1042")], "Order 1042 (Mechanical Keyboard) has shipped. Your tracking number is TRK-884201.", "1042"),
    "os2": ([L("1007")], "Order 1007 was delivered on 2026-09-26 (tracking TRK-660418).", "1007"),
    "os3": ([L("1050")], "Order 1050 is still processing and has not shipped yet.", "1050"),
    "os4": ([("lookup_order", {"email": "dev.patel@example.com"})],
            "Your order 1023 (Running Shoes) shipped on 2026-09-29. Tracking number: TRK-771930.", "1023"),
    "os5": ([L("1060")], "Order 1060 was delivered on 2026-09-30.", "1060"),
    "rw1": ([L("1007"), ("issue_refund", {"order_id": "1007", "amount": 64.0, "reason": "arrived broken"})],
            "I've refunded $64.00 for order 1007. It returns to your original payment method within 5 business days.", None),
    "rw2": ([L("1015"), ("issue_refund", {"order_id": "1015", "amount": 79.99, "reason": "defective"})],
            "I've refunded $79.99 for order 1015 to your original payment method.", None),
    "rw3": ([L("1031"), ("issue_refund", {"order_id": "1031", "amount": 59.0, "reason": "does not work"})],
            "I've refunded $59.00 for order 1031 to your original payment method.", None),
    "rw4": ([L("1060"), ("issue_refund", {"order_id": "1060", "amount": 36.0, "reason": "damaged"})],
            "I've refunded $36.00 for order 1060 to your original payment method.", None),
    "ro1": ([L("0991")], "Order 0991 was delivered more than 30 days ago, outside our 30-day refund window, so I can't refund it. A human teammate can review exceptions.", None),
    "ro2": ([L("0975")], "Order 0975 was delivered over 30 days ago, outside our 30-day refund window, so I can't refund it.", None),
    "ro3": ([L("1012")], "Order 1012 was already refunded (RF-0001), so I can't refund it again.", None),
    "ca1": ([L("1050"), ("cancel_order", {"order_id": "1050"})], "Order 1050 had not shipped, so I've cancelled it.", None),
    "ca2": ([L("1042")], "Order 1042 has already shipped (TRK-884201), so I can't cancel it. Once it arrives you can return it within 30 days.", None),
    "ca3": ([L("1001"), ("cancel_order", {"order_id": "1001"})], "Order 1001 had not shipped, so I've cancelled it.", None),
    "po1": ([S("return window")], "Our return window is 30 days from delivery.", None),
    "po2": ([S("standard shipping time")], "Standard shipping takes 3-5 business days.", None),
    "po3": ([S("warranty")], "All products carry a 1-year limited warranty against manufacturing defects.", None),
    "po4": ([S("payment methods")], "We accept major credit cards and PayPal.", None),
    "es1": ([("escalate_to_human", {"summary": "Customer asking a third time for a manager", "priority": "high"})],
            "I'm sorry about the repeated delays. I've escalated this to a human manager.", None),
    "es2": ([L("1088"), ("escalate_to_human", {"summary": "Full refund request on 1088 ($800)", "priority": "high"})],
            "Refunds over $500 need a human to review, so I've escalated order 1088 to our team.", None),
    "es3": ([L("0991"), ("escalate_to_human", {"summary": "Furious customer asks for an exception on 0991", "priority": "urgent"})],
            "I'm very sorry. I've escalated this to a human who will follow up about an exception.", None),
    "ms1": ([L("1042")], "Yes, order 1042 has already shipped (tracking TRK-884201), so it can no longer be cancelled.", None),
    "ms2": ([L("1050"), ("cancel_order", {"order_id": "1050"})], "Order 1050 had not shipped, so I've cancelled it.", None),
}

IN_TOK, OUT_TOK = 200, 30


def _script(calls: list[tuple[str, dict]], reply: str, tok_mult: int = 1) -> ScriptedProvider:
    script = [Response([{"type": "tool_use", "id": f"t{i}", "name": n, "input": a}], "tool_use",
                       IN_TOK * tok_mult, OUT_TOK * tok_mult) for i, (n, a) in enumerate(calls)]
    script.append(Response([{"type": "text", "text": reply}], "end_turn", IN_TOK * tok_mult, OUT_TOK * tok_mult))
    return ScriptedProvider(script)


def good(case) -> ScriptedProvider:
    calls, reply, _ = REFERENCE[case.id]
    return _script(calls, reply)


def r1(case) -> ScriptedProvider:
    """cancel_order described as 'any order question': status and multi-step questions get a cancel."""
    calls, reply, oid = REFERENCE[case.id]
    if case.category in ("order_status", "multi_step"):
        oid = oid or re.search(r"\b(\d{4})\b", case.input).group(1)
        return _script([("cancel_order", {"order_id": oid})], f"I've cancelled order {oid}.")
    return _script(calls, reply)


def r2(case) -> ScriptedProvider:
    """'More thorough': repeats the first read-only call twice more and spends 3x the tokens."""
    calls, reply, _ = REFERENCE[case.id]
    first = next((c for c in calls if c[0] in ("lookup_order", "search_help")), None)
    extra = [first, first] if first else []
    return _script(calls[:1] + extra + calls[1:] if first == calls[0] else extra + calls, reply, tok_mult=3)


def r3(case) -> ScriptedProvider:
    """Cents: refund amounts are x100 and the reply quotes the inflated dollar figure."""
    calls, reply, _ = REFERENCE[case.id]
    if case.category != "refund_within_policy":
        return _script(calls, reply)
    new = [(n, {**a, "amount": round(a["amount"] * 100, 2)} if n == "issue_refund" else a) for n, a in calls]
    amt = next(a["amount"] for n, a in new if n == "issue_refund")
    return _script(new, f"I've refunded ${amt:,.2f} for order {new[0][1]['order_id']}.")
