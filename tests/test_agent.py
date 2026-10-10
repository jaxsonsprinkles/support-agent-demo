import json

import pytest

from agent.loop import MAX_STEPS, RunResult, run_agent
from agent.provider import Response, ScriptedProvider, get_provider
from agent.prompt import SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, ToolBox


def tu(name, args, id="t1"):
    return Response([{"type": "tool_use", "id": id, "name": name, "input": args}], "tool_use", 10, 5)


def text(t):
    return Response([{"type": "text", "text": t}], "end_turn", 20, 8)


# --- tools ---
def test_schemas_are_anthropic_format_and_baseline_set():
    assert [s["name"] for s in TOOL_SCHEMAS] == ["lookup_order", "search_help", "issue_refund", "escalate_to_human"]
    for s in TOOL_SCHEMAS:
        assert s["description"] and s["input_schema"]["type"] == "object"


def test_lookup_by_id_and_email():
    tb = ToolBox()
    assert tb.call("lookup_order", {"order_id": "1042"})["orders"][0]["tracking_number"] == "TRK-884201"
    assert len(tb.call("lookup_order", {"email": "KAI.NGUYEN@example.com"})["orders"]) == 1
    assert "error" in tb.call("lookup_order", {"order_id": "9999"})
    assert "error" in tb.call("lookup_order", {})


@pytest.mark.parametrize("email", [
    "kai.nguyen@example.com", "KAI.NGUYEN@EXAMPLE.COM", "  Kai.Nguyen@Example.com  ",
    "\tkai.nguyen@example.com\n", "kai.nguyen @ example.com",
])
def test_lookup_by_email_ignores_case_and_whitespace(email):
    orders = ToolBox().call("lookup_order", {"email": email})["orders"]
    assert [o["customer_email"].lower() for o in orders] == ["kai.nguyen@example.com"]


def test_lookup_by_blank_or_unknown_email_errors():
    tb = ToolBox()
    assert tb.call("lookup_order", {"email": "   "}) == {"error": "provide order_id or email"}
    assert tb.call("lookup_order", {"email": " nobody@example.com "}) == {"error": "no orders for 'nobody@example.com'"}


def test_search_help_top3_and_relevance():
    r = ToolBox().call("search_help", {"query": "what is your return window?"})
    assert len(r["articles"]) == 3 and r["articles"][0]["id"] == "returns"
    assert "30 days" in r["articles"][0]["body"]
    assert ToolBox().call("search_help", {"query": "warranty"})["articles"][0]["id"] == "warranty"


# Top 3 for returns/shipping/warranty/payment questions, recorded before the gift-cards article existed.
# New articles must not change these results.
POLICY_SEARCH_BASELINE = {
    "What's your return window?": ["returns", "warranty", "payments"],
    "How long does standard shipping take?": ["shipping", "payments", "returns"],
    "Do your products come with a warranty?": ["warranty", "returns", "shipping"],
    "What payment methods do you accept?": ["payments", "returns", "warranty"],
    "warranty": ["warranty", "payments", "returns"],
    "shipping": ["shipping", "payments", "returns"],
    "refund": ["returns", "payments", "shipping"],
    "Can I return a broken item?": ["returns", "warranty", "shipping"],
    "my item arrived damaged": ["returns", "shipping", "payments"],
    "can I get my money back": ["returns", "shipping", "payments"],
    "how do I track my package": ["shipping", "payments", "returns"],
    "when will my order arrive": ["shipping", "payments", "returns"],
    "is accidental damage covered": ["warranty", "payments", "returns"],
    "do you accept credit cards": ["payments", "returns", "warranty"],
    "can I pay with PayPal": ["payments", "returns", "shipping"],
    "can I pay by invoice": ["payments", "returns", "shipping"],
    "can I exchange an item": ["returns", "shipping", "payments"],
    "how long do refunds take": ["returns", "shipping", "payments"],
    "can I cancel my order": ["shipping", "returns", "payments"],
    "express shipping": ["shipping", "payments", "returns"],
    "what does the warranty cover": ["warranty", "payments", "returns"],
    "do you take visa card payments": ["payments", "returns", "shipping"],
    "I want a refund for my order": ["returns", "shipping", "payments"],
    "return policy": ["returns", "payments", "shipping"],
    "repair my product": ["warranty", "payments", "returns"],
    "how do I pay": ["payments", "returns", "shipping"],
    "delivery time": ["shipping", "returns", "payments"],
}


@pytest.mark.parametrize("query,expected", POLICY_SEARCH_BASELINE.items())
def test_search_help_policy_results_unchanged(query, expected):
    assert [a["id"] for a in ToolBox().call("search_help", {"query": query})["articles"]] == expected


def test_search_help_card_query_only_changes_filler_slot():
    # "card" genuinely matches "Gift cards", so it takes slot 3 from returns, which scored 0 here.
    ids = [a["id"] for a in ToolBox().call("search_help", {"query": "when is my card charged"})["articles"]]
    assert ids == ["payments", "shipping", "gift-cards"]


@pytest.mark.parametrize("query", [
    "How do I buy a gift card?", "how do I redeem a gift card", "Do gift cards expire?",
    "what's the balance on my gift card", "can I use a voucher",
])
def test_search_help_finds_gift_cards(query):
    top = ToolBox().call("search_help", {"query": query})["articles"][0]
    assert top["id"] == "gift-cards" and "never expire" in top["body"]


def test_refund_ok_in_dollars_and_partial():
    tb = ToolBox()
    r = tb.call("issue_refund", {"order_id": "1007", "amount": 64.0, "reason": "broken"})
    assert r["amount_usd"] == 64.0 and r["refund_id"].startswith("RF-")
    assert "already refunded" in tb.call("issue_refund", {"order_id": "1007", "amount": 1, "reason": "x"})["error"]
    assert ToolBox().call("issue_refund", {"order_id": "1015", "amount": 20, "reason": "x"})["amount_usd"] == 20


@pytest.mark.parametrize("oid,amt,frag", [
    ("0991", 88.0, "refund window"),       # delivered 45 days ago
    ("1012", 25.0, "already refunded"),
    ("1042", 129.0, "not delivered"),      # shipped
    ("1001", 45.0, "not delivered"),       # processing
    ("1007", 64.01, "exceeds order total"),
    ("1007", 0, "positive"),
    ("4242", 5, "not found"),
])
def test_refund_policy_enforced(oid, amt, frag):
    assert frag in ToolBox().call("issue_refund", {"order_id": oid, "amount": amt, "reason": "r"})["error"]


def test_escalate_deterministic_tickets():
    tb = ToolBox()
    assert tb.call("escalate_to_human", {"summary": "s", "priority": "high"})["ticket_id"] == "TKT-0001"
    assert tb.call("escalate_to_human", {"summary": "s", "priority": "low"})["ticket_id"] == "TKT-0002"
    assert "error" in tb.call("escalate_to_human", {"summary": "s", "priority": "meh"})


def test_unknown_tool_and_bad_args():
    tb = ToolBox()
    assert "unknown tool" in tb.call("cancel_order", {"order_id": "1"})["error"]
    assert "bad arguments" in tb.call("search_help", {})["error"]


def test_toolbox_state_is_per_instance():
    ToolBox().call("issue_refund", {"order_id": "1007", "amount": 64, "reason": "r"})
    assert "error" not in ToolBox().call("issue_refund", {"order_id": "1007", "amount": 64, "reason": "r"})


# --- loop ---
def test_run_result_shape_and_tool_flow():
    p = ScriptedProvider([tu("lookup_order", {"order_id": "1042"}), text("It shipped: TRK-884201")])
    r = run_agent("Where is order 1042?", p)
    assert isinstance(r, RunResult)
    assert r.reply == "It shipped: TRK-884201"
    assert [c["name"] for c in r.tool_calls] == ["lookup_order"]
    assert r.tool_calls[0]["args"] == {"order_id": "1042"}
    assert r.tool_calls[0]["result"]["orders"][0]["status"] == "shipped"
    assert (r.steps, r.input_tokens, r.output_tokens) == (2, 30, 13)
    assert r.trace_id and not r.hit_step_limit
    assert set(r.to_dict()) == {"reply", "tool_calls", "steps", "input_tokens", "output_tokens", "trace_id", "hit_step_limit"}
    # tool result was fed back to the model, system prompt and tools passed
    assert p.calls[1]["messages"][-1]["content"][0]["type"] == "tool_result"
    assert p.calls[0]["system"] == SYSTEM_PROMPT and len(p.calls[0]["tools"]) == 4


def test_refund_through_loop():
    p = ScriptedProvider([tu("lookup_order", {"order_id": "1007"}),
                          tu("issue_refund", {"order_id": "1007", "amount": 64.0, "reason": "broken"}, "t2"),
                          text("Refunded $64.00")])
    r = run_agent("1007 arrived broken", p)
    assert r.tool_calls[1]["result"]["amount_usd"] == 64.0 and r.steps == 3


def test_step_limit():
    assert MAX_STEPS == 6
    p = ScriptedProvider([tu("lookup_order", {"order_id": "1042"}, f"t{i}") for i in range(20)])
    r = run_agent("loop forever", p)
    assert r.steps == MAX_STEPS and r.hit_step_limit and len(r.tool_calls) == MAX_STEPS and r.reply
    assert len(p.script) == 20 - MAX_STEPS


def test_max_steps_override():
    p = ScriptedProvider([tu("lookup_order", {"order_id": "1042"}, f"t{i}") for i in range(5)])
    assert run_agent("x", p, max_steps=2).steps == 2


def test_parallel_tool_calls_in_one_step():
    resp = Response([{"type": "tool_use", "id": "a", "name": "lookup_order", "input": {"order_id": "1042"}},
                     {"type": "tool_use", "id": "b", "name": "search_help", "input": {"query": "shipping"}}], "tool_use", 1, 1)
    r = run_agent("x", ScriptedProvider([resp, text("ok")]))
    assert [c["name"] for c in r.tool_calls] == ["lookup_order", "search_help"] and r.steps == 2


# --- structured logging ---
def _two_tool_run():
    resp = Response([{"type": "tool_use", "id": "a", "name": "lookup_order", "input": {"order_id": "1042"}},
                     {"type": "tool_use", "id": "b", "name": "search_help", "input": {"query": "shipping"}}], "tool_use", 1, 1)
    return run_agent("x", ScriptedProvider([resp, text("ok")]))


@pytest.mark.parametrize("level", ["INFO", "debug", "20"])
def test_tool_calls_logged_as_json_to_stderr(monkeypatch, capsys, level):
    monkeypatch.setenv("LOG_LEVEL", level)
    r = _two_tool_run()
    out, err = capsys.readouterr()
    assert out == ""
    lines = [json.loads(l) for l in err.splitlines()]
    assert [(l["tool"], l["args"]) for l in lines] == [("lookup_order", {"order_id": "1042"}), ("search_help", {"query": "shipping"})]
    assert all(l["event"] == "tool_call" and isinstance(l["duration_ms"], float) and l["duration_ms"] >= 0 for l in lines)
    assert r.reply == "ok" and len(r.tool_calls) == 2


@pytest.mark.parametrize("level", [None, "WARNING", "ERROR", "bogus"])
def test_tool_logs_silent_by_default_and_above_info(monkeypatch, capsys, level):
    if level is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", level)
    r = _two_tool_run()
    assert capsys.readouterr() == ("", "")
    assert r.reply == "ok" and len(r.tool_calls) == 2


def test_get_provider_rejects_unknown():
    with pytest.raises(ValueError):
        get_provider("bedrock")
