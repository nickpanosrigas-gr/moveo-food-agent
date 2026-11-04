# tests/test_agent_loop.py
"""
Unit tests for FoodOrderAgent state machine and confirmation guardrails (src/core/agent_loop.py).
Tests premature submission blocking, duplicate submission guards, cold feet detection,
and strict internal vs exposed tool call contracts using a mocked LLM.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
import pytest

from agent import FoodOrderAgent


class ScriptedMockLLM:
    """Mock LLM client that returns pre-scripted assistant messages."""
    manages_vram: bool = False
    model: str = "mock-model"
    num_ctx: int | None = 4096

    def __init__(self, responses: list[dict[str, Any]] | None = None):
        self.responses: list[dict[str, Any]] = responses or []
        self.call_count: int = 0
        self.history_records: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> dict[str, Any]:
        self.history_records.append(messages)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return {"role": "assistant", "content": "Anything else?"}

    def warmup_kv_cache_sync(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "ok"}

    async def warmup_kv_cache(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "ok"}

    def unload_model_sync(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def unload_model(self) -> dict[str, Any]:
        return {"status": "ok"}

    def close(self) -> None:
        pass


@pytest.fixture
def agent() -> FoodOrderAgent:
    a = FoodOrderAgent()
    # Mock remote MCP submission by default
    a.mcp_client.submit_order = MagicMock(return_value={
        "success": True,
        "order_id": "ORD-12345",
        "total": 8.50,
        "estimated_time": "15-20 minutes"
    })
    return a


# =====================================================================
# 1. Premature Submission Intercept & Confirmation Protocol
# =====================================================================

def test_premature_submission_intercepted(agent: FoodOrderAgent) -> None:
    # Pre-populate cart with an item
    agent.cart.add_item("classic_burger")
    assert agent.awaiting_confirmation is False

    # LLM prematurely tries to call submit_order on turn 1
    mock_llm = ScriptedMockLLM([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_premature",
                    "type": "function",
                    "function": {"name": "submit_order", "arguments": {}}
                }
            ]
        },
        {
            "role": "assistant",
            "content": "Your total is $8.50 for one Classic Burger. Ready to submit?"
        }
    ])
    agent.llm = mock_llm

    response = agent.send("Submit my order")

    # 1. MCP client was NOT called
    agent.mcp_client.submit_order.assert_not_called()

    # 2. Guardrail set awaiting_confirmation = True
    assert agent.awaiting_confirmation is True

    # 3. Tool call was NOT exposed to the user
    assert "tool_calls" not in response or len(response.get("tool_calls", [])) == 0

    # 4. Agent prompted user for confirmation
    assert "8.50" in response["message"]


def test_confirmed_submission_allowed(agent: FoodOrderAgent) -> None:
    agent.cart.add_item("classic_burger")
    agent.awaiting_confirmation = True

    mock_llm = ScriptedMockLLM([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_confirm",
                    "type": "function",
                    "function": {"name": "submit_order", "arguments": {}}
                }
            ]
        },
        {
            "role": "assistant",
            "content": "Order submitted! Order #ORD-12345."
        }
    ])
    agent.llm = mock_llm

    response = agent.send("Yes")

    # 1. MCP client WAS called
    agent.mcp_client.submit_order.assert_called_once()

    # 2. Exposed tool calls contain submit_order
    assert "tool_calls" in response
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["name"] == "submit_order"
    assert response["tool_calls"][0]["result"]["success"] is True

    # 3. Cart is marked as submitted
    assert agent.cart.is_submitted is True
    assert agent.awaiting_confirmation is False


# =====================================================================
# 2. Duplicate Submission Guard
# =====================================================================

def test_duplicate_submission_in_same_turn_blocked(agent: FoodOrderAgent) -> None:
    agent.cart.add_item("classic_burger")
    agent.awaiting_confirmation = True

    # LLM emits two submit_order tool calls in the same turn
    mock_llm = ScriptedMockLLM([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_sub_1",
                    "type": "function",
                    "function": {"name": "submit_order", "arguments": {}}
                },
                {
                    "id": "call_sub_2",
                    "type": "function",
                    "function": {"name": "submit_order", "arguments": {}}
                }
            ]
        },
        {
            "role": "assistant",
            "content": "Order submitted! Order #ORD-12345."
        }
    ])
    agent.llm = mock_llm

    response = agent.send("Yes")

    # submit_order must be called exactly once
    assert agent.mcp_client.submit_order.call_count == 1
    assert len(response["tool_calls"]) == 1


# =====================================================================
# 3. Empty Cart Submission Guard
# =====================================================================

def test_empty_cart_submission_rejected(agent: FoodOrderAgent) -> None:
    assert len(agent.cart.lines) == 0

    mock_llm = ScriptedMockLLM([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_empty_sub",
                    "type": "function",
                    "function": {"name": "submit_order", "arguments": {}}
                }
            ]
        },
        {
            "role": "assistant",
            "content": "Your cart is empty. What would you like to order?"
        }
    ])
    agent.llm = mock_llm

    agent.send("Submit")
    agent.mcp_client.submit_order.assert_not_called()
    assert agent.cart.is_submitted is False


# =====================================================================
# 4. Post-Confirmation "Cold Feet"
# =====================================================================

def test_post_confirmation_cold_feet_resets_confirmation(agent: FoodOrderAgent) -> None:
    agent.cart.add_item("classic_burger")
    agent.awaiting_confirmation = True

    # User says "Yes, and add a cola" -> LLM adds soda
    mock_llm = ScriptedMockLLM([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_add_soda",
                    "type": "function",
                    "function": {"name": "add_item", "arguments": {"item_id": "soda"}}
                }
            ]
        },
        {
            "role": "assistant",
            "content": "Added a Medium Cola. Your new total is $10.50. Ready to submit?"
        }
    ])
    agent.llm = mock_llm

    agent.send("Yes, and add a cola")

    # Cart was modified this turn -> awaiting_confirmation reset to False
    assert agent.cart.modified_in_current_turn is True
    agent.mcp_client.submit_order.assert_not_called()
    assert len(agent.cart.lines) == 2


# =====================================================================
# 5. Duplicate Failed Tool Call Loop Prevention
# =====================================================================

def test_duplicate_failed_tool_call_loop_broken(agent: FoodOrderAgent) -> None:
    # LLM tries calling add_item with an invalid option twice in a row
    invalid_call = {
        "id": "call_bad",
        "type": "function",
        "function": {"name": "add_item", "arguments": {"item_id": "spicy_burger", "options": {"patty": "chicken"}}}
    }

    mock_llm = ScriptedMockLLM([
        {"role": "assistant", "content": "", "tool_calls": [invalid_call]},
        {"role": "assistant", "content": "", "tool_calls": [invalid_call]},
        {"role": "assistant", "content": "Sorry, Spicy Burger only comes with beef."}
    ])
    agent.llm = mock_llm

    agent.send("Give me a spicy burger with chicken")

    # The internal tool trace records the blocked duplicate before history pruning
    internal_actions = agent.last_turn_log["internal_cart_actions"]
    assert len(internal_actions) >= 2
    assert internal_actions[1]["result"]["success"] is False
    assert "DUPLICATE_CALL_BLOCKED" in internal_actions[1]["result"]["error"]


# =====================================================================
# 6. Internal vs. Exposed Tool Calls Contract
# =====================================================================

def test_internal_tools_not_exposed_in_response(agent: FoodOrderAgent) -> None:
    mock_llm = ScriptedMockLLM([
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_add",
                    "type": "function",
                    "function": {"name": "add_item", "arguments": {"item_id": "classic_burger"}}
                }
            ]
        },
        {
            "role": "assistant",
            "content": "Added a Classic Burger. Anything else?"
        }
    ])
    agent.llm = mock_llm

    resp = agent.send("I want a burger")
    assert "tool_calls" not in resp or len(resp.get("tool_calls", [])) == 0
    assert resp["message"] == "Added a Classic Burger. Anything else?"


# =====================================================================
# 7. Structured Turn Logging
# =====================================================================

def test_structured_turn_logging(agent: FoodOrderAgent) -> None:
    mock_llm = ScriptedMockLLM([
        {
            "role": "assistant",
            "content": "Hello! What can I get for you today?"
        }
    ])
    agent.llm = mock_llm

    agent.send("Hello")
    turn_log = agent.last_turn_log

    assert turn_log is not None
    assert turn_log["user_message"] == "Hello"
    assert turn_log["agent_message"] == "Hello! What can I get for you today?"
    assert "cart_state" in turn_log
    assert turn_log["turn_id"] == 1