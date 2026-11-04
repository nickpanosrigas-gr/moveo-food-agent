# tests/test_mcp.py
"""
Unit tests for MCPOrderClient (src/mcp/client.py).
Tests header construction, argument mutation for forced failures, and normalization
of all 3 MCP rejection modes (schema validation -32602, server error/transient, and business refusal).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from unittest.mock import AsyncMock, patch
import pytest

from src.mcp.client import MCPOrderClient


@dataclass
class MockTextContent:
    text: str


@dataclass
class MockCallToolResult:
    isError: bool = False
    structuredContent: dict[str, Any] | None = None
    content: list[Any] | None = None


@pytest.fixture
def mcp_client() -> MCPOrderClient:
    return MCPOrderClient(
        server_url="https://mock.mcp.test/api/demo/mcp",
        applicant_email="applicant@example.com",
    )


# =====================================================================
# 1. Header Construction & Identification
# =====================================================================

def test_build_headers_standard(mcp_client: MCPOrderClient) -> None:
    headers = mcp_client._build_headers()
    assert headers["X-Applicant-Email"] == "applicant@example.com"
    assert "X-Demo-Force-Failure" not in headers


def test_build_headers_force_kitchen_busy(mcp_client: MCPOrderClient) -> None:
    headers = mcp_client._build_headers(force_failure="kitchen_busy")
    assert headers["X-Applicant-Email"] == "applicant@example.com"
    assert headers["X-Demo-Force-Failure"] == "kitchen_busy"


def test_build_headers_force_server_error(mcp_client: MCPOrderClient) -> None:
    headers = mcp_client._build_headers(force_failure="server_error")
    assert headers["X-Applicant-Email"] == "applicant@example.com"
    assert headers["X-Demo-Force-Failure"] == "server_error"


# =====================================================================
# 2. Argument Preparation & Test Failure Injections
# =====================================================================

def test_prepare_mcp_arguments_standard(mcp_client: MCPOrderClient) -> None:
    items = [{"item_id": "classic_burger", "quantity": 1, "options": {}, "extras": []}]
    args = mcp_client._prepare_mcp_arguments(items, special_instructions="no onions")
    assert args["items"] == items
    assert args["special_instructions"] == "no onions"


def test_prepare_mcp_arguments_schema_error_injection(mcp_client: MCPOrderClient) -> None:
    items = [{"item_id": "classic_burger", "quantity": 1, "options": {}, "extras": []}]
    args = mcp_client._prepare_mcp_arguments(items, force_failure="schema_error")
    # Mutates quantity to 0 to trigger genuine -32602 on MCP server
    assert args["items"][0]["quantity"] == 0


def test_prepare_mcp_arguments_business_refusal_injection(mcp_client: MCPOrderClient) -> None:
    items = [{"item_id": "classic_burger", "quantity": 1, "options": {}, "extras": []}]
    args = mcp_client._prepare_mcp_arguments(items, force_failure="business_refusal")
    # Inflates quantity to at least 10 to exceed $50 cap
    assert args["items"][0]["quantity"] >= 10


# =====================================================================
# 3. All 3 Spec Rejection Modes Parsing
# =====================================================================

def test_rejection_mode_1_schema_validation_error(mcp_client: MCPOrderClient) -> None:
    error_payload = {
        "error": {
            "code": -32602,
            "message": "Invalid params: items.0.quantity must be >= 1"
        }
    }
    mock_result = MockCallToolResult(
        isError=True,
        structuredContent=None,
        content=[MockTextContent(text=json.dumps(error_payload))]
    )

    parsed = mcp_client._parse_mcp_sdk_result(mock_result)
    assert parsed["success"] is False
    assert "-32602" in parsed["error"]
    assert "items.0.quantity must be >= 1" in parsed["error"]


def test_rejection_mode_2_kitchen_busy(mcp_client: MCPOrderClient) -> None:
    busy_payload = {
        "success": False,
        "error": "Kitchen is currently busy. Please try again."
    }
    mock_result = MockCallToolResult(
        isError=False,
        structuredContent=busy_payload,
        content=[MockTextContent(text=json.dumps(busy_payload))]
    )

    parsed = mcp_client._parse_mcp_sdk_result(mock_result)
    assert parsed["success"] is False
    assert "Kitchen is currently busy" in parsed["error"]


def test_rejection_mode_2_server_error(mcp_client: MCPOrderClient) -> None:
    server_error_payload = {
        "success": False,
        "error": "Internal Server Error in kitchen pipeline."
    }
    mock_result = MockCallToolResult(
        isError=True,
        content=[MockTextContent(text=json.dumps(server_error_payload))]
    )

    parsed = mcp_client._parse_mcp_sdk_result(mock_result)
    assert parsed["success"] is False
    assert "Internal Server Error" in parsed["error"]


def test_rejection_mode_3_business_refusal_over_50_limit(mcp_client: MCPOrderClient) -> None:
    refusal_payload = {
        "success": False,
        "error": "Order total ($52.00) exceeds $50 limit."
    }
    mock_result = MockCallToolResult(
        isError=False,
        structuredContent=refusal_payload,
        content=[MockTextContent(text=json.dumps(refusal_payload))]
    )

    parsed = mcp_client._parse_mcp_sdk_result(mock_result)
    assert parsed["success"] is False
    assert "exceeds $50 limit" in parsed["error"]


# =====================================================================
# 4. Successful Submission Normalization
# =====================================================================

def test_parse_success_from_structured_content(mcp_client: MCPOrderClient) -> None:
    success_payload = {
        "success": True,
        "order_id": "ORD-12345",
        "total": 13.00,
        "estimated_time": "15-20 minutes"
    }
    mock_result = MockCallToolResult(
        isError=False,
        structuredContent=success_payload,
        content=[MockTextContent(text=json.dumps(success_payload))]
    )

    parsed = mcp_client._parse_mcp_sdk_result(mock_result)
    assert parsed["success"] is True
    assert parsed["order_id"] == "ORD-12345"
    assert parsed["total"] == 13.00
    assert parsed["estimated_time"] == "15-20 minutes"


def test_parse_success_text_fallback_when_structured_none(mcp_client: MCPOrderClient) -> None:
    success_payload = {
        "success": True,
        "order_id": "ORD-54321",
        "total": 9.50,
        "estimated_time": "10-15 minutes"
    }
    mock_result = MockCallToolResult(
        isError=False,
        structuredContent=None,
        content=[MockTextContent(text=json.dumps(success_payload))]
    )

    parsed = mcp_client._parse_mcp_sdk_result(mock_result)
    assert parsed["success"] is True
    assert parsed["order_id"] == "ORD-54321"


# =====================================================================
# 5. Synchronous submit_order Dispatch
# =====================================================================

def test_submit_order_synchronous_wrapper(mcp_client: MCPOrderClient) -> None:
    mock_return = {"success": True, "order_id": "ORD-99999", "total": 15.00}
    with patch.object(mcp_client, "submit_order_async", new_callable=AsyncMock) as mock_async:
        mock_async.return_value = mock_return
        items = [{"item_id": "classic_burger", "quantity": 1}]
        
        result = mcp_client.submit_order(items=items, special_instructions="no pickles")
        assert result == mock_return
        mock_async.assert_called_once_with(items=items, special_instructions="no pickles", force_failure=None)