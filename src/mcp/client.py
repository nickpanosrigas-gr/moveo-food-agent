from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
from typing import Any

import httpx

from config.settings import settings
logger = logging.getLogger(__name__)

DEFAULT_MCP_URL = "https://webviews-git-moveo-food-nlp-assignment-moveo-ai.vercel.app/api/demo/mcp"


class MCPOrderClient:
    """
    MCP Client for Moveo's Streamable HTTP server (`submit_order` tool).

    Handles:
      - Required `X-Applicant-Email` authentication header on every request.
      - Optional `X-Demo-Force-Failure` header (`kitchen_busy` | `server_error`) for testing.
      - All 3 rejection paths defined in the specification:
          1. Schema validation (`isError: true` with JSON-RPC `-32602`, no `structuredContent`).
          2. Server error (`isError: true` carrying a `{"success": false, ...}` body).
          3. Business refusal / Transient failure (`isError: false` with `{"success": false, "error": "..."}`).
    """

    def __init__(
        self,
        server_url: str | None = None,
        applicant_email: str | None = None,
        timeout_seconds: float = 30.0,
    ):
        self.server_url = (server_url or settings.MCP_SERVER_URL).strip()
        self.applicant_email = (applicant_email or settings.APPLICANT_EMAIL).strip()

        if not self.applicant_email or self.applicant_email == "your_email_from_cv@example.com":
            logger.warning(
                "[MCPOrderClient] APPLICANT_EMAIL in .env is missing or still set to the placeholder! "
                "Update APPLICANT_EMAIL in .env before submitting an order."
            )
        self.timeout_seconds = timeout_seconds
        
        self.server_url = (
            server_url
            or os.getenv("MCP_SERVER_URL")
            or DEFAULT_MCP_URL
        ).strip()

        self.applicant_email = (
            applicant_email
            or os.getenv("APPLICANT_EMAIL")
            or os.getenv("X_APPLICANT_EMAIL")
            or ""
        ).strip()

        if not self.applicant_email:
            logger.warning(
                "[MCPOrderClient] APPLICANT_EMAIL is not set! "
                "The Moveo MCP server requires the 'X-Applicant-Email' header on all requests."
            )

        self.timeout_seconds = timeout_seconds

    def _build_headers(self, force_failure: str | None = None) -> dict[str, str]:
        """Constructs HTTP headers for the MCP Streamable HTTP transport."""
        headers: dict[str, str] = {
            "X-Applicant-Email": self.applicant_email,
        }
        if force_failure in ("kitchen_busy", "server_error"):
            headers["X-Demo-Force-Failure"] = force_failure
            logger.info(f"[MCPOrderClient] Injecting header X-Demo-Force-Failure: {force_failure}")
        return headers

    def _prepare_mcp_arguments(
        self,
        items: list[dict[str, Any]],
        special_instructions: str | None = None,
        force_failure: str | None = None,
    ) -> dict[str, Any]:
        """
        Prepares the `submit_order` arguments and optionally injects test mutations
        for `schema_error` (-32602) or `business_refusal` (>$50 limit) when requested by the UI.
        """
        import copy
        cloned_items = copy.deepcopy(items)

        if force_failure == "schema_error":
            # Force a genuine JSON-RPC -32602 schema validation error on the remote MCP server
            if cloned_items:
                cloned_items[0]["quantity"] = 0  # Violates `"minimum": 1` in MCP inputSchema
            else:
                cloned_items = [{"item_id": "classic_burger", "quantity": 0}]
            logger.info("[MCPOrderClient] Forcing 'schema_error' by sending quantity=0 to MCP server.")

        elif force_failure == "business_refusal":
            # Force an order total > $50.00 to trigger a genuine MCP business refusal
            if cloned_items:
                cloned_items[0]["quantity"] = max(int(cloned_items[0].get("quantity", 1)), 10)
            else:
                cloned_items = [{"item_id": "margherita", "quantity": 5, "options": {"size": "large"}, "extras": []}]
            logger.info("[MCPOrderClient] Forcing 'business_refusal' by inflating item quantity to exceed $50.")

        arguments: dict[str, Any] = {"items": cloned_items}
        if special_instructions is not None:
            arguments["special_instructions"] = special_instructions
        return arguments
    
    # -----------------------------------------------------------------
    # Public Synchronous Entry Point (Used by FoodOrderAgent.send)
    # -----------------------------------------------------------------
    def submit_order(
        self,
        items: list[dict[str, Any]],
        special_instructions: str | None = None,
        force_failure: str | None = None,
    ) -> dict[str, Any]:
        """
        Synchronously invokes the remote `submit_order` MCP tool.
        Safe to call from both standard synchronous scripts (`main.py`, `pytest`)
        and inside an active FastAPI `asyncio` event loop.
        """
        coro = self.submit_order_async(
            items=items,
            special_instructions=special_instructions,
            force_failure=force_failure,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=self.timeout_seconds + 5.0)
        else:
            return asyncio.run(coro)

    # -----------------------------------------------------------------
    # Asynchronous Implementation (Official MCP SDK + HTTP Fallback)
    # -----------------------------------------------------------------
    async def submit_order_async(
        self,
        items: list[dict[str, Any]],
        special_instructions: str | None = None,
        force_failure: str | None = None,
    ) -> dict[str, Any]:
        """
        Connects to the MCP server via Streamable HTTP transport, initializes a session,
        calls `submit_order`, and normalizes all success and error paths into a clean dict.
        """
        arguments = self._prepare_mcp_arguments(
            items=items,
            special_instructions=special_instructions,
            force_failure=force_failure,
        )
        headers = self._build_headers(force_failure=force_failure)

        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(
                url=self.server_url,
                headers=headers,
                timeout=self.timeout_seconds,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    call_result = await session.call_tool(
                        name="submit_order",
                        arguments=arguments,
                    )
                    return self._parse_mcp_sdk_result(call_result)

        except ImportError:
            logger.info("[MCPOrderClient] 'mcp' package not found; using direct Streamable HTTP JSON-RPC client.")
            return await self._call_via_raw_streamable_http(arguments=arguments, headers=headers)

        except Exception as exc:
            logger.error(f"[MCPOrderClient] MCP transport/protocol error: {exc}", exc_info=True)
            return self._normalize_exception(exc)

    # -----------------------------------------------------------------
    # Result Normalization (Handles all 3 Spec Rejection Modes)
    # -----------------------------------------------------------------
    def _parse_mcp_sdk_result(self, call_result: Any) -> dict[str, Any]:
        """
        Parses `CallToolResult` from the MCP SDK according to the Moveo assignment rules:
          1. Always inspect `isError` FIRST before reading `structuredContent`.
          2. On `isError: True` (schema error -32602 or `server_error`), extract the text block,
             parse JSON if present, and return `{"success": False, "error": ...}`.
          3. On `isError: False`, prefer `structuredContent` if available, falling back to
             parsing the first text block in `content`.
        """
        is_error = bool(getattr(call_result, "isError", False))
        raw_text = self._extract_text_from_content(getattr(call_result, "content", None))

        # Path 1: Tool result flagged with `isError: true` (Schema Validation -32602 or Server Error)
        if is_error:
            logger.warning(f"[MCPOrderClient] Tool returned isError=True: {raw_text}")
            parsed_error_body = self._try_parse_json(raw_text)

            if isinstance(parsed_error_body, dict):
                # e.g., `server_error` returns isError=True with a `{"success": false, "error": "..."}` body
                if "success" in parsed_error_body and "error" in parsed_error_body:
                    return parsed_error_body
                # e.g., JSON-RPC -32602 error payload embedded in text
                if "error" in parsed_error_body and isinstance(parsed_error_body["error"], dict):
                    rpc_err = parsed_error_body["error"]
                    code = rpc_err.get("code", -32602)
                    msg = rpc_err.get("message", raw_text)
                    return {"success": False, "error": f"Schema validation error ({code}): {msg}"}
                if "message" in parsed_error_body:
                    return {"success": False, "error": str(parsed_error_body["message"])}

            return {
                "success": False,
                "error": raw_text or "Order submission failed due to a server or schema validation error.",
            }

        # Path 2: Normal tool result (`isError: false`) -> Read `structuredContent` first
        structured = getattr(call_result, "structuredContent", None)
        if isinstance(structured, dict) and structured:
            return dict(structured)

        # Fallback to parsing the text block if `structuredContent` was not populated
        parsed_body = self._try_parse_json(raw_text)
        if isinstance(parsed_body, dict):
            return parsed_body

        return {
            "success": False,
            "error": raw_text or "Received an empty or unparseable response from the MCP server.",
        }

    # -----------------------------------------------------------------
    # Direct Streamable HTTP JSON-RPC Fallback (Zero-Dependency Backup)
    # -----------------------------------------------------------------
    async def _call_via_raw_streamable_http(
        self,
        arguments: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """
        Direct MCP Streamable HTTP transport implementation using `httpx`.
        Handles session initialization (`Mcp-Session-Id`) and both `application/json`
        and `text/event-stream` (SSE) responses.
        """
        req_headers = {
            **headers,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            # 1. Initialize MCP session
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "moveo-food-order-agent", "version": "1.0.0"},
                },
            }
            init_resp = await client.post(self.server_url, headers=req_headers, json=init_payload)
            init_resp.raise_for_status()

            session_id = init_resp.headers.get("Mcp-Session-Id") or init_resp.headers.get("mcp-session-id")
            if session_id:
                req_headers["Mcp-Session-Id"] = session_id

            # 2. Send initialized notification
            await client.post(
                self.server_url,
                headers=req_headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )

            # 3. Invoke `tools/call` for `submit_order`
            call_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "submit_order",
                    "arguments": arguments,
                },
            }
            call_resp = await client.post(self.server_url, headers=req_headers, json=call_payload)
            call_resp.raise_for_status()

            rpc_data = self._parse_http_or_sse_json(call_resp)
            if "error" in rpc_data:
                err_obj = rpc_data["error"]
                msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
                return {"success": False, "error": msg}

            result_obj = rpc_data.get("result", {})

            # Construct a lightweight proxy matching CallToolResult attributes
            class _RawCallResult:
                isError = result_obj.get("isError", False)
                structuredContent = result_obj.get("structuredContent")
                content = result_obj.get("content", [])

            return self._parse_mcp_sdk_result(_RawCallResult())

    # -----------------------------------------------------------------
    # Parsing Utilities
    # -----------------------------------------------------------------
    @staticmethod
    def _extract_text_from_content(content_list: Any) -> str:
        """Extracts concatenated text from MCP `content` blocks (supports objects and dicts)."""
        if not content_list or not isinstance(content_list, list):
            return ""
        chunks: list[str] = []
        for block in content_list:
            if isinstance(block, dict) and "text" in block:
                chunks.append(str(block["text"]))
            elif hasattr(block, "text") and block.text is not None:
                chunks.append(str(block.text))
        return "\n".join(chunks).strip()

    @staticmethod
    def _try_parse_json(text: str) -> Any:
        if not text:
            return None
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_http_or_sse_json(response: httpx.Response) -> dict[str, Any]:
        """Parses either standard `application/json` or `text/event-stream` SSE payloads."""
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" in content_type:
            for line in reversed(response.text.splitlines()):
                line = line.strip()
                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    if data_str:
                        try:
                            return json.loads(data_str)
                        except ValueError:
                            continue
        return response.json()

    @staticmethod
    def _normalize_exception(exc: Exception) -> dict[str, Any]:
        """Converts transport/JSON-RPC exceptions into a clean tool result dictionary."""
        msg = str(exc)
        if "-32602" in msg:
            return {"success": False, "error": f"Invalid order schema (-32602): {msg}"}
        return {
            "success": False,
            "error": f"Unable to reach the kitchen server right now ({exc.__class__.__name__}). Please try again.",
        }