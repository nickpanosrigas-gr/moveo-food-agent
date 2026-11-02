# src/core/agent_loop.py
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from config.settings import settings
from src.core.cart import OrderCart
from src.core.menu import MenuCatalog
from src.core.ollama_client import OllamaClient
from src.core.prompts import (
    build_ollama_tools,
    build_static_system_prompt,
    format_user_turn_with_context,
)
from src.mcp.client import MCPOrderClient
from src.utils.logger import TurnLogger

logger = logging.getLogger(__name__)

CHECKOUT_INTENT_REGEX = re.compile(
    r"\b(that'?s\s+it|that\s+is\s+it|nothing\s+else|no\s+more|i'?m\s+done|all\s+set|submit|checkout|place\s+order)\b",
    re.IGNORECASE,
)


class FoodOrderAgent:
    """
    Conversational Food Ordering Agent powered by local Ollama (`gemma4:12b`, 32k context).
    Keeps message[0] (`role: system`) and `tools` 100% static across turns so Ollama's
    KV cache achieves maximum prefix reuse and minimal Time-To-First-Token (TTFT).
    """

    def __init__(
        self,
        menu_path: str | Path = settings.MENU_PATH,
        ollama_url: str | None = None,
        ollama_model: str | None = None,
        ollama_ctx: int | None = None,
        mcp_url: str | None = None,
        applicant_email: str | None = None,
        log_path: str | Path = settings.DEFAULT_LOG_PATH,
        max_steps: int = 6,
        warmup_on_init: bool = False,
    ):
        self.menu = MenuCatalog.from_yaml(menu_path)
        self.cart = OrderCart(self.menu)
        self.llm = OllamaClient(
            base_url=ollama_url,
            model=ollama_model,
            num_ctx=ollama_ctx,
        )
        self.mcp_client = MCPOrderClient(
            server_url=mcp_url or settings.MCP_SERVER_URL,
            applicant_email=applicant_email or settings.APPLICANT_EMAIL,
        )
        self.turn_logger = TurnLogger(log_path=log_path)

        self.max_steps = max_steps
        self.static_system_prompt = build_static_system_prompt(self.menu)
        self.tools = build_ollama_tools(self.menu)

        # Message 0 is ALWAYS the static system prompt so Ollama's prefix KV cache never invalidates
        self.history: list[dict[str, Any]] = [
            {"role": "system", "content": self.static_system_prompt}
        ]

        self.awaiting_confirmation: bool = False
        self.can_retry_submission: bool = False
        self.force_failure_mode: str | None = None
        self.last_turn_log: dict[str, Any] | None = None
        self.last_mcp_payload: dict[str, Any] | None = None

        if warmup_on_init:
            try:
                self.llm.warmup_kv_cache_sync(self.static_system_prompt, self.tools)
            except Exception as exc:
                logger.warning(f"[FoodOrderAgent] Initial KV warmup warning: {exc}")

    # -----------------------------------------------------------------
    # KV Cache Warmup Helpers (Used by CLI & FastAPI Session Startup)
    # -----------------------------------------------------------------
    def warmup_sync(self) -> dict[str, Any]:
        return self.llm.warmup_kv_cache_sync(self.static_system_prompt, self.tools)

    async def warmup_async(self) -> dict[str, Any]:
        return await self.llm.warmup_kv_cache(self.static_system_prompt, self.tools)
    
    def unload_sync(self) -> dict[str, Any]:
        return self.llm.unload_model_sync()

    # -----------------------------------------------------------------
    # Required Public Entry Point: send(message: str) -> dict
    # -----------------------------------------------------------------
    def send(self, message: str, force_failure: str | None = None) -> dict[str, Any]:
        """
        Processes a user message through the Ollama agentic loop and returns:
          {
            "message": str,
            "tool_calls": [ ... ]  # Present ONLY when `submit_order` was invoked this turn
          }
        """
        if force_failure and force_failure != "none":
            self.force_failure_mode = force_failure

        self.last_mcp_payload = None
        user_text = (message or "").strip()
        self.cart.start_turn()
        was_ready_to_submit = self.awaiting_confirmation or self.can_retry_submission

        # Inject current cart context into the user message so `self.history[0]` (system) stays 100% static
        contextualized_user_msg = format_user_turn_with_context(
            user_message=user_text,
            cart_formatted=self.cart.format_for_prompt(),
            awaiting_confirmation=was_ready_to_submit and not self.cart.modified_in_current_turn,
        )
        self.history.append({"role": "user", "content": contextualized_user_msg})

        exposed_tool_calls: list[dict[str, Any]] = []
        internal_tool_trace: list[dict[str, Any]] = []
        final_agent_text: str = ""
        submitted_in_this_turn: bool = False

        for step in range(self.max_steps):
            assistant_msg = self.llm.chat(
                messages=self.history,
                tools=self.tools,
                temperature=0.1,
            )
            self.history.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls") or []

            # Base case: No tool calls requested -> final conversational response reached
            if not tool_calls:
                final_agent_text = (assistant_msg.get("content") or "").strip()
                break

            # Execute requested tool calls
            for tc in tool_calls:
                fn_block = tc.get("function", {})
                fn_name = fn_block.get("name", "")
                fn_args = fn_block.get("arguments") or {}

                # Detailed debug log for EVERY tool invocation
                logger.info(f"[TOOL CALL] {fn_name} | Args: {json.dumps(fn_args, ensure_ascii=False)}")

                # 1. Local Cart Tools (Kept internal; never added to exposed_tool_calls)
                if fn_name in ("add_item", "edit_item", "remove_item"):
                    tool_result = self.cart.execute_tool(fn_name, fn_args)
                    logger.info(
                        f"[TOOL RESULT] {fn_name} -> success={tool_result.get('success')} | "
                        f"Action: {tool_result.get('action', 'none')} | "
                        f"Cart Total: ${self.cart.total_price:.2f}"
                    )

                    if self.cart.modified_in_current_turn:
                        self.awaiting_confirmation = False
                        self.can_retry_submission = False
                        was_ready_to_submit = False

                    internal_tool_trace.append(
                        {"name": fn_name, "arguments": fn_args, "result": tool_result}
                    )
                    self.history.append(
                        {
                            "role": "tool",
                            "name": fn_name,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )

                elif fn_name == "set_special_instructions":
                    notes = fn_args.get("special_instructions")
                    self.cart.update_special_instructions(notes)
                    tool_result = {
                        "success": True,
                        "special_instructions": self.cart.special_instructions,
                        "cart": self.cart.get_state_snapshot(),
                    }
                    logger.info(f"[TOOL RESULT] set_special_instructions -> \"{self.cart.special_instructions}\"")
                    internal_tool_trace.append(
                        {"name": fn_name, "arguments": fn_args, "result": tool_result}
                    )
                    self.history.append(
                        {
                            "role": "tool",
                            "name": fn_name,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )

                # 2. Remote MCP Tool (`submit_order`)
                elif fn_name == "submit_order":
                    if submitted_in_this_turn or self.cart.is_submitted:
                        logger.warning("[MCP CALL] submit_order skipped (already submitted in this turn)")
                        dup_res = {
                            "status": "ALREADY_SUBMITTED",
                            "instruction": "The order was already submitted in this turn. State the order_id and estimated_time to the user.",
                            "last_result": exposed_tool_calls[-1]["result"] if exposed_tool_calls else {},
                        }
                        self.history.append(
                            {
                                "role": "tool",
                                "name": fn_name,
                                "content": json.dumps(dup_res, ensure_ascii=False),
                            }
                        )
                        continue

                    if not self.cart.lines:
                        logger.warning("[MCP CALL] submit_order rejected (empty cart)")
                        guard_res = {
                            "success": False,
                            "error": "Cannot submit an empty cart. Please ask the user what they would like to order.",
                        }
                        self.history.append(
                            {
                                "role": "tool",
                                "name": fn_name,
                                "content": json.dumps(guard_res, ensure_ascii=False),
                            }
                        )
                        continue

                    if self.cart.modified_in_current_turn or not was_ready_to_submit:
                        logger.info(
                            f"[MCP GUARD] Blocked premature submission. Prompting confirmation for ${self.cart.total_price:.2f}"
                        )
                        self.awaiting_confirmation = True
                        guard_res = {
                            "status": "CONFIRMATION_REQUIRED_FIRST",
                            "instruction": (
                                f"Do NOT submit the order yet. Summarize the current order items and exact "
                                f"total price (${self.cart.total_price:.2f}) and ask the user: 'Ready to submit?'"
                            ),
                            "cart": self.cart.get_state_snapshot(),
                        }
                        internal_tool_trace.append(
                            {"name": "submit_order_blocked_for_confirmation", "arguments": fn_args, "result": guard_res}
                        )
                        self.history.append(
                            {
                                "role": "tool",
                                "name": fn_name,
                                "content": json.dumps(guard_res, ensure_ascii=False),
                            }
                        )
                        continue

                    if fn_args.get("special_instructions"):
                        self.cart.update_special_instructions(fn_args.get("special_instructions"))

                    active_failure = self.force_failure_mode
                    self.force_failure_mode = None

                    mcp_arguments = self.mcp_client._prepare_mcp_arguments(
                        items=self.cart.to_mcp_items(),
                        special_instructions=self.cart.special_instructions,
                        force_failure=active_failure,
                    )

                    logger.info(
                        f"[MCP CALL] Submitting order to server | Forced failure: {active_failure} | "
                        f"Items count: {len(mcp_arguments['items'])}"
                    )

                    mcp_result = self.mcp_client.submit_order(
                        items=self.cart.to_mcp_items(),
                        special_instructions=self.cart.special_instructions,
                        force_failure=active_failure,
                    )
                    submitted_in_this_turn = True

                    logger.info(f"[MCP RESULT] submit_order -> {json.dumps(mcp_result, ensure_ascii=False)}")

                    self.last_mcp_payload = {
                        "name": "submit_order",
                        "arguments": mcp_arguments,
                        "result": mcp_result,
                        "forced_failure_mode": active_failure,
                    }

                    exposed_tool_calls.append(
                        {
                            "name": "submit_order",
                            "arguments": mcp_arguments,
                            "result": mcp_result,
                        }
                    )

                    if mcp_result.get("success") is True:
                        self.cart.mark_submitted(order_id=mcp_result.get("order_id"))
                        self.awaiting_confirmation = False
                        self.can_retry_submission = False
                    else:
                        self.can_retry_submission = True

                    self.history.append(
                        {
                            "role": "tool",
                            "name": fn_name,
                            "content": json.dumps(mcp_result, ensure_ascii=False),
                        }
                    )

                else:
                    logger.warning(f"[TOOL CALL] Unknown tool requested: '{fn_name}'")
                    unknown_res = {"success": False, "error": f"Unknown tool '{fn_name}'."}
                    self.history.append(
                        {
                            "role": "tool",
                            "name": fn_name,
                            "content": json.dumps(unknown_res, ensure_ascii=False),
                        }
                    )

        if not final_agent_text:
            final_agent_text = self._build_fallback_text(exposed_tool_calls)

        self._update_confirmation_state(
            user_text=user_text,
            agent_text=final_agent_text,
            exposed_tool_calls=exposed_tool_calls,
        )

        self.last_turn_log = self.turn_logger.log_turn(
            user_message=user_text,
            agent_message=final_agent_text,
            exposed_tool_calls=exposed_tool_calls,
            internal_tool_calls=internal_tool_trace,
            cart_snapshot=self.cart.get_state_snapshot(),
            extra_metadata={"model": self.llm.model, "num_ctx": self.llm.num_ctx},
        )

        result_payload: dict[str, Any] = {"message": final_agent_text}
        if exposed_tool_calls:
            result_payload["tool_calls"] = exposed_tool_calls

        return result_payload

    def _update_confirmation_state(
        self,
        user_text: str,
        agent_text: str,
        exposed_tool_calls: list[dict[str, Any]],
    ) -> None:
        if not self.cart.lines or self.cart.is_submitted:
            self.awaiting_confirmation = False
            return

        if exposed_tool_calls:
            return

        lower_agent = agent_text.lower()
        asked_to_submit = any(
            phrase in lower_agent
            for phrase in (
                "ready to submit",
                "would you like to submit",
                "shall i submit",
                "confirm your order",
                "ready to place",
            )
        )

        if asked_to_submit:
            self.awaiting_confirmation = True
        elif not self.cart.modified_in_current_turn and CHECKOUT_INTENT_REGEX.search(user_text):
            self.awaiting_confirmation = True

    def _build_fallback_text(self, exposed_tool_calls: list[dict[str, Any]]) -> str:
        if exposed_tool_calls:
            last_res = exposed_tool_calls[-1]["result"]
            if last_res.get("success"):
                oid = last_res.get("order_id", "UNKNOWN")
                eta = last_res.get("estimated_time", "15-20 minutes")
                return f"Order submitted! Order #{oid}. Estimated time: {eta}."
            err = last_res.get("error", "An unexpected error occurred.")
            return f"Sorry, the order couldn't be submitted — {err} Would you like me to try again?"

        if self.cart.lines:
            return f"Your current total is ${self.cart.total_price:.2f}. Anything else?"
        return "What can I get started for you today?"