# src/core/agent_loop.py
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from config.settings import settings
from src.clients import LLMClientProtocol, create_llm_client, llm_uses_vram
from src.core.cart import OrderCart
from src.core.menu import MenuCatalog
from src.core.prompts import (
    build_llm_tools,
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

THINKING_STRIP_REGEX = re.compile(
    r"<think>[\s\S]*?</think>|(?:\A|^)\s*</think>|<\/?think>",
    re.IGNORECASE,
)


def _sanitize_agent_text(text: str) -> str:
    """Removes any <think>...</think> blocks, stray </think> tags, and extra whitespace."""
    if not text:
        return ""
    cleaned = THINKING_STRIP_REGEX.sub("", text)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned).strip()
    return cleaned


class FoodOrderAgent:
    def __init__(
        self,
        menu_path: str | Path = settings.MENU_PATH,
        llm_provider: str | None = None,
        llm_url: str | None = None,
        llm_model: str | None = None,
        llm_ctx: int | None = None,
        llm_api_key: str | None = None,
        mcp_url: str | None = None,
        applicant_email: str | None = None,
        log_path: str | Path = settings.DEFAULT_LOG_PATH,
        max_steps: int = 6,
        warmup_on_init: bool = False,
    ):
        self.menu = MenuCatalog.from_yaml(menu_path)
        self.cart = OrderCart(self.menu)
        self.llm: LLMClientProtocol = create_llm_client(
            provider=llm_provider,
            base_url=llm_url,
            model=llm_model,
            num_ctx=llm_ctx,
            api_key=llm_api_key,
        )
        self.mcp_client = MCPOrderClient(
            server_url=mcp_url or settings.MCP_SERVER_URL,
            applicant_email=applicant_email or settings.APPLICANT_EMAIL,
        )
        self.turn_logger = TurnLogger(log_path=log_path)

        self.max_steps = max_steps
        self.static_system_prompt = build_static_system_prompt(self.menu)
        self.tools = build_llm_tools(self.menu)

        self.history: list[dict[str, Any]] = [
            {"role": "system", "content": self.static_system_prompt}
        ]

        self.awaiting_confirmation: bool = False
        self.can_retry_submission: bool = False
        self.force_failure_mode: str | None = None
        self.last_turn_log: dict[str, Any] | None = None
        self.last_mcp_payload: dict[str, Any] | None = None

        if warmup_on_init and getattr(self.llm, "manages_vram", False):
            try:
                self.llm.warmup_kv_cache_sync(self.static_system_prompt, self.tools)
            except Exception as exc:
                logger.warning(f"[FoodOrderAgent] Initial KV warmup warning: {exc}")

    def warmup_sync(self) -> dict[str, Any]:
        return self.llm.warmup_kv_cache_sync(self.static_system_prompt, self.tools)

    async def warmup_async(self) -> dict[str, Any]:
        return await self.llm.warmup_kv_cache(self.static_system_prompt, self.tools)

    def unload_sync(self) -> dict[str, Any]:
        return self.llm.unload_model_sync()

    def send(self, message: str, force_failure: str | None = None) -> dict[str, Any]:
        user_text = (message or "").strip()
        
        # Defense-in-depth: intercept empty/non-alphanumeric messages
        if not re.sub(r"[^\w]", "", user_text):
            fallback_text = "Sorry, I couldn't hear you."
            return {
                "message": fallback_text,
                "tool_calls": [],
            }

        if force_failure and force_failure != "none":
            self.force_failure_mode = force_failure

        self.last_mcp_payload = None
        user_text = (message or "").strip()
        self.cart.start_turn()
        was_ready_to_submit = self.awaiting_confirmation or self.can_retry_submission

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

        last_failed_call: tuple[str, str] | None = None

        for step in range(self.max_steps):
            assistant_msg = self.llm.chat(
                messages=self.history,
                tools=self.tools,
                temperature=0.1,
            )

            tool_calls = assistant_msg.get("tool_calls") or []
            content = (assistant_msg.get("content") or "").strip()

            if not tool_calls and not content:
                logger.warning(f"[AGENT] Received completely empty response on step {step}. Breaking loop.")
                break

            self.history.append(assistant_msg)

            if not tool_calls:
                final_agent_text = content
                break

            for idx, tc in enumerate(tool_calls):
                call_id = tc.get("id") or f"call_{step}_{idx}"
                fn_block = tc.get("function", {})
                fn_name = fn_block.get("name", "")
                fn_args = fn_block.get("arguments") or {}

                args_canonical = json.dumps(fn_args, sort_keys=True, ensure_ascii=False)
                call_signature = (fn_name, args_canonical)

                logger.info(f"[TOOL CALL] {fn_name} ({call_id}) | Args: {args_canonical}")

                # INTERCEPT DUPLICATE FAILED TOOL CALLS
                if last_failed_call is not None and call_signature == last_failed_call:
                    logger.warning(
                        f"[AGENT] Blocked duplicate failed tool call on step {step}: {fn_name} with {args_canonical}"
                    )
                    duplicate_error_res = {
                        "success": False,
                        "error": (
                            f"DUPLICATE_CALL_BLOCKED: You just called '{fn_name}' with the exact same arguments and it failed. "
                            f"Do NOT repeat this identical call. "
                            f"Read the previous error message, remove the invalid parameter(s) from your arguments, "
                            f"or answer the user in conversational text."
                        ),
                        "cart": self.cart.get_state_snapshot(),
                    }
                    internal_tool_trace.append(
                        {"name": fn_name, "arguments": fn_args, "result": duplicate_error_res}
                    )
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": fn_name,
                            "content": json.dumps(duplicate_error_res, ensure_ascii=False),
                        }
                    )
                    continue

                # 1. Local Cart Tools (Including clear_cart)
                if fn_name in ("add_item", "edit_item", "remove_item", "clear_cart"):
                    tool_result = self.cart.execute_tool(fn_name, fn_args)
                    is_success = tool_result.get("success", False)

                    logger.info(
                        f"[TOOL RESULT] {fn_name} -> success={is_success} | "
                        f"Action: {tool_result.get('action', 'none')} | "
                        f"Cart Total: ${self.cart.total_price:.2f}"
                    )

                    if not is_success:
                        last_failed_call = call_signature
                    else:
                        last_failed_call = None

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
                            "tool_call_id": call_id,
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
                    last_failed_call = None
                    logger.info(f'[TOOL RESULT] set_special_instructions -> "{self.cart.special_instructions}"')
                    internal_tool_trace.append(
                        {"name": fn_name, "arguments": fn_args, "result": tool_result}
                    )
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
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
                                "tool_call_id": call_id,
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
                                "tool_call_id": call_id,
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
                                "tool_call_id": call_id,
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
                        last_failed_call = None
                    else:
                        self.can_retry_submission = True
                        last_failed_call = call_signature

                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": fn_name,
                            "content": json.dumps(mcp_result, ensure_ascii=False),
                        }
                    )

                else:
                    logger.warning(f"[TOOL CALL] Unknown tool requested: '{fn_name}'")
                    unknown_res = {"success": False, "error": f"Unknown tool '{fn_name}'."}
                    last_failed_call = call_signature
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": fn_name,
                            "content": json.dumps(unknown_res, ensure_ascii=False),
                        }
                    )

        final_agent_text = _sanitize_agent_text(final_agent_text)

        if not final_agent_text:
            final_agent_text = self._build_fallback_text(exposed_tool_calls)

        self._update_confirmation_state(
            user_text=user_text,
            agent_text=final_agent_text,
            exposed_tool_calls=exposed_tool_calls,
        )

        self._prune_history_failures()

        self.last_turn_log = self.turn_logger.log_turn(
            user_message=user_text,
            agent_message=final_agent_text,
            exposed_tool_calls=exposed_tool_calls,
            internal_tool_calls=internal_tool_trace,
            cart_snapshot=self.cart.get_state_snapshot(),
            extra_metadata={
                "llm_provider": settings.LLM_PROVIDER,
                "model": self.llm.model,
                "num_ctx": getattr(self.llm, "num_ctx", None),
            },
        )

        result_payload: dict[str, Any] = {"message": final_agent_text}
        if exposed_tool_calls:
            result_payload["tool_calls"] = exposed_tool_calls

        return result_payload

    def _prune_history_failures(self) -> None:
        """
        Collapses consecutive duplicate failed tool cycles in self.history to a single instance.
        Prevents cross-turn context contamination when the model repeats an invalid call.
        """
        if len(self.history) <= 2:
            return

        def _calls_sig(calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
            return [
                (
                    c.get("function", {}).get("name", ""),
                    json.dumps(c.get("function", {}).get("arguments", {}), sort_keys=True),
                )
                for c in calls
            ]

        pruned: list[dict[str, Any]] = [self.history[0]]
        i = 1

        while i < len(self.history):
            msg = self.history[i]

            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                curr_calls = msg.get("tool_calls", [])
                curr_sig = _calls_sig(curr_calls)

                tool_msgs: list[dict[str, Any]] = []
                j = i + 1
                while j < len(self.history) and self.history[j].get("role") == "tool":
                    tool_msgs.append(self.history[j])
                    j += 1

                all_failed = len(tool_msgs) > 0 and all(
                    json.loads(tm.get("content", "{}")).get("success") is False
                    for tm in tool_msgs
                    if tm.get("content")
                )

                is_duplicate = False
                if all_failed and len(pruned) >= 2 and pruned[-1].get("role") == "tool":
                    prev_asst_idx = None
                    for k in range(len(pruned) - 1, -1, -1):
                        if pruned[k].get("role") == "assistant":
                            prev_asst_idx = k
                            break

                    if prev_asst_idx is not None:
                        prev_sig = _calls_sig(pruned[prev_asst_idx].get("tool_calls", []))
                        if prev_sig == curr_sig:
                            is_duplicate = True

                if is_duplicate:
                    logger.info("[AGENT] Pruned duplicate failed tool cycle from conversation history.")
                    i = j
                    continue

                pruned.append(msg)
                pruned.extend(tool_msgs)
                i = j
            else:
                pruned.append(msg)
                i += 1

        self.history = pruned

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

        return "I didn't quite catch that. Could you repeat what you'd like to order?"