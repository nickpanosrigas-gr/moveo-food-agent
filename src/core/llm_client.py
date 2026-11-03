# src/core/llm_client.py
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from config.settings import settings

import re

THINKING_REGEX = re.compile(r"<think>[\s\S]*?</think>|(?:\A|^)\s*</think>|<\/?think>", re.IGNORECASE)

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
        default_keep_alive: str = "15m",
        timeout_seconds: float = 60.0,
    ):
        self.base_url = (base_url or settings.LLM_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.num_ctx = num_ctx or settings.LLM_CTX
        self.default_keep_alive = default_keep_alive
        self.timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self._sync_client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        try:
            self._sync_client.close()
        except Exception:
            pass

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        keep_alive: str | int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": 512,
            "id_slot": 0,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.info(f"[llama-server] Calling {self.model} via /v1/chat/completions...")
        response = self._sync_client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return self._normalize_message(msg)

    def warmup_kv_cache_sync(
        self,
        static_system_prompt: str,
        tools: list[dict[str, Any]],
        keep_alive: str | int = "15m",
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": static_system_prompt},
                {"role": "user", "content": "[LIVE CART CONTEXT]\nCURRENT CART STATE: Empty (0 items, Total: €0.00).\nAWAITING_FINAL_CONFIRMATION: False\n[/LIVE CART CONTEXT]\n\nUser: hi"},
            ],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 1,
            "temperature": 0.1,
            "id_slot": 0,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        logger.info(f"[llama-server] Warming up KV cache for static system prompt...")
        resp = self._sync_client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        logger.info(f"[llama-server] KV cache ready ({prompt_tokens} prompt tokens cached).")
        return {
            "status": "loaded_and_kv_cached",
            "model": self.model,
            "cached_tokens": prompt_tokens,
        }

    async def warmup_kv_cache(
        self,
        static_system_prompt: str,
        tools: list[dict[str, Any]],
        keep_alive: str | int = "15m",
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": static_system_prompt},
                {"role": "user", "content": "[LIVE CART CONTEXT]\nCURRENT CART STATE: Empty (0 items, Total: €0.00).\nAWAITING_FINAL_CONFIRMATION: False\n[/LIVE CART CONTEXT]\n\nUser: hi"},
            ],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 1,
            "temperature": 0.1,
            "id_slot": 0,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        logger.info(f"[llama-server Async] Warming up KV cache...")
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            resp = await client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
            logger.info(f"[llama-server Async] KV cache ready ({prompt_tokens} prompt tokens cached).")
            return {
                "status": "loaded_and_kv_cached",
                "model": self.model,
                "cached_tokens": prompt_tokens,
            }

    def unload_model_sync(self) -> dict[str, Any]:
        logger.info(f"[llama-server] Offloading model from GPU VRAM via /models/unload...")
        try:
            resp = self._sync_client.post("/models/unload", json={"model": self.model})
            if resp.status_code in (200, 204):
                logger.info(f"[llama-server] Model successfully offloaded from VRAM.")
                return {"status": "unloaded", "model": self.model}
            self._sync_client.post("/slots/0?action=erase")
            return {"status": "slot_cleared", "model": self.model}
        except Exception as exc:
            logger.warning(f"[llama-server] Model unload warning: {exc}")
            return {"status": "error", "error": str(exc)}
        finally:
            self.close()

    async def unload_model(self) -> dict[str, Any]:
        logger.info(f"[llama-server Async] Offloading model from GPU VRAM...")
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            try:
                resp = await client.post("/models/unload", json={"model": self.model})
                if resp.status_code in (200, 204):
                    logger.info(f"[llama-server Async] Model offloaded successfully.")
                    return {"status": "unloaded", "model": self.model}
                await client.post("/slots/0?action=erase")
                return {"status": "slot_cleared", "model": self.model}
            except Exception as exc:
                return {"status": "error", "error": str(exc)}

    @staticmethod
    def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
        raw_tool_calls = msg.get("tool_calls") or []
        normalized_calls: list[dict[str, Any]] = []

        for tc in raw_tool_calls:
            call_id = tc.get("id") or "call_0"
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", {})

            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except ValueError:
                    parsed_args = {}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}

            normalized_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": parsed_args,
                    },
                }
            )

        # Fallback to reasoning_content if content is None, but clean thinking tags
        content = msg.get("content")
        if content is None:
            content = msg.get("reasoning_content") or ""

        # Clean any <think>...</think>, </think>, or orphaned tags
        cleaned_content = THINKING_REGEX.sub("", str(content)).strip()

        out: dict[str, Any] = {
            "role": "assistant",
            "content": cleaned_content,
        }
        if normalized_calls:
            out["tool_calls"] = normalized_calls
        return out