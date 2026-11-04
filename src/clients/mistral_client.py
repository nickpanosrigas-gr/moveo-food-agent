# src/clients/mistral_client.py
from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any

import httpx
from config.settings import settings
from src.clients.errors import request_with_retry_sync

THINKING_REGEX = re.compile(r"<think>[\s\S]*?</think>|(?:\A|^)\s*</think>|<\/?think>", re.IGNORECASE)

logger = logging.getLogger(__name__)


class MistralClient:
    """
    Synchronous HTTP client for Mistral Chat Completions API (e.g. deepseek-v4-1-flash-1017).
    Includes industry-standard error mapping (400, 401, 404, 422, 5xx) and exponential backoff for 429s.
    """

    manages_vram: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ):
        self.api_key = (api_key or settings.MISTRAL_API_KEY).strip()
        self.base_url = (base_url or settings.MISTRAL_API_URL).rstrip("/")
        self.model = model or settings.MISTRAL_LLM
        self.num_ctx = None
        self.max_retries = max_retries
        self.timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        self._sync_client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )

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
        **_kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages_for_mistral(messages),
            "stream": False,
            "temperature": temperature,
            "max_tokens": 512,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.info(f"[Mistral API] Calling {self.model} via /v1/chat/completions...")
        response = request_with_retry_sync(
            request_fn=lambda: self._sync_client.post("/v1/chat/completions", json=payload),
            provider="Mistral LLM",
            context_hint=f"model='{self.model}', endpoint='/v1/chat/completions'",
            max_retries=self.max_retries,
        )
        data = response.json()

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return self._normalize_message(msg)

    def warmup_kv_cache_sync(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "noop_cloud_provider", "provider": "mistral", "model": self.model}

    async def warmup_kv_cache(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "noop_cloud_provider", "provider": "mistral", "model": self.model}

    def unload_model_sync(self) -> dict[str, Any]:
        self.close()
        return {"status": "noop_cloud_provider", "provider": "mistral", "model": self.model}

    async def unload_model(self) -> dict[str, Any]:
        return {"status": "noop_cloud_provider", "provider": "mistral", "model": self.model}

    @staticmethod
    def _format_messages_for_mistral(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            msg_copy = copy.deepcopy(msg)
            if msg_copy.get("role") == "assistant" and msg_copy.get("tool_calls"):
                for tc in msg_copy["tool_calls"]:
                    fn = tc.get("function", {})
                    args = fn.get("arguments")
                    if isinstance(args, dict):
                        fn["arguments"] = json.dumps(args, ensure_ascii=False)
            formatted.append(msg_copy)
        return formatted

    @staticmethod
    def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
        raw_tool_calls = msg.get("tool_calls") or []
        normalized_calls: list[dict[str, Any]] = []

        for idx, tc in enumerate(raw_tool_calls):
            call_id = tc.get("id") or f"call{idx:05d}"
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

        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        elif content is None:
            content = msg.get("reasoning_content") or ""

        cleaned_content = THINKING_REGEX.sub("", str(content)).strip()

        out: dict[str, Any] = {
            "role": "assistant",
            "content": cleaned_content,
        }
        if normalized_calls:
            out["tool_calls"] = normalized_calls
        return out