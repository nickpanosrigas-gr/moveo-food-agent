# src/core/ollama_client.py
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    High-speed Ollama client for `gemma4:12b` with persistent HTTP connection pooling,
    disabled thinking overhead (`think: False`), KV-cache warmup, and guaranteed VRAM unload.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        num_ctx: int | None = None,
        default_keep_alive: str = "15m",
        timeout_seconds: float = 60.0,
    ):
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.num_ctx = num_ctx or settings.OLLAMA_CTX
        self.default_keep_alive = default_keep_alive
        self.timeout = httpx.Timeout(timeout_seconds, connect=10.0)

        # Persistent connection pool avoids TCP handshake overhead on every loop step
        self._sync_client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def close(self) -> None:
        """Closes the persistent synchronous HTTP client."""
        try:
            self._sync_client.close()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # 1. Synchronous Chat Completion (Used by FoodOrderAgent.send)
    # -----------------------------------------------------------------
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
            "think": False,  # Disables internal thinking traces for 2-3x faster tool calls
            "keep_alive": keep_alive if keep_alive is not None else self.default_keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": temperature,
                "num_predict": 256,  # Voice/chat responses & tool calls are always short
            },
        }
        if tools:
            payload["tools"] = tools

        logger.info(f"[Ollama] Calling {self.model} (ctx={self.num_ctx})...")
        response = self._sync_client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        msg = data.get("message", {})
        return self._normalize_message(msg)

    # -----------------------------------------------------------------
    # 2. KV-Cache Warmup & VRAM Unload (Sync + Async)
    # -----------------------------------------------------------------
    def warmup_kv_cache_sync(
        self,
        static_system_prompt: str,
        tools: list[dict[str, Any]],
        keep_alive: str | int = "15m",
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": static_system_prompt}],
            "tools": tools,
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": 0.0,
                "num_predict": 1,
            },
        }
        logger.info(
            f"[Ollama] Pre-loading {self.model} (num_ctx={self.num_ctx}) and caching system prompt in VRAM..."
        )
        resp = self._sync_client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        prompt_eval_count = data.get("prompt_eval_count", 0)
        logger.info(f"[Ollama] KV cache ready ({prompt_eval_count} prompt tokens cached).")
        return {
            "status": "loaded_and_kv_cached",
            "model": self.model,
            "num_ctx": self.num_ctx,
            "cached_tokens": prompt_eval_count,
        }

    def unload_model_sync(self) -> dict[str, Any]:
        """
        Synchronously unloads `gemma4:12b` from GPU VRAM immediately (`keep_alive: 0`).
        Called automatically when exiting `main.py` or closing `FoodOrderAgent`.
        """
        logger.info(f"[Ollama] Unloading {self.model} from GPU VRAM (keep_alive=0)...")
        payload = {
            "model": self.model,
            "messages": [],
            "keep_alive": 0,
        }
        try:
            resp = self._sync_client.post("/api/chat", json=payload)
            resp.raise_for_status()
            logger.info(f"[Ollama] {self.model} successfully offloaded from VRAM.")
            return {"status": "unloaded", "model": self.model}
        except Exception as exc:
            logger.warning(f"[Ollama] Failed to unload {self.model} from VRAM: {exc}")
            return {"status": "error", "error": str(exc)}
        finally:
            self.close()

    async def warmup_kv_cache(
        self,
        static_system_prompt: str,
        tools: list[dict[str, Any]],
        keep_alive: str | int = "15m",
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": static_system_prompt}],
            "tools": tools,
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": 0.0,
                "num_predict": 1,
            },
        }
        logger.info(
            f"[Ollama Async] Pre-loading {self.model} (num_ctx={self.num_ctx}) & caching system prompt..."
        )
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return {
                "status": "loaded_and_kv_cached",
                "model": self.model,
                "num_ctx": self.num_ctx,
                "cached_tokens": data.get("prompt_eval_count", 0),
            }

    async def unload_model(self) -> dict[str, Any]:
        """Async version used by FastAPI (`server/app.py`) on tab close or 15m inactivity."""
        logger.info(f"[Ollama Async] Unloading {self.model} from GPU VRAM (keep_alive=0)...")
        payload = {
            "model": self.model,
            "messages": [],
            "keep_alive": 0,
        }
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            logger.info(f"[Ollama Async] {self.model} successfully offloaded from VRAM.")
            return {"status": "unloaded", "model": self.model}

    # -----------------------------------------------------------------
    # Internal Normalization
    # -----------------------------------------------------------------
    @staticmethod
    def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
        raw_tool_calls = msg.get("tool_calls") or []
        normalized_calls: list[dict[str, Any]] = []

        for tc in raw_tool_calls:
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
                    "function": {
                        "name": fn_name,
                        "arguments": parsed_args,
                    }
                }
            )

        out: dict[str, Any] = {
            "role": "assistant",
            "content": (msg.get("content") or "").strip(),
        }
        if normalized_calls:
            out["tool_calls"] = normalized_calls
        return out