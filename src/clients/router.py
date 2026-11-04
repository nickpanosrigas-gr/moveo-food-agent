from __future__ import annotations

import logging
from typing import Protocol, Any, AsyncIterator, Iterable

from config.settings import settings
from src.clients.llama_client import LlamaClient
from src.clients.mistral_client import MistralClient
from server.speech import SpeechServiceClient
from server.mistral_speech import MistralSpeechClient

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    manages_vram: bool
    model: str
    num_ctx: int | None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
    def warmup_kv_cache_sync(self, static_system_prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any]: ...
    async def warmup_kv_cache(self, static_system_prompt: str, tools: list[dict[str, Any]]) -> dict[str, Any]: ...
    def unload_model_sync(self) -> dict[str, Any]: ...
    async def unload_model(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class SpeechClientProtocol(Protocol):
    manages_vram: bool

    async def warmup_models(self, keep_alive: int | None = None) -> dict[str, Any]: ...
    async def unload_models(self) -> dict[str, Any]: ...
    async def transcribe_chunks(
        self,
        audio_chunks: Iterable[bytes] | bytes,
        filename: str = "speech.wav",
        language: str = "en",
        keep_alive: int | None = None,
    ) -> str: ...
    async def synthesize_chunk(self, text_chunk: str, keep_alive: int | None = None) -> bytes: ...
    def synthesize_stream(self, full_text: str, keep_alive: int | None = None) -> AsyncIterator[dict[str, Any]]: ...
    async def close(self) -> None: ...


def llm_uses_vram() -> bool:
    return settings.LLM_PROVIDER == "llama-server"


def speech_uses_vram() -> bool:
    return settings.SPEECH_PROVIDER == "speech_service"


def any_service_uses_vram() -> bool:
    return llm_uses_vram() or speech_uses_vram()


def create_llm_client(
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
    api_key: str | None = None,
) -> LLMClientProtocol:
    """Master router factory for instantiating the configured LLM client."""
    active_provider = (provider or settings.LLM_PROVIDER).strip().lower()

    if active_provider == "mistral":
        logger.info(f"[Router] Using Mistral LLM Client (model={model or settings.MISTRAL_LLM})")
        return MistralClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    elif active_provider == "llama-server":
        logger.info(f"[Router] Using Llama-Server Client (model={model or settings.LLAMA_MODEL})")
        return LlamaClient(
            base_url=base_url,
            model=model,
            num_ctx=num_ctx,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{active_provider}'")


def create_speech_client(
    provider: str | None = None,
    default_keep_alive: int = 900,
) -> SpeechClientProtocol:
    """Master router factory for instantiating the configured Speech (STT/TTS) client."""
    active_provider = (provider or settings.SPEECH_PROVIDER).strip().lower()

    if active_provider == "mistral":
        logger.info(
            f"[Router] Using Mistral Speech Client (STT={settings.MISTRAL_STT}, TTS={settings.MISTRAL_TTS})"
        )
        return MistralSpeechClient()
    elif active_provider == "speech_service":
        logger.info(f"[Router] Using Local SpeechServiceClient (url={settings.SPEECH_SERVICE_URL})")
        return SpeechServiceClient(default_keep_alive=default_keep_alive)
    else:
        raise ValueError(f"Unsupported SPEECH_PROVIDER: '{active_provider}'")