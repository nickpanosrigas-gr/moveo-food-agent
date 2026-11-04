# server/speech.py
from __future__ import annotations

import asyncio
import base64
import logging
import re
from typing import Any, AsyncIterator, Iterable

import httpx

from config.settings import settings
from src.clients.errors import request_with_retry_async

logger = logging.getLogger(__name__)

SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])|(?<=\n)\s*')


class SpeechServiceClient:
    """
    Stateless async client for the GPU-accelerated Speech Service (Faster-Whisper + Kokoro-82M).
    Exposes primitives for VRAM lifecycle management, chunked STT, and sentence-chunked TTS.
    """

    manages_vram: bool = True

    def __init__(
        self,
        base_url: str | None = None,
        default_keep_alive: int = 900,
        stt_model: str = "large-v3-turbo",
        stt_compute: str = "int8_float16",
        tts_voice: str = "af_heart",
        tts_lang: str = "a",
        tts_speed: float = 1.0,
        max_retries: int = 3,
    ):
        self.base_url = (base_url or settings.SPEECH_SERVICE_URL).rstrip("/")
        self.default_keep_alive = default_keep_alive
        self.stt_model = stt_model
        self.stt_compute = stt_compute
        self.tts_voice = tts_voice
        self.tts_lang = tts_lang
        self.tts_speed = tts_speed
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    # -------------------------------------------------------------------------
    # 1. VRAM Lifecycle Primitives
    # -------------------------------------------------------------------------
    async def warmup_models(self, keep_alive: int | None = None) -> dict[str, Any]:
        ttl = keep_alive if keep_alive is not None else self.default_keep_alive
        logger.info(f"Pre-loading STT & TTS models into VRAM (keep_alive={ttl}s)...")

        stt_task = request_with_retry_async(
            request_fn=lambda: self._client.post(
                "/manage/stt",
                json={
                    "model_size": self.stt_model,
                    "compute_type": self.stt_compute,
                    "keep_alive": ttl,
                },
            ),
            provider="speech_service",
            context_hint=f"url='{self.base_url}/manage/stt'",
            max_retries=self.max_retries,
        )
        tts_task = request_with_retry_async(
            request_fn=lambda: self._client.post(
                "/manage/tts",
                json={
                    "lang_code": self.tts_lang,
                    "keep_alive": ttl,
                },
            ),
            provider="speech_service",
            context_hint=f"url='{self.base_url}/manage/tts'",
            max_retries=self.max_retries,
        )

        stt_resp, tts_resp = await asyncio.gather(stt_task, tts_task, return_exceptions=True)
        return {
            "stt": stt_resp.json() if isinstance(stt_resp, httpx.Response) else str(stt_resp),
            "tts": tts_resp.json() if isinstance(tts_resp, httpx.Response) else str(tts_resp),
        }

    async def unload_models(self) -> dict[str, Any]:
        logger.info("Unloading STT & TTS models from GPU VRAM (keep_alive=0)...")
        stt_task = request_with_retry_async(
            request_fn=lambda: self._client.post("/manage/stt", json={"keep_alive": 0}),
            provider="speech_service",
            context_hint=f"url='{self.base_url}/manage/stt'",
            max_retries=self.max_retries,
        )
        tts_task = request_with_retry_async(
            request_fn=lambda: self._client.post("/manage/tts", json={"keep_alive": 0}),
            provider="speech_service",
            context_hint=f"url='{self.base_url}/manage/tts'",
            max_retries=self.max_retries,
        )

        stt_resp, tts_resp = await asyncio.gather(stt_task, tts_task, return_exceptions=True)
        return {
            "stt": stt_resp.json() if isinstance(stt_resp, httpx.Response) else str(stt_resp),
            "tts": tts_resp.json() if isinstance(tts_resp, httpx.Response) else str(tts_resp),
        }

    # -------------------------------------------------------------------------
    # 2. Speech-to-Text (STT)
    # -------------------------------------------------------------------------
    async def transcribe_chunks(
        self,
        audio_chunks: Iterable[bytes] | bytes,
        filename: str = "speech.wav",
        language: str = "en",
        keep_alive: int | None = None,
    ) -> str:
        ttl = keep_alive if keep_alive is not None else self.default_keep_alive
        audio_bytes = (
            b"".join(audio_chunks)
            if not isinstance(audio_chunks, (bytes, bytearray))
            else bytes(audio_chunks)
        )

        if not audio_bytes:
            return ""

        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {
            "model_size": self.stt_model,
            "compute_type": self.stt_compute,
            "language": language,
            "keep_alive": str(ttl),
        }

        response = await request_with_retry_async(
            request_fn=lambda: self._client.post("/stt", files=files, data=data),
            provider="speech_service STT",
            context_hint=f"url='{self.base_url}/stt', model='{self.stt_model}'",
            max_retries=self.max_retries,
        )
        return response.json().get("text", "").strip()

    # -------------------------------------------------------------------------
    # 3. Text-to-Speech (TTS)
    # -------------------------------------------------------------------------
    @staticmethod
    def split_into_chunks(text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        chunks = [c.strip() for c in SENTENCE_SPLIT_REGEX.split(cleaned) if c.strip()]
        return chunks if chunks else [cleaned]

    async def synthesize_chunk(self, text_chunk: str, keep_alive: int | None = None) -> bytes:
        ttl = keep_alive if keep_alive is not None else self.default_keep_alive
        payload = {
            "text": text_chunk,
            "voice": self.tts_voice,
            "speed": self.tts_speed,
            "lang_code": self.tts_lang,
            "keep_alive": ttl,
        }
        response = await request_with_retry_async(
            request_fn=lambda: self._client.post("/tts", json=payload),
            provider="speech_service TTS",
            context_hint=f"url='{self.base_url}/tts', voice='{self.tts_voice}'",
            max_retries=self.max_retries,
        )
        return response.content

    async def synthesize_stream(
        self, full_text: str, keep_alive: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        sentences = self.split_into_chunks(full_text)
        total = len(sentences)
        if total == 0:
            return

        for idx, sentence in enumerate(sentences):
            wav_bytes = await self.synthesize_chunk(sentence, keep_alive=keep_alive)
            yield {
                "type": "agent_response_chunk",
                "chunk_index": idx,
                "total_chunks": total,
                "is_final": idx == (total - 1),
                "text_chunk": sentence,
                "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
                "sample_rate": 24000,
            }