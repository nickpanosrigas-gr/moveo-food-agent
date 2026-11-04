# server/mistral_speech.py
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import random
import re
import wave
from typing import Any, AsyncIterator, Iterable

import httpx
import websockets
from websockets.exceptions import InvalidStatus, WebSocketException

from config.settings import settings
from src.clients.errors import (
    AuthenticationError,
    BadRequestError,
    ClientAPIError,
    EndpointOrModelNotFoundError,
    RateLimitError,
    UpstreamServerError,
    request_with_retry_async,
)

logger = logging.getLogger(__name__)

SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])|(?<=\n)\s*')
INTERNAL_VOICE_ID = "en_paul_neutral"


class MistralSpeechClient:
    manages_vram: bool = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        stt_model: str | None = None,
        tts_model: str | None = None,
        max_retries: int = 3,
    ):
        self.api_key = (api_key or settings.MISTRAL_API_KEY).strip()
        self.base_url = (base_url or settings.MISTRAL_API_URL).rstrip("/")
        self.ws_base_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        self.stt_model = stt_model or settings.MISTRAL_STT
        self.tts_model = tts_model or settings.MISTRAL_TTS
        self.tts_voice = INTERNAL_VOICE_ID
        self.max_retries = max_retries

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def warmup_models(self, keep_alive: int | None = None) -> dict[str, Any]:
        return {"status": "noop_cloud_provider", "provider": "mistral"}

    async def unload_models(self) -> dict[str, Any]:
        return {"status": "noop_cloud_provider", "provider": "mistral"}

    async def transcribe_chunks(
        self,
        audio_chunks: Iterable[bytes] | bytes,
        filename: str = "speech.wav",
        language: str = "en",
        keep_alive: int | None = None,
    ) -> str:
        audio_bytes = (
            b"".join(audio_chunks)
            if not isinstance(audio_chunks, (bytes, bytearray))
            else bytes(audio_chunks)
        )
        if not audio_bytes:
            return ""

        if "realtime" in self.stt_model.lower():
            return await self._transcribe_realtime_ws_with_retry(audio_bytes, language=language)
        return await self._transcribe_batch_rest(audio_bytes, filename=filename, language=language)

    async def _transcribe_realtime_ws_with_retry(self, wav_bytes: bytes, language: str = "en") -> str:
        for attempt in range(self.max_retries + 1):
            try:
                return await self._transcribe_realtime_ws(wav_bytes, language=language)
            except InvalidStatus as exc:
                status = getattr(exc.response, "status_code", None)
                body = ""
                if hasattr(exc.response, "body") and exc.response.body:
                    body = exc.response.body.decode("utf-8", errors="ignore")[:250]

                if status in (400, 422):
                    raise BadRequestError(
                        "Mistral Realtime STT",
                        f"Invalid parameters for model '{self.stt_model}': {body or exc}",
                        status_code=status,
                    ) from exc
                if status in (401, 403):
                    raise AuthenticationError(
                        "Mistral Realtime STT",
                        f"Authentication rejected on WebSocket handshake: {body or exc}",
                        status_code=status,
                    ) from exc
                if status == 404:
                    raise EndpointOrModelNotFoundError(
                        "Mistral Realtime STT",
                        f"Realtime STT endpoint or model '{self.stt_model}' not found (404): {body or exc}",
                        status_code=404,
                    ) from exc
                if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    delay = (2 ** attempt) * random.uniform(0.75, 1.25)
                    logger.warning(
                        f"[Mistral Realtime STT] WebSocket HTTP {status}. "
                        f"Retrying in {delay:.2f}s (retry {attempt + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    continue
                if status == 429:
                    raise RateLimitError(
                        "Mistral Realtime STT",
                        f"Rate limit exceeded on WebSocket handshake: {body or exc}",
                        status_code=429,
                    ) from exc
                raise UpstreamServerError(
                    "Mistral Realtime STT",
                    f"WebSocket handshake failed with HTTP {status}: {body or exc}",
                    status_code=status,
                ) from exc
            except (TimeoutError, WebSocketException, OSError) as exc:
                if attempt < self.max_retries:
                    delay = (2 ** attempt) * random.uniform(0.75, 1.25)
                    logger.warning(
                        f"[Mistral Realtime STT] Transient WebSocket error ({exc}). "
                        f"Retrying in {delay:.2f}s (retry {attempt + 1}/{self.max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    continue
                raise UpstreamServerError(
                    "Mistral Realtime STT",
                    f"WebSocket connection failed after {self.max_retries + 1} attempts: {exc}",
                ) from exc

        raise UpstreamServerError("Mistral Realtime STT", "Realtime transcription failed.")

    async def _transcribe_realtime_ws(self, wav_bytes: bytes, language: str = "en") -> str:
        pcm_bytes, sample_rate = self._extract_pcm_from_wav(wav_bytes)
        ws_url = f"{self.ws_base_url}/v1/audio/transcriptions/realtime?model={self.stt_model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        chunk_size = max((sample_rate // 10) * 2, 3200)
        deltas: list[str] = []

        async with websockets.connect(ws_url, additional_headers=headers, open_timeout=10.0) as ws:
            raw_init = await asyncio.wait_for(ws.recv(), timeout=15.0)
            init_event = json.loads(raw_init)
            if init_event.get("type") == "error":
                self._raise_ws_payload_error(init_event)

            session_update = {
                "type": "session.update",
                "session": {
                    "audio_format": {
                        "encoding": "pcm_s16le",
                        "sample_rate": sample_rate,
                    },
                    "language": language,
                },
            }
            await ws.send(json.dumps(session_update))

            for offset in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[offset : offset + chunk_size]
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
            await ws.send(json.dumps({"type": "input_audio.end"}))

            async for raw_msg in ws:
                event = json.loads(raw_msg)
                ev_type = event.get("type", "")

                if ev_type == "transcription.text.delta":
                    delta_text = event.get("text") or event.get("delta") or ""
                    deltas.append(delta_text)
                elif ev_type == "transcription.done":
                    final_text = event.get("text")
                    if final_text:
                        return str(final_text).strip()
                    break
                elif ev_type == "error":
                    self._raise_ws_payload_error(event)

        return "".join(deltas).strip()

    def _raise_ws_payload_error(self, event: dict[str, Any]) -> None:
        err_obj = event.get("error", event)
        msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
        code = str(err_obj.get("code", "")).lower() if isinstance(err_obj, dict) else ""

        if "404" in code or "not_found" in code or ("model" in msg.lower() and "not found" in msg.lower()):
            raise EndpointOrModelNotFoundError(
                "Mistral Realtime STT",
                f"Model '{self.stt_model}' not found: {msg}",
                status_code=404,
            )
        if "400" in code or "invalid" in code:
            raise BadRequestError("Mistral Realtime STT", msg, status_code=400)
        if "429" in code or "rate" in code:
            raise RateLimitError("Mistral Realtime STT", msg, status_code=429)
        raise ClientAPIError("Mistral Realtime STT", msg)

    async def _transcribe_batch_rest(
        self, audio_bytes: bytes, filename: str = "speech.wav", language: str = "en"
    ) -> str:
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model": self.stt_model, "language": language}
        response = await request_with_retry_async(
            request_fn=lambda: self._client.post("/v1/audio/transcriptions", files=files, data=data),
            provider="Mistral STT",
            context_hint=f"model='{self.stt_model}', endpoint='/v1/audio/transcriptions'",
            max_retries=self.max_retries,
        )
        return response.json().get("text", "").strip()

    @staticmethod
    def _extract_pcm_from_wav(wav_bytes: bytes) -> tuple[bytes, int]:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                sample_rate = wf.getframerate()
                pcm_frames = wf.readframes(wf.getnframes())
                return pcm_frames, sample_rate
        except wave.Error:
            return wav_bytes, 16000

    @staticmethod
    def _get_wav_sample_rate(wav_bytes: bytes, default: int = 24000) -> int:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                return wf.getframerate()
        except Exception:
            return default

    @staticmethod
    def split_into_chunks(text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        chunks = [c.strip() for c in SENTENCE_SPLIT_REGEX.split(cleaned) if c.strip()]
        return chunks if chunks else [cleaned]

    async def synthesize_chunk(self, text_chunk: str, keep_alive: int | None = None) -> bytes:
        payload = {
            "model": self.tts_model,
            "input": text_chunk,
            "voice_id": self.tts_voice,
            "response_format": "wav",
        }
        response = await request_with_retry_async(
            request_fn=lambda: self._client.post("/v1/audio/speech", json=payload),
            provider="Mistral TTS",
            context_hint=f"model='{self.tts_model}', voice='{self.tts_voice}', endpoint='/v1/audio/speech'",
            max_retries=self.max_retries,
        )

        content_type = response.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            body = response.json()
            audio_b64 = body.get("audio_data", "")
            return base64.b64decode(audio_b64) if audio_b64 else b""
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
            sample_rate = self._get_wav_sample_rate(wav_bytes, default=24000)
            yield {
                "type": "agent_response_chunk",
                "chunk_index": idx,
                "total_chunks": total,
                "is_final": idx == (total - 1),
                "text_chunk": sentence,
                "audio_b64": base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else None,
                "sample_rate": sample_rate,
            }