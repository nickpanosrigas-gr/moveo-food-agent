import os
import re
import base64
import asyncio
import logging
from typing import AsyncIterator, Iterable
import httpx

from config.settings import settings
logger = logging.getLogger(__name__)

# Matches sentence boundaries (. ! ? or —) without splitting decimals like "$11.50"
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])|(?<=\n)\s*')


class SpeechServiceClient:
    """
    Stateless async client for the GPU-accelerated Speech Service (Faster-Whisper + Kokoro-82M).
    Exposes primitives for VRAM lifecycle management, chunked STT, and sentence-chunked TTS.
    """

    def __init__(
        self,
        base_url: str | None = None,
        default_keep_alive: int = 900,
        stt_model: str = "large-v3-turbo",
        stt_compute: str = "int8_float16",
        tts_voice: str = "af_heart",
        tts_lang: str = "a",
        tts_speed: float = 1.0,
    ):
        self.base_url = (base_url or settings.SPEECH_URL).rstrip("/")
        self.default_keep_alive = default_keep_alive
        self.stt_model = stt_model
        self.stt_compute = stt_compute
        self.tts_voice = tts_voice
        self.tts_lang = tts_lang
        self.tts_speed = tts_speed
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    # -------------------------------------------------------------------------
    # 1. VRAM Lifecycle Primitives (Called by Session Manager in server/app.py)
    # -------------------------------------------------------------------------
    async def warmup_models(self, keep_alive: int | None = None) -> dict:
        """
        Pre-loads both Faster-Whisper and Kokoro-82M into GPU VRAM in parallel
        when a user opens the web app. Resets the 15-minute inactivity timer.
        """
        ttl = keep_alive if keep_alive is not None else self.default_keep_alive
        logger.info(f"Pre-loading STT & TTS models into VRAM (keep_alive={ttl}s)...")

        stt_task = self._client.post(
            "/manage/stt",
            json={
                "model_size": self.stt_model,
                "compute_type": self.stt_compute,
                "keep_alive": ttl,
            },
        )
        tts_task = self._client.post(
            "/manage/tts",
            json={
                "lang_code": self.tts_lang,
                "keep_alive": ttl,
            },
        )

        stt_resp, tts_resp = await asyncio.gather(stt_task, tts_task, return_exceptions=True)
        return {
            "stt": stt_resp.json() if isinstance(stt_resp, httpx.Response) and stt_resp.is_success else str(stt_resp),
            "tts": tts_resp.json() if isinstance(tts_resp, httpx.Response) and tts_resp.is_success else str(tts_resp),
        }

    async def unload_models(self) -> dict:
        """
        Forces immediate VRAM release (0 MB) for both Whisper and Kokoro.
        Call this when the last active tab closes or after 15 minutes of inactivity.
        """
        logger.info("Unloading STT & TTS models from GPU VRAM (keep_alive=0)...")
        stt_task = self._client.post("/manage/stt", json={"keep_alive": 0})
        tts_task = self._client.post("/manage/tts", json={"keep_alive": 0})

        stt_resp, tts_resp = await asyncio.gather(stt_task, tts_task, return_exceptions=True)
        return {
            "stt": stt_resp.json() if isinstance(stt_resp, httpx.Response) and stt_resp.is_success else str(stt_resp),
            "tts": tts_resp.json() if isinstance(tts_resp, httpx.Response) and tts_resp.is_success else str(tts_resp),
        }

    # -------------------------------------------------------------------------
    # 2. Speech-to-Text (STT) — Accepts Buffered or Streamed Audio Chunks
    # -------------------------------------------------------------------------
    async def transcribe_chunks(
        self,
        audio_chunks: Iterable[bytes] | bytes,
        filename: str = "speech.wav",
        language: str = "en",
        keep_alive: int | None = None,
    ) -> str:
        """
        Combines incoming audio chunks from the client VAD recording and sends
        them to Faster-Whisper. Automatically refreshes the 15-minute VRAM timer.
        """
        ttl = keep_alive if keep_alive is not None else self.default_keep_alive
        audio_bytes = b"".join(audio_chunks) if not isinstance(audio_chunks, (bytes, bytearray)) else bytes(audio_chunks)

        if not audio_bytes:
            return ""

        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {
            "model_size": self.stt_model,
            "compute_type": self.stt_compute,
            "language": language,
            "keep_alive": str(ttl),  # Crucial: prevents speech_service from unloading after transcription!
        }

        response = await self._client.post("/stt", files=files, data=data)
        response.raise_for_status()
        return response.json().get("text", "").strip()

    # -------------------------------------------------------------------------
    # 3. Text-to-Speech (TTS) — Sentence-Chunked Audio + Text Sync Generator
    # -------------------------------------------------------------------------
    @staticmethod
    def split_into_chunks(text: str) -> list[str]:
        """
        Splits an agent response into natural sentence chunks without breaking
        prices like '$11.50' or order IDs like '#ORD-12345'.
        Example: "Updated: Large Cola. Your total is $22.25. Ready to submit?"
        -> ["Updated: Large Cola.", "Your total is $22.25.", "Ready to submit?"]
        """
        cleaned = text.strip()
        if not cleaned:
            return []
        chunks = [c.strip() for c in SENTENCE_SPLIT_REGEX.split(cleaned) if c.strip()]
        return chunks if chunks else [cleaned]

    async def synthesize_chunk(self, text_chunk: str, keep_alive: int | None = None) -> bytes:
        """Synthesizes a single text chunk into 24kHz WAV bytes via Kokoro-82M."""
        ttl = keep_alive if keep_alive is not None else self.default_keep_alive
        payload = {
            "text": text_chunk,
            "voice": self.tts_voice,
            "speed": self.tts_speed,
            "lang_code": self.tts_lang,
            "keep_alive": ttl,  # Crucial: keeps Kokoro in VRAM for the next chunk/turn!
        }
        response = await self._client.post("/tts", json=payload)
        response.raise_for_status()
        return response.content

    async def synthesize_stream(
        self, full_text: str, keep_alive: int | None = None
    ) -> AsyncIterator[dict]:
        """
        Splits the LLM response into sentence chunks and pipelined-synthesizes them.
        Yields synchronized `{chunk_index, total_chunks, text_chunk, audio_b64}` dicts
        so the frontend can stream text and audio in lockstep with minimal Time-To-First-Audio.
        """
        sentences = self.split_into_chunks(full_text)
        total = len(sentences)
        if total == 0:
            return

        # Fire off synthesis tasks in sequence (since speech_service uses a single Kokoro lock)
        # while yielding chunk 1 immediately to the WebSocket as soon as it finishes (~120ms).
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