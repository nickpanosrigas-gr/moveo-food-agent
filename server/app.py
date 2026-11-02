# In server/app.py
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from server.speech import SpeechServiceClient
from src.core.agent_loop import FoodOrderAgent
from src.utils.logger import setup_logging

# Initialize single consistent logger across the app and Uvicorn
setup_logging()
logger = logging.getLogger("server.app")

FailureModeType = Literal["none", "kitchen_busy", "server_error", "schema_error", "business_refusal"]
INACTIVITY_TIMEOUT_SECONDS = 900


class SessionManager:
    """Manages per-tab FoodOrderAgent instances and deduplicates VRAM warmups."""

    def __init__(self, speech_client: SpeechServiceClient):
        self.speech = speech_client
        self.sessions: dict[str, FoodOrderAgent] = {}
        self.last_active: dict[str, float] = {}
        self.models_loaded: bool = False
        self._warmup_in_progress = False
        self._watchdog_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    def start_watchdog(self) -> None:
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._inactivity_loop())

    async def stop_watchdog(self) -> None:
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
        tasks = [self.speech.unload_models()]
        temp_agent = next(iter(self.sessions.values()), None)
        if temp_agent:
            tasks.append(temp_agent.llm.unload_model())
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.speech.close()

    async def get_or_create_agent(self, session_id: str) -> FoodOrderAgent:
        async with self._lock:
            now = time.monotonic()
            self.last_active[session_id] = now

            if session_id not in self.sessions:
                logger.info(f"[SESSION] Initialized new agent session '{session_id}'")
                self.sessions[session_id] = FoodOrderAgent(
                    log_path=f"logs/session_{session_id}.jsonl"
                )

            agent = self.sessions[session_id]

            # Deduplicate warmup in case React strict-mode calls this concurrently
            if not self.models_loaded and not self._warmup_in_progress:
                self._warmup_in_progress = True
                asyncio.create_task(self._run_warmup(agent))

            return agent

    async def _run_warmup(self, agent: FoodOrderAgent) -> None:
        try:
            logger.info("[VRAM] Warming up Faster-Whisper, Kokoro, and Ollama in parallel...")
            await asyncio.gather(
                self.speech.warmup_models(keep_alive=INACTIVITY_TIMEOUT_SECONDS),
                agent.warmup_async(),
                return_exceptions=True,
            )
            self.models_loaded = True
            logger.info("[VRAM] All models and KV caches ready.")
        finally:
            self._warmup_in_progress = False

    async def touch_session(self, session_id: str) -> None:
        async with self._lock:
            self.last_active[session_id] = time.monotonic()

    async def close_session(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            closing_agent = self.sessions.pop(session_id, None)
            self.last_active.pop(session_id, None)
            remaining = len(self.sessions)

        if remaining == 0:
            await asyncio.sleep(5.0)
            async with self._lock:
                if len(self.sessions) == 0 and self.models_loaded:
                    logger.info("[SESSION] All tabs closed. Offloading GPU VRAM...")
                    self.models_loaded = False
                    tasks = [self.speech.unload_models()]
                    if closing_agent is not None:
                        tasks.append(closing_agent.llm.unload_model())
                    unload_res = await asyncio.gather(*tasks, return_exceptions=True)
                    return {"status": "unloaded_vram", "details": [str(r) for r in unload_res]}

        return {"status": "session_closed", "active_sessions_remaining": remaining}

    async def reset_session(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            self.sessions[session_id] = FoodOrderAgent(
                log_path=f"logs/session_{session_id}.jsonl"
            )
            self.last_active[session_id] = time.monotonic()
            agent = self.sessions[session_id]
        logger.info(f"[SESSION] Reset cart and conversation for '{session_id}'")
        return {
            "status": "reset",
            "session_id": session_id,
            "cart_state": agent.cart.get_state_snapshot(),
        }

    async def _inactivity_loop(self) -> None:
        while True:
            await asyncio.sleep(30.0)
            now = time.monotonic()
            should_unload = False

            async with self._lock:
                expired = [
                    sid for sid, ts in self.last_active.items()
                    if (now - ts) >= INACTIVITY_TIMEOUT_SECONDS
                ]
                for sid in expired:
                    logger.info(f"[SESSION] Expired idle session '{sid}' (15m)")
                    self.last_active.pop(sid, None)
                    self.sessions.pop(sid, None)

                if not self.last_active and self.models_loaded:
                    self.models_loaded = False
                    should_unload = True

            if should_unload:
                logger.info("[VRAM] All sessions idle for 15m. Offloading VRAM...")
                await self.speech.unload_models()


speech_client = SpeechServiceClient(default_keep_alive=INACTIVITY_TIMEOUT_SECONDS)
session_manager = SessionManager(speech_client=speech_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    session_manager.start_watchdog()
    yield
    await session_manager.stop_watchdog()


app = FastAPI(
    title="Moveo Agentic Food Ordering Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextChatRequest(BaseModel):
    session_id: str = Field(default="default")
    message: str = Field(..., min_length=1)
    force_failure: FailureModeType | None = Field(default=None)


class SessionLifecycleRequest(BaseModel):
    session_id: str = Field(default="default")


def _ndjson_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


async def _stream_agent_turn(
    agent: FoodOrderAgent,
    user_text: str,
    force_failure: str | None = None,
) -> AsyncIterator[bytes]:
    try:
        agent_response = await asyncio.to_thread(
            agent.send,
            user_text,
            force_failure,
        )
        agent_message: str = agent_response.get("message", "")
        tool_calls: list[dict[str, Any]] = agent_response.get("tool_calls", [])

        yield _ndjson_line(
            {
                "type": "turn_metadata",
                "full_message": agent_message,
                "tool_calls": tool_calls,
                "cart_state": agent.cart.get_state_snapshot(),
                "turn_log": agent.last_turn_log,
                "mcp_payload": agent.last_mcp_payload,
            }
        )

        try:
            chunk_count = 0
            async for chunk_event in speech_client.synthesize_stream(
                full_text=agent_message,
                keep_alive=INACTIVITY_TIMEOUT_SECONDS,
            ):
                chunk_count += 1
                yield _ndjson_line(
                    {
                        "type": "agent_chunk",
                        "chunk_index": chunk_event["chunk_index"],
                        "total_chunks": chunk_event["total_chunks"],
                        "is_final": chunk_event["is_final"],
                        "text_chunk": chunk_event["text_chunk"],
                        "audio_b64": chunk_event["audio_b64"],
                        "sample_rate": chunk_event["sample_rate"],
                    }
                )

            if chunk_count == 0 and agent_message:
                yield _ndjson_line(
                    {
                        "type": "agent_chunk",
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "is_final": True,
                        "text_chunk": agent_message,
                        "audio_b64": None,
                        "sample_rate": 24000,
                    }
                )

        except Exception as tts_exc:
            logger.error(f"[TTS] Stream synthesis error: {tts_exc}", exc_info=True)
            yield _ndjson_line(
                {
                    "type": "agent_chunk",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "is_final": True,
                    "text_chunk": agent_message,
                    "audio_b64": None,
                    "sample_rate": 24000,
                    "tts_error": str(tts_exc),
                }
            )

        yield _ndjson_line({"type": "turn_complete"})

    except Exception as exc:
        logger.error(f"[AGENT] Turn error: {exc}", exc_info=True)
        yield _ndjson_line({"type": "error", "error": str(exc)})


@app.post("/api/chat/text")
async def chat_text_endpoint(req: TextChatRequest) -> StreamingResponse:
    agent = await session_manager.get_or_create_agent(req.session_id)
    await session_manager.touch_session(req.session_id)
    return StreamingResponse(
        _stream_agent_turn(agent=agent, user_text=req.message, force_failure=req.force_failure),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat/voice")
async def chat_voice_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    force_failure: FailureModeType | None = Form(None),
) -> StreamingResponse:
    agent = await session_manager.get_or_create_agent(session_id)
    await session_manager.touch_session(session_id)
    audio_bytes = await file.read()
    filename = file.filename or f"recording_{uuid.uuid4().hex[:6]}.wav"

    async def _voice_pipeline_generator() -> AsyncIterator[bytes]:
        if not audio_bytes:
            yield _ndjson_line({"type": "error", "error": "Empty audio recording received."})
            return

        try:
            transcribed_text = await speech_client.transcribe_chunks(
                audio_chunks=audio_bytes,
                filename=filename,
                keep_alive=INACTIVITY_TIMEOUT_SECONDS,
            )
            logger.info(f"[STT] Transcribed user voice: \"{transcribed_text}\"")
        except Exception as stt_exc:
            logger.error(f"[STT] Failed to transcribe: {stt_exc}", exc_info=True)
            yield _ndjson_line({"type": "error", "error": f"Speech-to-text failed: {stt_exc}"})
            return

        if not transcribed_text.strip():
            yield _ndjson_line({"type": "transcription", "text": "", "empty_speech": True})
            yield _ndjson_line({"type": "turn_complete"})
            return

        yield _ndjson_line({"type": "transcription", "text": transcribed_text})

        async for event_bytes in _stream_agent_turn(
            agent=agent,
            user_text=transcribed_text,
            force_failure=force_failure,
        ):
            yield event_bytes

    return StreamingResponse(
        _voice_pipeline_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/session/start")
async def start_session_endpoint(req: SessionLifecycleRequest) -> dict[str, Any]:
    agent = await session_manager.get_or_create_agent(req.session_id)
    return {
        "status": "ready",
        "session_id": req.session_id,
        "cart_state": agent.cart.get_state_snapshot(),
        "logs": agent.turn_logger.get_recent_logs(),
    }


@app.post("/api/session/close")
async def close_session_endpoint(req: SessionLifecycleRequest) -> dict[str, Any]:
    return await session_manager.close_session(req.session_id)


@app.post("/api/session/reset")
async def reset_session_endpoint(req: SessionLifecycleRequest) -> dict[str, Any]:
    return await session_manager.reset_session(req.session_id)