# src/utils/logger.py
from __future__ import annotations

import json
import logging
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class UvicornLoggerNameFilter(logging.Filter):
    """
    Rewrites the logger name from 'uvicorn.error' to 'uvicorn.server' for non-error logs.
    Prevents standard INFO lifecycle logs from showing '[uvicorn.error]'.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.error":
            record.name = "uvicorn.server" if record.levelno < logging.ERROR else "uvicorn.error"
        return True


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures a single, consistent logging format across the entire application
    (both CLI and FastAPI/Uvicorn), while silencing noisy network libraries and
    renaming misleading uvicorn.error logs.
    """
    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    filter_instance = UvicornLoggerNameFilter()

    # 1. Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicate output
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    console_handler.addFilter(filter_instance)
    root_logger.addHandler(console_handler)

    # 2. Silence verbose HTTP library logs
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 3. Align Uvicorn loggers to use the exact same formatter and filter
    for uvicorn_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        u_log = logging.getLogger(uvicorn_name)
        u_log.handlers.clear()
        u_log.addHandler(console_handler)
        u_log.addFilter(filter_instance)
        u_log.propagate = False


class TurnLogger:
    """Structured conversation turn logger (Requirement #5)."""

    def __init__(
        self,
        log_path: str | Path = "logs/conversation_turns.jsonl",
        session_id: str | None = None,
        max_memory_entries: int = 200,
    ):
        self.log_path = Path(log_path)
        self.session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        self._turn_counter = 0
        self._lock = threading.Lock()
        self._memory_buffer: deque[dict[str, Any]] = deque(maxlen=max_memory_entries)

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"[TurnLogger] Could not create log directory {self.log_path.parent}: {exc}")

    def log_turn(
        self,
        user_message: str,
        agent_message: str,
        exposed_tool_calls: list[dict[str, Any]] | None = None,
        internal_tool_calls: list[dict[str, Any]] | None = None,
        cart_snapshot: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._turn_counter += 1
            turn_number = self._turn_counter

            entry: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self.session_id,
                "turn_id": turn_number,
                "user_message": user_message,
                "agent_message": agent_message,
                "tool_calls": exposed_tool_calls or [],
                "internal_cart_actions": internal_tool_calls or [],
                "cart_state": cart_snapshot or {},
            }

            if extra_metadata:
                entry["metadata"] = extra_metadata

            self._memory_buffer.append(entry)

            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.error(f"[TurnLogger] Failed to write turn {turn_number} to {self.log_path}: {exc}")

            return entry

    def get_recent_logs(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._memory_buffer)

    def clear_memory_logs(self) -> None:
        with self._lock:
            self._memory_buffer.clear()
            self._turn_counter = 0