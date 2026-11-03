# config/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    # LLM Backend (llama-server)
    LLM_URL: str = os.getenv("LLM_URL").strip().rstrip("/")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B").strip()
    LLM_CTX: int = int(os.getenv("LLM_CTX", "16384"))

    # Moveo MCP & Speech Service Configuration
    APPLICANT_EMAIL: str = os.getenv("APPLICANT_EMAIL", "").strip()
    MCP_SERVER_URL: str = os.getenv(
        "MCP_SERVER_URL",
        "https://webviews-git-moveo-food-nlp-assignment-moveo-ai.vercel.app/api/demo/mcp",
    ).strip()
    SPEECH_URL: str = os.getenv("SPEECH_URL").strip().rstrip("/")

    # Paths
    MENU_PATH: Path = ROOT_DIR / "menu.yaml"
    DEFAULT_LOG_PATH: Path = ROOT_DIR / "logs" / "conversation_turns.jsonl"


settings = Settings()