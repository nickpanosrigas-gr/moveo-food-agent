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
    # Ollama LLM Configuration
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://192.168.1.220:11434").strip().rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:12b").strip()
    OLLAMA_CTX: int = int(os.getenv("OLLAMA_CTX", "32000"))

    # Moveo MCP & Speech Service Configuration
    APPLICANT_EMAIL: str = os.getenv("APPLICANT_EMAIL", "").strip()
    MCP_SERVER_URL: str = os.getenv(
        "MCP_SERVER_URL",
        "https://webviews-git-moveo-food-nlp-assignment-moveo-ai.vercel.app/api/demo/mcp",
    ).strip()
    SPEECH_URL: str = os.getenv("SPEECH_URL", "http://localhost:8000").strip().rstrip("/")

    # Paths
    MENU_PATH: Path = ROOT_DIR / "menu.yaml"
    DEFAULT_LOG_PATH: Path = ROOT_DIR / "logs" / "conversation_turns.jsonl"


settings = Settings()