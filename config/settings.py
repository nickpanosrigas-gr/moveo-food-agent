# config/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

LLMProviderType = Literal["mistral", "llama-server"]
SpeechProviderType = Literal["mistral", "speech_service"]


@dataclass(frozen=True)
class Settings:
    # Provider Selectors (Default to mistral)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "llama-server").strip().lower()
    SPEECH_PROVIDER: str = os.getenv("SPEECH_PROVIDER", "speech_service").strip().lower()

    # Mistral Configuration
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "").strip()
    MISTRAL_API_URL: str = os.getenv("MISTRAL_API_URL", "https://api.mistral.ai").strip().rstrip("/")
    MISTRAL_LLM: str = os.getenv("MISTRAL_LLM", "ministral-14b-2512").strip()
    MISTRAL_STT: str = os.getenv("MISTRAL_STT", "voxtral-mini-transcribe-realtime-3").strip()
    MISTRAL_TTS: str = os.getenv("MISTRAL_TTS", "voxtral-mini-tts-3-0").strip()

    # Llama-Server Configuration (Required only when LLM_PROVIDER == "llama-server")
    LLAMA_URL: str = os.getenv("LLAMA_URL", "https://llama.pali.autos").strip().rstrip("/")
    LLAMA_MODEL: str = os.getenv("LLAMA_MODEL", "Qwen3.6-35B-A3B").strip()
    LLAMA_CTX: int = int(os.getenv("LLAMA_CTX", "16384") or "16384")

    # Local Speech Service Configuration (Required only when SPEECH_PROVIDER == "speech_service")
    SPEECH_SERVICE_URL: str = os.getenv("SPEECH_SERVICE_URL", "https://speech.pali.autos").strip().rstrip("/")

    # Moveo MCP Configuration
    APPLICANT_EMAIL: str = os.getenv("APPLICANT_EMAIL", "nickpanosrigas@gmail.com").strip()
    MCP_SERVER_URL: str = os.getenv(
        "MCP_SERVER_URL",
        "https://webviews.moveo.ai/api/demo/mcp",
    ).strip()

    # Paths
    MENU_PATH: Path = ROOT_DIR / "menu.yaml"
    DEFAULT_LOG_PATH: Path = ROOT_DIR / "logs" / "conversation_turns.jsonl"

    def __post_init__(self) -> None:
        if self.LLM_PROVIDER not in ("mistral", "llama-server"):
            raise ValueError(
                f"Invalid LLM_PROVIDER '{self.LLM_PROVIDER}'. Must be 'mistral' or 'llama-server'."
            )

        if self.LLM_PROVIDER == "mistral":
            if not self.MISTRAL_API_KEY:
                raise ValueError("MISTRAL_API_KEY is required when LLM_PROVIDER='mistral'.")
            if not self.MISTRAL_LLM:
                raise ValueError("MISTRAL_LLM is required when LLM_PROVIDER='mistral'.")
        elif self.LLM_PROVIDER == "llama-server":
            if not self.LLAMA_URL:
                raise ValueError("LLAMA_URL is required when LLM_PROVIDER='llama-server'.")
            if not self.LLAMA_MODEL:
                raise ValueError("LLAMA_MODEL is required when LLM_PROVIDER='llama-server'.")
            if self.LLAMA_CTX <= 0:
                raise ValueError("LLAMA_CTX must be a positive integer when LLM_PROVIDER='llama-server'.")

        if self.SPEECH_PROVIDER not in ("mistral", "speech_service"):
            raise ValueError(
                f"Invalid SPEECH_PROVIDER '{self.SPEECH_PROVIDER}'. Must be 'mistral' or 'speech_service'."
            )

        if self.SPEECH_PROVIDER == "mistral":
            if not self.MISTRAL_API_KEY:
                raise ValueError("MISTRAL_API_KEY is required when SPEECH_PROVIDER='mistral'.")
            if not self.MISTRAL_STT:
                raise ValueError("MISTRAL_STT is required when SPEECH_PROVIDER='mistral'.")
            if not self.MISTRAL_TTS:
                raise ValueError("MISTRAL_TTS is required when SPEECH_PROVIDER='mistral'.")
        elif self.SPEECH_PROVIDER == "speech_service":
            if not self.SPEECH_SERVICE_URL:
                raise ValueError("SPEECH_SERVICE_URL is required when SPEECH_PROVIDER='speech_service'.")


settings = Settings()