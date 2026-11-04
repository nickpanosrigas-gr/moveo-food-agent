# src/clients/__init__.py
from src.clients.errors import (
    AuthenticationError,
    BadRequestError,
    ClientAPIError,
    EndpointOrModelNotFoundError,
    RateLimitError,
    UpstreamServerError,
)
from src.clients.llama_client import LlamaClient
from src.clients.mistral_client import MistralClient
from src.clients.router import (
    LLMClientProtocol,
    SpeechClientProtocol,
    any_service_uses_vram,
    create_llm_client,
    create_speech_client,
    llm_uses_vram,
    speech_uses_vram,
)

__all__ = [
    "LlamaClient",
    "MistralClient",
    "LLMClientProtocol",
    "SpeechClientProtocol",
    "create_llm_client",
    "create_speech_client",
    "llm_uses_vram",
    "speech_uses_vram",
    "any_service_uses_vram",
    "ClientAPIError",
    "BadRequestError",
    "AuthenticationError",
    "EndpointOrModelNotFoundError",
    "RateLimitError",
    "UpstreamServerError",
]