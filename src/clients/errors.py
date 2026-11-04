# src/clients/errors.py
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class ClientAPIError(RuntimeError):
    """Base exception for LLM and Speech client errors."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        prefix = f"[{provider} HTTP {status_code}] " if status_code else f"[{provider}] "
        super().__init__(f"{prefix}{message}")


class BadRequestError(ClientAPIError):
    """HTTP 400 / 422: Invalid payload, parameter, or schema."""


class AuthenticationError(ClientAPIError):
    """HTTP 401 / 403: Invalid or missing API key / permissions."""


class EndpointOrModelNotFoundError(ClientAPIError):
    """HTTP 404: Model name or endpoint URL does not exist."""


class RateLimitError(ClientAPIError):
    """HTTP 429: Rate limit exceeded after maximum retries."""


class UpstreamServerError(ClientAPIError):
    """HTTP 5xx or network transport failure."""


def extract_error_detail(response: httpx.Response) -> str:
    """Extracts human-readable error messages from Mistral, OpenAI/llama-server, or FastAPI JSON payloads."""
    try:
        data = response.json()
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], str):
                return data["message"]
            if "error" in data:
                err = data["error"]
                if isinstance(err, dict) and "message" in err:
                    return str(err["message"])
                return str(err)
            if "detail" in data:
                return str(data["detail"])
    except Exception:
        pass
    text = response.text.strip()
    return text[:300] if text else f"HTTP {response.status_code} {response.reason_phrase}"


def raise_mapped_http_error(
    response: httpx.Response,
    provider: str,
    context_hint: str = "",
) -> None:
    """Maps non-2xx HTTP responses to specific, actionable ClientAPIError subclasses."""
    if response.is_success:
        return

    status = response.status_code
    detail = extract_error_detail(response)
    hint_suffix = f" ({context_hint})" if context_hint else ""

    if status in (400, 422):
        raise BadRequestError(
            provider,
            f"Bad request / validation failed{hint_suffix}: {detail}",
            status_code=status,
        )
    if status in (401, 403):
        raise AuthenticationError(
            provider,
            f"Authentication failed. Verify your API key / credentials{hint_suffix}: {detail}",
            status_code=status,
        )
    if status == 404:
        raise EndpointOrModelNotFoundError(
            provider,
            f"Resource, endpoint, or model not found{hint_suffix}: {detail}",
            status_code=404,
        )
    if status == 429:
        raise RateLimitError(
            provider,
            f"Rate limit exceeded{hint_suffix}: {detail}",
            status_code=429,
        )
    if status >= 500:
        raise UpstreamServerError(
            provider,
            f"Upstream server error{hint_suffix}: {detail}",
            status_code=status,
        )

    raise ClientAPIError(provider, f"Unexpected HTTP error{hint_suffix}: {detail}", status_code=status)


def _compute_backoff(
    attempt: int,
    response: httpx.Response | None = None,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
) -> float:
    """Calculates exponential backoff with full jitter, honoring Retry-After headers on 429s."""
    if response is not None:
        retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after) + random.uniform(0.1, 0.5), max_delay)
            except ValueError:
                pass
    exp = min(max_delay, base_delay * (2 ** attempt))
    return exp * random.uniform(0.75, 1.25)


def request_with_retry_sync(
    request_fn: Callable[[], httpx.Response],
    provider: str,
    context_hint: str = "",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> httpx.Response:
    """
    Executes a synchronous httpx request with exponential backoff for 429/5xx/network errors.
    Performs 1 initial attempt + up to max_retries retries.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = request_fn()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = _compute_backoff(attempt, base_delay=base_delay)
                logger.warning(
                    f"[{provider}] Network/timeout error ({exc.__class__.__name__}: {exc}). "
                    f"Retrying in {delay:.2f}s (retry {attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)
                continue
            raise UpstreamServerError(
                provider,
                f"Connection failed after {max_retries + 1} attempts ({context_hint}): {exc}",
            ) from exc

        if response.is_success:
            return response

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            delay = _compute_backoff(attempt, response=response, base_delay=base_delay)
            detail = extract_error_detail(response)
            logger.warning(
                f"[{provider}] HTTP {response.status_code} ({detail}). "
                f"Retrying in {delay:.2f}s (retry {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)
            continue

        raise_mapped_http_error(response, provider=provider, context_hint=context_hint)

    raise UpstreamServerError(provider, f"Request failed ({context_hint}): {last_exc}")


async def request_with_retry_async(
    request_fn: Callable[[], Awaitable[httpx.Response]],
    provider: str,
    context_hint: str = "",
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> httpx.Response:
    """
    Executes an asynchronous httpx request with exponential backoff for 429/5xx/network errors.
    Performs 1 initial attempt + up to max_retries retries.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = await request_fn()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = _compute_backoff(attempt, base_delay=base_delay)
                logger.warning(
                    f"[{provider}] Async network/timeout error ({exc.__class__.__name__}: {exc}). "
                    f"Retrying in {delay:.2f}s (retry {attempt + 1}/{max_retries})..."
                )
                await asyncio.sleep(delay)
                continue
            raise UpstreamServerError(
                provider,
                f"Connection failed after {max_retries + 1} attempts ({context_hint}): {exc}",
            ) from exc

        if response.is_success:
            return response

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            delay = _compute_backoff(attempt, response=response, base_delay=base_delay)
            detail = extract_error_detail(response)
            logger.warning(
                f"[{provider}] HTTP {response.status_code} ({detail}). "
                f"Retrying in {delay:.2f}s (retry {attempt + 1}/{max_retries})..."
            )
            await asyncio.sleep(delay)
            continue

        raise_mapped_http_error(response, provider=provider, context_hint=context_hint)

    raise UpstreamServerError(provider, f"Async request failed ({context_hint}): {last_exc}")