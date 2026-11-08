"""Google Gemini provider (Google AI Studio / Gemini API).

Keys starting with ``AIzaSy`` are Google API keys, not OpenAI keys, and Gemini
uses a different wire format than OpenAI. This provider translates the bot's
generic message list into Gemini's ``contents``/``system_instruction`` shape and
back, so everything above it (memory, streaming, commands) is unchanged.

Endpoints used:
    POST {base}/models/{model}:generateContent
    POST {base}/models/{model}:streamGenerateContent?alt=sse
with the key sent in the ``x-goog-api-key`` header.

Docs: https://ai.google.dev/api/generate-content
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from ..config import AIConfig
from .base import AIError, AIProvider, ChatMessage

log = logging.getLogger("talkingtaco.gemini")

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Preferred models for auto-detection, best first. "flash" models are the right
# trade-off for chat: fast and cheap. Anything unavailable is simply skipped.
_PREFERRED_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
)


class GeminiProvider(AIProvider):
    name = "gemini"

    def __init__(self, config: AIConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.timeout)
        self._base = (config.base_url or DEFAULT_BASE_URL).rstrip("/")
        # Model may be empty/"auto": resolved lazily from the API's model list.
        requested = (config.model or "").strip()
        self._model = "" if requested.lower() in {"", "auto", "default"} else requested
        self._resolve_lock = asyncio.Lock()

    # ------------------------------------------------------------------ setup
    def _url(self, model: str, stream: bool) -> str:
        method = "streamGenerateContent" if stream else "generateContent"
        url = f"{self._base}/models/{model}:{method}"
        return f"{url}?alt=sse" if stream else url

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self._config.api_key,
        }

    async def _get_model(self) -> str:
        if self._model:
            return self._model
        async with self._resolve_lock:
            if self._model:
                return self._model
            self._model = await self._discover_model()
            log.info("Auto-selected Gemini model: %s", self._model)
            return self._model

    async def _discover_model(self) -> str:
        """Ask the API which models this key can use and pick a good one."""
        try:
            response = await self._client.get(
                f"{self._base}/models", headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise AIError(f"Could not list Gemini models: {exc}") from exc

        if response.status_code >= 400:
            raise AIError(
                "Could not list Gemini models "
                f"({response.status_code}: {_error_text(response)}). "
                "Set AI_MODEL explicitly, e.g. AI_MODEL=gemini-2.5-flash"
            )

        try:
            entries = response.json().get("models", [])
        except ValueError as exc:
            raise AIError("Gemini model list was not valid JSON") from exc

        by_name: dict[str, dict[str, Any]] = {}
        for entry in entries:
            name = str(entry.get("name", "")).removeprefix("models/")
            methods = entry.get("supportedGenerationMethods") or []
            if name and "generateContent" in methods:
                by_name[name] = entry

        for preferred in _PREFERRED_MODELS:
            if preferred in by_name:
                return preferred
        # Otherwise: any flash model, else anything that can generate content.
        flash = [n for n in by_name if "flash" in n and "thinking" not in n]
        if flash:
            return min(flash)
        if by_name:
            return min(by_name)

        available = ", ".join(sorted(by_name)) or "none"
        raise AIError(
            "No Gemini model on this key supports generateContent "
            f"(available: {available}). Check the key's project in AI Studio."
        )

    # ------------------------------------------------------------- translation
    @staticmethod
    def _build_payload(
        messages: Sequence[ChatMessage], config: AIConfig, model: str, thinking: bool
    ) -> dict[str, Any]:
        """Translate generic chat messages into Gemini's request shape."""
        systems: list[str] = []
        contents: list[dict[str, Any]] = []
        last_role: str | None = None

        for message in messages:
            if message.role == "system":
                systems.append(message.content)
                continue

            role = "model" if message.role == "assistant" else "user"
            # Gemini expects alternating turns starting with "user".
            if not contents and role == "model":
                continue
            if role == last_role:
                contents[-1]["parts"].append({"text": message.content})
            else:
                contents.append({"role": role, "parts": [{"text": message.content}]})
                last_role = role

        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": config.max_tokens,
                "temperature": config.temperature,
            },
        }
        if systems:
            payload["system_instruction"] = {"parts": [{"text": "\n\n".join(systems)}]}
        if thinking:
            # Thinking tokens count against maxOutputTokens and delay the first
            # visible token, so it is off by default for snappy Discord replies.
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": config.thinking_budget
            }
        return payload

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Pull assistant text out of a Gemini response, with useful errors."""
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            if block:
                raise AIError(
                    f"Gemini blocked the prompt ({block}). Try rephrasing it."
                )
            raise AIError("Gemini returned no candidates (possibly filtered).")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and not part.get("thought")
        )
        if text:
            return text

        reason = candidate.get("finishReason")
        if reason == "SAFETY":
            raise AIError("Gemini refused to answer for safety reasons.")
        if reason == "MAX_TOKENS":
            raise AIError(
                "Gemini hit the token limit before writing anything. "
                "Raise AI_MAX_TOKENS (or AI_THINKING_BUDGET=0)."
            )
        if reason:
            raise AIError(f"Gemini stopped without text (finishReason={reason}).")
        raise AIError("Gemini returned an empty reply.")

    # --------------------------------------------------------------- complete
    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        model = await self._get_model()
        data = await self._post(model, messages, stream=False)
        try:
            return self._extract_text(data).strip()
        except AIError:
            raise
        except Exception as exc:  # defensive: never leak raw KeyErrors upward
            raise AIError("Gemini returned an unexpected response shape") from exc

    async def _post(
        self, model: str, messages: Sequence[ChatMessage], stream: bool
    ) -> dict[str, Any]:
        url = self._url(model, stream=False)
        payload = self._build_payload(messages, self._config, model, thinking=True)

        try:
            response = await self._client.post(url, headers=self._headers(), json=payload)
            # Not every model supports thinkingConfig; retry once without it.
            if response.status_code == 400 and "thinking" in response.text.lower():
                log.debug("Model %s rejected thinkingConfig; retrying without it", model)
                payload = self._build_payload(
                    messages, self._config, model, thinking=False
                )
                response = await self._client.post(
                    url, headers=self._headers(), json=payload
                )
        except httpx.HTTPError as exc:
            raise AIError(f"Could not reach Gemini: {exc}") from exc

        if response.status_code >= 400:
            raise AIError(_friendly_error(response))

        try:
            return response.json()
        except ValueError as exc:
            raise AIError("Gemini returned a non-JSON response") from exc

    # ----------------------------------------------------------------- stream
    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        model = await self._get_model()
        payload = self._build_payload(messages, self._config, model, thinking=True)

        try:
            async with self._client.stream(
                "POST",
                self._url(model, stream=True),
                headers=self._headers(),
                json=payload,
            ) as response:
                if response.status_code == 400:
                    body = await response.aread()
                    if b"thinking" in body.lower():
                        log.debug("Stream rejected thinkingConfig; retrying without it")
                        async for chunk in self._stream_without_thinking(model, messages):
                            yield chunk
                        return
                    raise AIError(_friendly_error(response, body))
                if response.status_code >= 400:
                    body = await response.aread()
                    raise AIError(_friendly_error(response, body))

                emitted = False
                async for line in response.aiter_lines():
                    text = _parse_sse_line(line)
                    if text:
                        emitted = True
                        yield text
                if not emitted:
                    # Streaming produced nothing: fall back to a normal call so
                    # the user still gets an answer instead of silence.
                    yield await self.complete(messages)
        except httpx.HTTPError as exc:
            raise AIError(f"Could not reach Gemini: {exc}") from exc

    async def _stream_without_thinking(
        self, model: str, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[str]:
        payload = self._build_payload(messages, self._config, model, thinking=False)
        async with self._client.stream(
            "POST",
            self._url(model, stream=True),
            headers=self._headers(),
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise AIError(_friendly_error(response, body))
            async for line in response.aiter_lines():
                text = _parse_sse_line(line)
                if text:
                    yield text

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()


def _parse_sse_line(line: str) -> str | None:
    """Extract text from one Gemini SSE line."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    try:
        return GeminiProvider._extract_text(data)
    except AIError:
        # Streaming chunks can legitimately carry no text (e.g. usage metadata).
        return None


def _error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    return str(body)[:300]


def _friendly_error(response: httpx.Response, body: bytes | None = None) -> str:
    """Turn Gemini's error codes into something a human can act on."""
    detail = _error_text(response) if body is None else body.decode("utf-8", "replace")[:300]

    if response.status_code in (401, 403):
        return (
            f"Gemini rejected the API key ({response.status_code}: {detail}). "
            "Check AI_API_KEY in .env - keys start with AIzaSy and come from "
            "https://aistudio.google.com/apikey"
        )
    if response.status_code == 404:
        return (
            f"Gemini model not found ({detail}). Set AI_MODEL to a model your key "
            "can use - run `python -m talkingtaco --check` to list them."
        )
    if response.status_code == 429:
        return (
            f"Gemini rate limit / quota exceeded ({detail}). Wait a moment or "
            "check your plan at https://ai.dev/rate-limit"
        )
    return f"Gemini API error {response.status_code}: {detail}"
