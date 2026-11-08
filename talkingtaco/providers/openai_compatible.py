"""OpenAI-compatible chat provider.

Works with OpenAI, Azure OpenAI, OpenRouter, Groq, Together, Ollama, LM Studio,
vLLM, LiteLLM, or your own private endpoint -- anything exposing
``POST {base_url}/chat/completions``.

This is intentionally the *only* place that knows about HTTP/JSON details, so
pointing the bot at your own API later is a matter of changing ``AI_BASE_URL``
(and ``AI_MODEL`` / ``AI_API_KEY``), not editing code.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import httpx

from ..config import AIConfig
from .base import AIError, AIProvider, ChatMessage


class OpenAICompatibleProvider(AIProvider):
    name = "openai"

    def __init__(self, config: AIConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.timeout)
        self._url = config.base_url.rstrip("/") + "/chat/completions"

    # ------------------------------------------------------------------ utils
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _payload(self, messages: Sequence[ChatMessage], stream: bool) -> dict:
        return {
            "model": self._config.model,
            "messages": [m.as_dict() for m in messages],
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "stream": stream,
        }

    @staticmethod
    def _explain(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = response.text[:500]
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and err.get("message"):
                return f"{response.status_code}: {err['message']}"
            if isinstance(err, str):
                return f"{response.status_code}: {err}"
            if body.get("message"):
                return f"{response.status_code}: {body['message']}"
        return f"{response.status_code}: {body}"

    # --------------------------------------------------------------- complete
    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers(),
                json=self._payload(messages, stream=False),
            )
        except httpx.HTTPError as exc:
            raise AIError(f"Could not reach AI endpoint: {exc}") from exc

        if response.status_code >= 400:
            raise AIError(f"AI endpoint rejected the request ({self._explain(response)})")

        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content")
            if content is None:
                content = choice.get("text", "")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AIError("AI endpoint returned an unexpected response shape") from exc

        text = (content or "").strip()
        if not text:
            raise AIError("AI endpoint returned an empty reply")
        return text

    # ----------------------------------------------------------------- stream
    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST",
                self._url,
                headers=self._headers(),
                json=self._payload(messages, stream=True),
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise AIError(
                        f"AI endpoint rejected the request ({self._explain(response)})"
                    )
                async for line in response.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk:
                        yield chunk
        except httpx.HTTPError as exc:
            raise AIError(f"Could not reach AI endpoint: {exc}") from exc

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()


def _parse_sse_line(line: str) -> str | None:
    """Extract the text delta from one Server-Sent-Events line."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload in {"", "[DONE]"}:
        return None
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if not content:
        # Some servers send the whole message instead of a delta.
        content = (choices[0].get("message") or {}).get("content")
    return content or None
