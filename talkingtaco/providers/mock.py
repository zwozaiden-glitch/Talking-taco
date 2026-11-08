"""Offline provider used for demos, tests, and "I'll add my API later" setups.

Set ``AI_PROVIDER=mock`` and the bot runs end-to-end with zero credentials,
so you can verify the Discord wiring before your real endpoint exists.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from .base import AIProvider, ChatMessage

_CANNED = (
    "I'm running in mock mode, so this is a canned reply -- but the whole pipeline works!",
    "Taco here! Wire up AI_BASE_URL and AI_MODEL and I'll start giving real answers.",
    "Mock provider reporting for duty. Your Discord plumbing is officially alive.",
    "Nice, the bot received your message. Once your API is set I'll have something smarter to say.",
)


class MockProvider(AIProvider):
    name = "mock"

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay

    def _reply(self, messages: Sequence[ChatMessage]) -> str:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        snippet = last.strip().replace("\n", " ")[:120]
        # Deterministic per prompt: same input -> same reply, in both
        # complete() and stream(), which keeps tests and demos reproducible.
        base = _CANNED[hash(snippet) % len(_CANNED)]
        return f"{base}\n\n(You said: {snippet!r})" if snippet else base

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._reply(messages)

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        for word in self._reply(messages).split(" "):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield word + " "
