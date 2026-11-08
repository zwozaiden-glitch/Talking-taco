"""Provider abstraction: anything that can turn a message list into a reply."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class AIError(RuntimeError):
    """Raised when the upstream AI API cannot produce a usable answer."""


class AIProvider(abc.ABC):
    """Minimal interface the bot needs.

    To support a brand-new API shape, subclass this, implement ``complete``
    and (optionally) ``stream``, then register it in ``talkingtaco.providers``.
    The rest of the bot never needs to change.
    """

    name: str = "base"

    @abc.abstractmethod
    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant's full reply for ``messages``."""

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """Yield reply chunks as they arrive.

        Default implementation degrades gracefully to a single chunk so a
        provider can opt out of streaming just by not overriding this.
        """
        yield await self.complete(messages)

    async def aclose(self) -> None:
        """Release network resources. Safe to call multiple times."""
        return
