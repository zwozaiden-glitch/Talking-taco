"""Per-conversation short-term memory.

Each Discord channel (or DM) gets its own rolling window, so the bot can follow
a thread of conversation without ever growing unbounded or leaking history
between unrelated channels.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .providers import ChatMessage


@dataclass
class _Conversation:
    messages: list[ChatMessage] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)


class ConversationMemory:
    """Bounded LRU store of conversation windows."""

    def __init__(
        self,
        max_turns: int = 12,
        max_conversations: int = 500,
        max_chars: int = 12000,
        ttl_seconds: float = 60 * 60 * 6,
    ) -> None:
        self.max_turns = max(1, max_turns)
        self.max_conversations = max(1, max_conversations)
        self.max_chars = max(500, max_chars)
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[int, _Conversation] = OrderedDict()

    # ------------------------------------------------------------------ public
    def history(self, conversation_id: int) -> list[ChatMessage]:
        convo = self._store.get(conversation_id)
        if convo is None:
            return []
        if self._expired(convo):
            self._store.pop(conversation_id, None)
            return []
        self._store.move_to_end(conversation_id)
        return list(convo.messages)

    def append(
        self,
        conversation_id: int,
        user_message: str,
        assistant_message: str,
    ) -> None:
        convo = self._store.setdefault(conversation_id, _Conversation())
        convo.last_used = time.monotonic()
        convo.messages.append(ChatMessage("user", user_message))
        convo.messages.append(ChatMessage("assistant", assistant_message))
        self._trim(convo)
        self._store.move_to_end(conversation_id)
        self._evict()

    def clear(self, conversation_id: int) -> None:
        self._store.pop(conversation_id, None)

    def build_prompt(
        self,
        conversation_id: int,
        user_message: str,
        system_prompt: str,
    ) -> list[ChatMessage]:
        prompt: list[ChatMessage] = []
        if system_prompt:
            prompt.append(ChatMessage("system", system_prompt))
        prompt.extend(self.history(conversation_id))
        prompt.append(ChatMessage("user", user_message))
        return prompt

    def stats(self, conversation_id: int) -> dict[str, int]:
        convo = self._store.get(conversation_id)
        count = len(convo.messages) if convo else 0
        return {"messages": count, "conversations": len(self._store)}

    # ----------------------------------------------------------------- helpers
    def _expired(self, convo: _Conversation) -> bool:
        return (
            self.ttl_seconds > 0
            and (time.monotonic() - convo.last_used) > self.ttl_seconds
        )

    def _trim(self, convo: _Conversation) -> None:
        limit = self.max_turns * 2  # user + assistant per turn
        if len(convo.messages) > limit:
            del convo.messages[: len(convo.messages) - limit]

        # Second pass: respect a rough character budget, newest first.
        used = 0
        kept: list[ChatMessage] = []
        for msg in reversed(convo.messages):
            size = len(msg.content)
            if kept and used + size > self.max_chars:
                break
            kept.append(msg)
            used += size
        kept.reverse()
        convo.messages = kept

    def _evict(self) -> None:
        while len(self._store) > self.max_conversations:
            self._store.popitem(last=False)


def trim_to_budget(messages: Iterable[ChatMessage], max_chars: int) -> list[ChatMessage]:
    """Keep the newest messages that fit within ``max_chars`` (system kept first)."""
    messages = list(messages)
    if not messages:
        return []
    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]

    used = sum(len(m.content) for m in system)
    kept: list[ChatMessage] = []
    for msg in reversed(rest):
        size = len(msg.content)
        if kept and used + size > max_chars:
            break
        kept.append(msg)
        used += size
    kept.reverse()
    return system + kept
