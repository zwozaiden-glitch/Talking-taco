"""Environment-driven configuration for the Talking Taco bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _get_int(name: str, default: int) -> int:
    raw = _get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = _get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DEFAULT_SYSTEM_PROMPT = (
    "You are Taco, a friendly and witty AI assistant living inside a Discord server. "
    "Keep replies conversational and reasonably short, since this is chat, not an essay. "
    "Match the language the user writes in. If you are unsure about something, say so "
    "instead of inventing facts. Never claim to be a human."
)


@dataclass(frozen=True)
class AIConfig:
    """Everything needed to talk to your AI endpoint.

    Any OpenAI-compatible API works. Point ``base_url`` at your server and set
    ``api_key`` / ``model`` -- no code changes required when you swap providers.
    """

    base_url: str = field(default_factory=lambda: _get("AI_BASE_URL"))
    api_key: str = field(default_factory=lambda: _get("AI_API_KEY"))
    model: str = field(default_factory=lambda: _get("AI_MODEL", "gpt-4o-mini"))
    system_prompt: str = field(
        default_factory=lambda: _get("AI_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    )
    max_tokens: int = field(default_factory=lambda: _get_int("AI_MAX_TOKENS", 600))
    temperature: float = field(default_factory=lambda: _get_float("AI_TEMPERATURE", 0.8))
    timeout: float = field(default_factory=lambda: _get_float("AI_TIMEOUT", 60.0))
    provider: str = field(default_factory=lambda: _get("AI_PROVIDER", "openai").lower())

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty means OK)."""
        problems: list[str] = []
        if not self.base_url:
            problems.append("AI_BASE_URL is not set (e.g. https://api.openai.com/v1)")
        if not self.model:
            problems.append("AI_MODEL is not set (e.g. gpt-4o-mini)")
        if not self.api_key:
            problems.append(
                "AI_API_KEY is not set (use any placeholder if your endpoint ignores it)"
            )
        return problems


@dataclass(frozen=True)
class BotConfig:
    token: str = field(default_factory=lambda: _get("DISCORD_TOKEN"))
    # "mention" | "prefix" | "both"
    trigger_mode: str = field(default_factory=lambda: _get("BOT_TRIGGER_MODE", "both").lower())
    prefix: str = field(default_factory=lambda: _get("BOT_PREFIX", "!taco"))
    memory_turns: int = field(default_factory=lambda: _get_int("MEMORY_TURNS", 12))
    cooldown_seconds: float = field(default_factory=lambda: _get_float("COOLDOWN_SECONDS", 3.0))
    allowed_channel_ids: tuple[int, ...] = field(default_factory=lambda: _parse_ids("ALLOWED_CHANNEL_IDS"))
    ignored_channel_ids: tuple[int, ...] = field(default_factory=lambda: _parse_ids("IGNORED_CHANNEL_IDS"))
    reply_to_all_messages: bool = field(
        default_factory=lambda: _get("REPLY_TO_ALL_MESSAGES", "false").lower() in {"1", "true", "yes"}
    )

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.token:
            problems.append("DISCORD_TOKEN is not set")
        if self.trigger_mode not in {"mention", "prefix", "both"}:
            problems.append("BOT_TRIGGER_MODE must be one of: mention, prefix, both")
        return problems


def _parse_ids(name: str) -> tuple[int, ...]:
    raw = _get(name)
    if not raw:
        return ()
    ids: list[int] = []
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            ids.append(int(chunk))
    return tuple(ids)


def load_ai_config() -> AIConfig:
    return AIConfig()


def load_bot_config() -> BotConfig:
    return BotConfig()
