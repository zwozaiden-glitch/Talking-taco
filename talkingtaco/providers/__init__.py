"""Provider registry.

Swapping backends is a config change, never a code change::

    AI_PROVIDER=openai    # any OpenAI-compatible endpoint (default)
    AI_PROVIDER=mock      # no network, canned replies

To add a custom API shape, write a class implementing ``AIProvider`` and
register it here -- roughly ten lines of wiring.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import AIConfig
from .base import AIError, AIProvider, ChatMessage, Role
from .mock import MockProvider
from .openai_compatible import OpenAICompatibleProvider

ProviderFactory = Callable[[AIConfig], AIProvider]

_REGISTRY: dict[str, ProviderFactory] = {
    "openai": OpenAICompatibleProvider,
    "mock": lambda config: MockProvider(),
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _REGISTRY[name.lower()] = factory


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def create_provider(config: AIConfig) -> AIProvider:
    key = (config.provider or "openai").lower()
    try:
        factory = _REGISTRY[key]
    except KeyError:
        raise AIError(
            f"Unknown AI_PROVIDER {config.provider!r}. "
            f"Available: {', '.join(available_providers())}"
        ) from None
    return factory(config)


__all__ = [
    "AIError",
    "AIProvider",
    "ChatMessage",
    "MockProvider",
    "OpenAICompatibleProvider",
    "Role",
    "available_providers",
    "create_provider",
    "register_provider",
]
