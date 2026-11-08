"""Package init for the Talking Taco bot."""

from __future__ import annotations

__version__ = "1.0.0"

from .config import AIConfig, BotConfig, load_ai_config, load_bot_config
from .memory import ConversationMemory
from .providers import AIError, AIProvider, ChatMessage, create_provider

__all__ = [
    "AIConfig",
    "AIError",
    "AIProvider",
    "BotConfig",
    "ChatMessage",
    "ConversationMemory",
    "__version__",
    "create_provider",
    "load_ai_config",
    "load_bot_config",
]
