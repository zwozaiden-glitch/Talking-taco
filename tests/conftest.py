"""Keep tests hermetic: never let a developer's real .env leak into assertions."""

from __future__ import annotations

import os

import pytest

_LEAKY_PREFIXES = ("AI_", "DISCORD_", "BOT_", "MEMORY_", "COOLDOWN_", "REPLY_TO_", "ALLOWED_", "IGNORED_")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove config vars for the duration of each test."""
    for name in list(os.environ):
        if name.startswith(_LEAKY_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    yield
