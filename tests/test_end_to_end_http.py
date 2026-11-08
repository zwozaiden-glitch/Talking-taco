"""Full-stack test: Discord message -> real HTTP call -> reply posted.

This is the closest thing to running the bot for real without Discord: it spins
up the fake OpenAI-compatible server, points the actual provider at it, and
drives the real ``TacoBot.on_message`` path.
"""

from __future__ import annotations

import threading

import httpx
import pytest

from scripts.fake_api_server import Handler
from talkingtaco.bot import TacoBot
from talkingtaco.config import AIConfig, BotConfig
from talkingtaco.providers.openai_compatible import OpenAICompatibleProvider
from tests.test_integration_fake_discord import (
    FakeChannel,
    FakeUser,
    dispatch,
)


class _ServerThread:
    """Run the fake API on a background thread with an OS-assigned port."""

    def __init__(self) -> None:
        from http.server import ThreadingHTTPServer

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        return False


def _live_bot(port: int) -> TacoBot:
    ai = AIConfig(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="anything",
        model="demo-model",
        system_prompt="Be Taco.",
        max_tokens=200,
        temperature=0.0,
        timeout=15.0,
        provider="openai",
    )
    settings = BotConfig(
        token="fake", trigger_mode="both", prefix="!taco",
        memory_turns=6, cooldown_seconds=0.0,
    )
    bot = TacoBot(
        bot_config=settings,
        ai_config=ai,
        provider=OpenAICompatibleProvider(ai, client=httpx.AsyncClient(timeout=15)),
    )
    bot._connection.user = FakeUser(999, bot=True)
    return bot


@pytest.mark.asyncio
async def test_bot_talks_to_real_http_endpoint():
    with _ServerThread() as server:
        bot = _live_bot(server.port)
        channel = FakeChannel(42)

        msg = await dispatch(bot, "<@999> hello from the network", channel=channel)

        assert msg.replies, "bot must reply after a successful HTTP round trip"
        output = msg.full_output
        assert "FakeAI" in output
        assert "demo-model" in output
        assert "hello from the network" in output

        # Memory recorded the exchange for the next turn.
        history = bot.memory.history(42)
        assert [m.role for m in history] == ["user", "assistant"]

        await bot.provider.aclose()


@pytest.mark.asyncio
async def test_second_turn_sends_conversation_context():
    with _ServerThread() as server:
        bot = _live_bot(server.port)
        channel = FakeChannel(43)

        await dispatch(bot, "!taco first", channel=channel)
        second = await dispatch(bot, "!taco second", channel=channel)

        # system + (user, assistant) history + new user = 4 messages.
        assert "4 message(s) of context" in second.full_output
        await bot.provider.aclose()


@pytest.mark.asyncio
async def test_unreachable_endpoint_gives_friendly_error():
    bot = _live_bot(port=1)  # nothing listens here
    bot.provider = OpenAICompatibleProvider(
        bot.ai_config, client=httpx.AsyncClient(timeout=2)
    )
    msg = await dispatch(bot, "!taco are you there")
    assert msg.replies
    assert "short-circuited" in msg.replies[0]
    await bot.provider.aclose()
