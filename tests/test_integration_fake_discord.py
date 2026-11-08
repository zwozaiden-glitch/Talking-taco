"""End-to-end tests of the reply pipeline using fake Discord objects.

This exercises the real TacoBot code path -- trigger detection, memory,
streaming, chunking -- without touching the Discord network.
"""

from __future__ import annotations

import types

import pytest

from talkingtaco.bot import TacoBot
from talkingtaco.config import AIConfig, BotConfig
from talkingtaco.providers.mock import MockProvider


# --------------------------------------------------------------- fake objects
class FakeUser:
    def __init__(self, user_id: int, bot: bool = False) -> None:
        self.id = user_id
        self.bot = bot


class FakeSender:
    """A message the bot posted; records the edits made to it."""

    def __init__(self, channel) -> None:
        self.channel = channel
        self.edits: list[str] = []
        self.sent: list[str] = []

    async def edit(self, content: str) -> None:
        self.edits.append(content)


class FakeMessage:
    def __init__(self, bot: TacoBot, content: str, author: FakeUser, channel) -> None:
        self.bot = bot
        self.content = content
        self.author = author
        self.channel = channel
        self.mentions: list[FakeUser] = []
        self.replies: list[str] = []
        self.senders: list[FakeSender] = []
        self.reactions: list[str] = []
        if bot.user and f"<@{bot.user.id}>" in content:
            self.mentions.append(bot.user)

    async def reply(self, content: str, mention_author: bool = False):
        self.replies.append(content)
        sender = FakeSender(self.channel)
        self.senders.append(sender)
        return sender

    async def add_reaction(self, emoji: str) -> None:
        self.reactions.append(emoji)

    @property
    def full_output(self) -> str:
        """Everything the bot produced, streamed edits included."""
        parts = list(self.replies)
        for sender in self.senders:
            parts.extend(sender.edits)
        return "\n".join(parts)


class FakeChannel:
    def __init__(self, channel_id: int) -> None:
        self.id = channel_id
        self.sent: list[str] = []

    def typing(self):
        class _Typing:
            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *exc):
                return False

        return _Typing()

    async def send(self, content: str):
        self.sent.append(content)


# ------------------------------------------------------------------- fixture
def make_bot(**bot_overrides) -> TacoBot:
    ai = AIConfig(
        base_url="https://example.test/v1",
        api_key="k",
        model="m",
        system_prompt="Be Taco.",
        max_tokens=100,
        temperature=0.5,
        timeout=5.0,
        provider="openai",
    )
    settings_kwargs = {
        "token": "fake",
        "trigger_mode": "both",
        "prefix": "!taco",
        "memory_turns": 6,
        "cooldown_seconds": 0.0,
    }
    settings_kwargs.update(bot_overrides)
    settings = BotConfig(**settings_kwargs)
    bot = TacoBot(bot_config=settings, ai_config=ai, provider=MockProvider())
    # discord.py sets .user on login; simulate it for trigger detection.
    bot._connection.user = FakeUser(999, bot=True)
    return bot


async def dispatch(bot: TacoBot, content: str, channel=None, author=None):
    channel = channel or FakeChannel(1)
    message = FakeMessage(bot, content, author or FakeUser(42), channel)
    await bot.on_message(message)
    return message


# --------------------------------------------------------------------- tests
@pytest.mark.asyncio
async def test_mention_triggers_reply():
    bot = make_bot()
    msg = await dispatch(bot, "<@999> what is a taco?")
    assert msg.replies, "bot should have replied to a mention"
    assert "what is a taco?" in msg.full_output


@pytest.mark.asyncio
async def test_prefix_triggers_reply():
    bot = make_bot()
    msg = await dispatch(bot, "!taco hello there")
    assert msg.replies
    assert "hello there" in msg.full_output


@pytest.mark.asyncio
async def test_unrelated_chatter_is_ignored():
    bot = make_bot()
    msg = await dispatch(bot, "just chatting with friends")
    assert msg.replies == []


@pytest.mark.asyncio
async def test_empty_mention_prompts_for_question():
    bot = make_bot()
    msg = await dispatch(bot, "<@999>")
    assert msg.replies
    assert "question" in msg.replies[0].lower()


@pytest.mark.asyncio
async def test_bots_are_ignored():
    bot = make_bot()
    msg = await dispatch(bot, "!taco hi", author=FakeUser(7, bot=True))
    assert msg.replies == []


@pytest.mark.asyncio
async def test_memory_accumulates_across_messages():
    bot = make_bot()
    channel = FakeChannel(55)
    await dispatch(bot, "!taco first question", channel=channel)
    await dispatch(bot, "!taco second question", channel=channel)
    history = bot.memory.history(55)
    assert len(history) == 4
    assert history[0].content == "first question"


@pytest.mark.asyncio
async def test_ignored_channel_gets_no_reply():
    bot = make_bot()
    bot.settings = BotConfig(token="fake", trigger_mode="both", prefix="!taco",
                             ignored_channel_ids=(77,), cooldown_seconds=0.0)
    msg = await dispatch(bot, "!taco hi", channel=FakeChannel(77))
    assert msg.replies == []


@pytest.mark.asyncio
async def test_allowlist_blocks_other_channels():
    bot = make_bot(allowed_channel_ids=(5,))
    blocked = await dispatch(bot, "!taco hi", channel=FakeChannel(6))
    assert blocked.replies == []
    allowed = await dispatch(bot, "!taco hi", channel=FakeChannel(5))
    assert allowed.replies


@pytest.mark.asyncio
async def test_provider_failure_returns_friendly_error():
    bot = make_bot()

    async def boom(self, messages):
        from talkingtaco.providers import AIError

        raise AIError("upstream exploded")
        yield ""  # pragma: no cover

    bot.provider.stream = types.MethodType(boom, bot.provider)
    msg = await dispatch(bot, "!taco hi")
    assert msg.replies
    assert "short-circuited" in msg.replies[0]


@pytest.mark.asyncio
async def test_long_reply_is_split_across_messages():
    bot = make_bot()

    async def long_reply(self, messages):
        yield "word " * 900

    bot.provider.stream = types.MethodType(long_reply, bot.provider)
    channel = FakeChannel(9)
    msg = await dispatch(bot, "!taco essay please", channel=channel)
    assert channel.sent, "overflow chunks should be sent to the channel"
    assert all(len(c) <= 2000 for c in channel.sent)
    # Nothing was silently dropped: everything fits in the delivered messages.
    delivered = "".join(s.edits[-1] for s in msg.senders) + "".join(channel.sent)
    assert delivered.count("word") == 900


@pytest.mark.asyncio
async def test_cooldown_acks_with_reaction_and_stays_quiet():
    bot = make_bot(cooldown_seconds=30.0)
    first = await dispatch(bot, "!taco one")
    assert first.replies, "first message should be answered"
    second = await dispatch(bot, "!taco two")
    assert second.replies == [], "second message is inside the cooldown"
    assert second.reactions, "bot should react with a 'wait' signal"


@pytest.mark.asyncio
async def test_reply_to_all_messages_mode_answers_plain_chat():
    bot = make_bot(reply_to_all_messages=True)
    msg = await dispatch(bot, "how is everyone today")
    assert msg.replies, "reply_to_all_messages should trigger on plain text"


@pytest.mark.asyncio
async def test_mention_mode_ignores_prefix():
    bot = make_bot(trigger_mode="mention")
    msg = await dispatch(bot, "!taco hi there")
    assert msg.replies == [], "prefix must be ignored in mention-only mode"
