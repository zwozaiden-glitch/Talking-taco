"""Tests for config-free logic: memory, splitting, providers, API contract."""

from __future__ import annotations

import json

import httpx
import pytest

from talkingtaco.config import AIConfig
from talkingtaco.memory import ConversationMemory
from talkingtaco.providers import AIError, ChatMessage, MockProvider
from talkingtaco.providers.openai_compatible import OpenAICompatibleProvider


# ------------------------------------------------------------------ helpers
def make_config(**overrides) -> AIConfig:
    base = {
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
        "model": "test-model",
        "system_prompt": "sys",
        "max_tokens": 100,
        "temperature": 0.5,
        "timeout": 5.0,
        "provider": "openai",
    }
    base.update(overrides)
    return AIConfig(**base)


# ------------------------------------------------------------------- memory
def test_memory_keeps_turns_and_orders_them():
    mem = ConversationMemory(max_turns=2)
    mem.append(1, "a", "A")
    mem.append(1, "b", "B")
    mem.append(1, "c", "C")
    history = mem.history(1)
    assert [m.content for m in history] == ["b", "B", "c", "C"]


def test_memory_is_isolated_per_conversation():
    mem = ConversationMemory(max_turns=4)
    mem.append(1, "hello", "hi")
    assert mem.history(2) == []


def test_memory_build_prompt_puts_system_first_then_history_then_user():
    mem = ConversationMemory(max_turns=4)
    mem.append(7, "old q", "old a")
    prompt = mem.build_prompt(7, "new q", "be nice")
    assert [m.role for m in prompt] == ["system", "user", "assistant", "user"]
    assert prompt[0].content == "be nice"
    assert prompt[-1].content == "new q"


def test_memory_respects_char_budget():
    mem = ConversationMemory(max_turns=50, max_chars=1000)
    for i in range(50):
        mem.append(1, f"question {i} " + "x" * 100, f"answer {i} " + "y" * 100)
    total = sum(len(m.content) for m in mem.history(1))
    assert total <= 1000


def test_memory_clear_and_eviction():
    mem = ConversationMemory(max_turns=2, max_conversations=2)
    for cid in range(5):
        mem.append(cid, "q", "a")
    assert len(mem._store) <= 2
    mem.clear(4)
    assert mem.history(4) == []


# -------------------------------------------------------------- message split
def test_split_message_short_passthrough():
    from talkingtaco.bot import split_message

    assert split_message("hello") == ["hello"]


def test_split_message_never_exceeds_limit():
    from talkingtaco.bot import split_message

    text = "\n\n".join("word " * 60 for _ in range(40))
    chunks = split_message(text, limit=2000)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
    assert "".join(chunks).replace("\n", "") .replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_split_message_hard_wraps_giant_paragraph():
    from talkingtaco.bot import split_message

    text = "z" * 1000 + " " + "y" * 1000 + " " + "x" * 1000
    chunks = split_message(text, limit=500)
    assert all(len(c) <= 500 for c in chunks)


# ----------------------------------------------------------------- providers
@pytest.mark.asyncio
async def test_mock_provider_echoes_and_streams():
    provider = MockProvider()
    reply = await provider.complete([ChatMessage("user", "hi there")])
    assert "hi there" in reply

    chunks = [c async for c in provider.stream([ChatMessage("user", "hi there")])]
    assert "".join(chunks).strip() == reply


@pytest.mark.asyncio
async def test_openai_compatible_complete_success():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "  hello world  "}}]}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(make_config(), client=client)

    reply = await provider.complete([ChatMessage("user", "hi")])

    assert reply == "hello world"
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert seen["body"]["stream"] is False
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_surfaces_api_error_message():
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(make_config(), client=client)

    with pytest.raises(AIError) as exc:
        await provider.complete([ChatMessage("user", "hi")])
    assert "bad key" in str(exc.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_rejects_empty_reply():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(make_config(), client=client)
    with pytest.raises(AIError):
        await provider.complete([ChatMessage("user", "hi")])
    await client.aclose()


@pytest.mark.asyncio
async def test_openai_compatible_stream_parses_sse():
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request):
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(make_config(), client=client)

    chunks = [c async for c in provider.stream([ChatMessage("user", "hi")])]
    assert chunks == ["Hel", "lo"]
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_handles_legacy_text_completion_shape():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"text": "legacy answer"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(make_config(), client=client)
    assert await provider.complete([ChatMessage("user", "hi")]) == "legacy answer"
    await client.aclose()


# ---------------------------------------------------------------- config/validation
def test_ai_config_reports_missing_fields():
    cfg = AIConfig(base_url="", api_key="", model="")
    problems = cfg.validate()
    assert len(problems) == 3
    assert any("AI_BASE_URL" in p for p in problems)


def test_bot_config_validation():
    from talkingtaco.config import BotConfig

    assert BotConfig(token="").validate() == ["DISCORD_TOKEN is not set"]
    assert BotConfig(token="x", trigger_mode="weird").validate()
    assert BotConfig(token="x", trigger_mode="both").validate() == []


def test_unknown_provider_raises_clear_error():
    from talkingtaco.providers import create_provider

    with pytest.raises(AIError) as exc:
        create_provider(make_config(provider="nope"))
    assert "nope" in str(exc.value)
    assert "mock" in str(exc.value)
