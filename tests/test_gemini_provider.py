"""Tests for the Google Gemini provider (wire format, errors, streaming)."""

from __future__ import annotations

import json

import httpx
import pytest

from talkingtaco.config import AIConfig
from talkingtaco.providers import AIError, ChatMessage
from talkingtaco.providers.gemini import GeminiProvider

BASE = "https://generativelanguage.googleapis.com/v1beta"


def make_config(**overrides) -> AIConfig:
    base = {
        "base_url": BASE,
        "api_key": "AIzaSyTESTKEY",
        "model": "gemini-2.5-flash",
        "system_prompt": "Be Taco.",
        "max_tokens": 300,
        "temperature": 0.7,
        "timeout": 5.0,
        "provider": "gemini",
        "thinking_budget": 0,
    }
    base.update(overrides)
    return AIConfig(**base)


def reply_json(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}], "role": "model"}}]}


def provider_with(handler, **overrides) -> tuple[GeminiProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiProvider(make_config(**overrides), client=client), client


# ------------------------------------------------------- request translation
@pytest.mark.asyncio
async def test_sends_key_header_and_correct_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply_json("hi there"))

    provider, client = provider_with(handler)
    result = await provider.complete([ChatMessage("user", "hello")])

    assert result == "hi there"
    assert seen["url"] == f"{BASE}/models/gemini-2.5-flash:generateContent"
    assert seen["key"] == "AIzaSyTESTKEY"
    await client.aclose()


@pytest.mark.asyncio
async def test_translates_system_prompt_and_history():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply_json("ok"))

    provider, client = provider_with(handler)
    await provider.complete(
        [
            ChatMessage("system", "Be Taco."),
            ChatMessage("user", "first"),
            ChatMessage("assistant", "reply one"),
            ChatMessage("user", "second"),
        ]
    )

    body = seen["body"]
    assert body["system_instruction"] == {"parts": [{"text": "Be Taco."}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "first"}]},
        {"role": "model", "parts": [{"text": "reply one"}]},
        {"role": "user", "parts": [{"text": "second"}]},
    ]
    assert body["generationConfig"]["maxOutputTokens"] == 300
    assert body["generationConfig"]["temperature"] == 0.7
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    await client.aclose()


@pytest.mark.asyncio
async def test_consecutive_same_role_messages_are_merged():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply_json("ok"))

    provider, client = provider_with(handler)
    await provider.complete(
        [ChatMessage("user", "part A"), ChatMessage("user", "part B")]
    )
    # Gemini rejects two consecutive user turns.
    assert seen["body"]["contents"] == [
        {"role": "user", "parts": [{"text": "part A"}, {"text": "part B"}]}
    ]
    await client.aclose()


@pytest.mark.asyncio
async def test_leading_assistant_message_is_dropped():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply_json("ok"))

    provider, client = provider_with(handler)
    await provider.complete(
        [ChatMessage("assistant", "stale"), ChatMessage("user", "real question")]
    )
    # Conversations must start with a user turn.
    assert seen["body"]["contents"][0]["role"] == "user"
    await client.aclose()


@pytest.mark.asyncio
async def test_omits_system_instruction_when_prompt_empty():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply_json("ok"))

    provider, client = provider_with(handler, system_prompt="")
    await provider.complete([ChatMessage("user", "hi")])
    assert "system_instruction" not in seen["body"]
    await client.aclose()


# --------------------------------------------------------------- responses
@pytest.mark.asyncio
async def test_joins_multiple_text_parts():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "Hello "}, {"text": "world"}]}}
                ]
            },
        )

    provider, client = provider_with(handler)
    assert await provider.complete([ChatMessage("user", "hi")]) == "Hello world"
    await client.aclose()


@pytest.mark.asyncio
async def test_ignores_thought_parts():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "internal musing", "thought": True},
                                {"text": "visible answer"},
                            ]
                        }
                    }
                ]
            },
        )

    provider, client = provider_with(handler)
    assert await provider.complete([ChatMessage("user", "hi")]) == "visible answer"
    await client.aclose()


@pytest.mark.asyncio
async def test_blocked_prompt_raises_clear_error():
    def handler(request):
        return httpx.Response(
            200, json={"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}
        )

    provider, client = provider_with(handler)
    with pytest.raises(AIError) as exc:
        await provider.complete([ChatMessage("user", "hi")])
    assert "blocked" in str(exc.value).lower()
    await client.aclose()


@pytest.mark.asyncio
async def test_safety_finish_reason_raises():
    def handler(request):
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]},
        )

    provider, client = provider_with(handler)
    with pytest.raises(AIError) as exc:
        await provider.complete([ChatMessage("user", "hi")])
    assert "safety" in str(exc.value).lower()
    await client.aclose()


# ------------------------------------------------------------ error mapping
@pytest.mark.asyncio
async def test_bad_key_error_is_actionable():
    def handler(request):
        return httpx.Response(
            403, json={"error": {"message": "API key not valid.", "code": 403}}
        )

    provider, client = provider_with(handler)
    with pytest.raises(AIError) as exc:
        await provider.complete([ChatMessage("user", "hi")])
    message = str(exc.value)
    assert "API key not valid" in message
    assert "AI_API_KEY" in message
    await client.aclose()


@pytest.mark.asyncio
async def test_rate_limit_error_mentions_quota():
    def handler(request):
        return httpx.Response(
            429, json={"error": {"message": "Quota exceeded for quota metric"}}
        )

    provider, client = provider_with(handler)
    with pytest.raises(AIError) as exc:
        await provider.complete([ChatMessage("user", "hi")])
    assert "quota" in str(exc.value).lower() or "rate limit" in str(exc.value).lower()
    await client.aclose()


@pytest.mark.asyncio
async def test_missing_model_error_suggests_check():
    def handler(request):
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    provider, client = provider_with(handler)
    with pytest.raises(AIError) as exc:
        await provider.complete([ChatMessage("user", "hi")])
    assert "--check" in str(exc.value)
    await client.aclose()


# -------------------------------------------------------------- thinking
@pytest.mark.asyncio
async def test_retries_without_thinking_when_model_rejects_it():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if "thinkingConfig" in body.get("generationConfig", {}):
            return httpx.Response(
                400,
                json={"error": {"message": "thinkingConfig is not supported"}},
            )
        return httpx.Response(200, json=reply_json("worked on retry"))

    provider, client = provider_with(handler)
    assert await provider.complete([ChatMessage("user", "hi")]) == "worked on retry"
    assert len(calls) == 2
    assert "thinkingConfig" not in calls[1]["generationConfig"]
    await client.aclose()


# ------------------------------------------------------------ model discovery
@pytest.mark.asyncio
async def test_auto_model_selection_prefers_flash():
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-2.5-pro",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/gemini-2.5-flash",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/embedding-001",
                            "supportedGenerationMethods": ["embedContent"],
                        },
                    ]
                },
            )
        return httpx.Response(200, json=reply_json("from flash"))

    provider, client = provider_with(handler, model="auto")
    assert await provider.complete([ChatMessage("user", "hi")]) == "from flash"
    assert provider._model == "gemini-2.5-flash"
    await client.aclose()


@pytest.mark.asyncio
async def test_model_discovery_skips_models_without_generate_content():
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
                        {
                            "name": "models/gemini-9-flash",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                    ]
                },
            )
        return httpx.Response(200, json=reply_json("ok"))

    provider, client = provider_with(handler, model="")
    await provider.complete([ChatMessage("user", "hi")])
    assert provider._model == "gemini-9-flash"
    await client.aclose()


@pytest.mark.asyncio
async def test_model_discovery_with_no_usable_model_errors():
    def handler(request):
        return httpx.Response(200, json={"models": []})

    provider, client = provider_with(handler, model="auto")
    with pytest.raises(AIError) as exc:
        await provider.complete([ChatMessage("user", "hi")])
    assert "No Gemini model" in str(exc.value)
    await client.aclose()


# ------------------------------------------------------------------ streaming
@pytest.mark.asyncio
async def test_stream_uses_sse_endpoint_and_parses_chunks():
    seen = {}
    sse = (
        'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n\n'
        'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}\n\n'
    )

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    provider, client = provider_with(handler)
    chunks = [c async for c in provider.stream([ChatMessage("user", "hi")])]

    assert chunks == ["Hel", "lo"]
    assert ":streamGenerateContent" in seen["url"]
    assert "alt=sse" in seen["url"]
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_falls_back_when_server_sends_nothing():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "streamGenerateContent" in str(request.url):
            return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=reply_json("fallback answer"))

    provider, client = provider_with(handler)
    chunks = [c async for c in provider.stream([ChatMessage("user", "hi")])]
    assert "".join(chunks).strip() == "fallback answer"
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_error_status_raises():
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "quota"}})

    provider, client = provider_with(handler)
    with pytest.raises(AIError):
        async for _ in provider.stream([ChatMessage("user", "hi")]):
            pass
    await client.aclose()


# ------------------------------------------------------------ end-to-end
@pytest.mark.asyncio
async def test_bot_pipeline_with_gemini_backend():
    """The full bot path works when the AI backend is Gemini."""
    from talkingtaco.bot import TacoBot
    from talkingtaco.config import BotConfig
    from tests.test_integration_fake_discord import FakeChannel, FakeUser, dispatch

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"models": [{"name": "models/gemini-2.5-flash",
                                  "supportedGenerationMethods": ["generateContent"]}]},
            )
        body = json.loads(request.content)
        question = body["contents"][-1]["parts"][0]["text"]
        return httpx.Response(
            200,
            json=reply_json(f"Gemini says: you asked about {question!r}"),
        )

    ai = make_config(model="auto")
    settings = BotConfig(
        token="fake", trigger_mode="both", prefix="!taco",
        memory_turns=6, cooldown_seconds=0.0,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bot = TacoBot(
        bot_config=settings, ai_config=ai, provider=GeminiProvider(ai, client=client)
    )
    bot._connection.user = FakeUser(999, bot=True)

    msg = await dispatch(bot, "!taco what is a taco?", channel=FakeChannel(1))

    assert msg.replies
    assert "what is a taco?" in msg.full_output
    assert bot.memory.history(1), "exchange should be remembered"
    await client.aclose()
