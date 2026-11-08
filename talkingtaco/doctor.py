"""Preflight checks: verify the token and the AI endpoint before going live.

    python -m talkingtaco --check

Runs three independent checks and tells you exactly what to fix:

1. Config      -- required env vars are present and well-formed
2. AI endpoint -- ``POST /chat/completions`` returns a usable answer
3. Discord     -- the bot token authenticates and the bot's identity is known
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from .config import AIConfig, BotConfig
from .providers import AIError, ChatMessage, create_provider

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


async def check_ai(ai: AIConfig) -> CheckResult:
    problems = ai.validate()
    if problems:
        return CheckResult("AI config", FAIL, "; ".join(problems))

    try:
        provider = create_provider(ai)
    except AIError as exc:
        return CheckResult("AI config", FAIL, str(exc))

    try:
        if provider.name == "mock":
            return CheckResult("AI endpoint", WARN, "AI_PROVIDER=mock (canned replies)")
        reply = await provider.complete(
            [ChatMessage("user", "Reply with exactly: pong")]
        )
        preview = reply.strip().replace("\n", " ")[:80]
        return CheckResult("AI endpoint", OK, f"{ai.model} -> {preview!r}")
    except AIError as exc:
        return CheckResult("AI endpoint", FAIL, str(exc))
    finally:
        await provider.aclose()


async def check_discord(token: str) -> CheckResult:
    if not token:
        return CheckResult("Discord token", FAIL, "DISCORD_TOKEN is not set")

    headers = {"Authorization": f"Bot {token}"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                "https://discord.com/api/v10/users/@me", headers=headers
            )
    except httpx.HTTPError as exc:
        return CheckResult("Discord token", FAIL, f"network error: {exc}")

    if response.status_code == 401:
        return CheckResult(
            "Discord token", FAIL, "invalid token (401) -- reset it in the dev portal"
        )
    if response.status_code >= 400:
        return CheckResult(
            "Discord token", FAIL, f"HTTP {response.status_code}: {response.text[:120]}"
        )

    try:
        data = response.json()
    except ValueError:
        return CheckResult("Discord token", FAIL, "unreadable response")

    name = data.get("username", "?")
    bot_id = data.get("id", "?")
    return CheckResult(
        "Discord token",
        OK,
        f"logged in as {name} (id={bot_id}) -- invite with "
        f"https://discord.com/oauth2/authorize?client_id={bot_id}"
        "&scope=bot+applications.commands&permissions=68608",
    )


async def run_checks(bot_config: BotConfig, ai_config: AIConfig) -> list[CheckResult]:
    results: list[CheckResult] = []

    problems = bot_config.validate()
    if problems:
        results.append(CheckResult("Bot config", FAIL, "; ".join(problems)))
    else:
        results.append(
            CheckResult(
                "Bot config",
                OK,
                f"trigger={bot_config.trigger_mode} prefix={bot_config.prefix!r} "
                f"memory={bot_config.memory_turns} turns",
            )
        )

    results.append(await check_ai(ai_config))
    results.append(await check_discord(bot_config.token))
    return results


def format_results(results: list[CheckResult]) -> str:
    width = max(len(r.name) for r in results) if results else 0
    lines = []
    for result in results:
        lines.append(f"[{result.status:4}] {result.name.ljust(width)}  {result.detail}")
    failures = [r for r in results if r.status == FAIL]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} problem(s) found -- fix the FAIL lines above.")
    else:
        lines.append("All checks passed. Start the bot with: python -m talkingtaco")
    return "\n".join(lines)


def main() -> int:
    from .config import load_ai_config, load_bot_config

    results = asyncio.run(run_checks(load_bot_config(), load_ai_config()))
    print(format_results(results))
    return 1 if any(r.status == FAIL for r in results) else 0
