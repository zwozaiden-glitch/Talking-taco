"""The Discord client: listens for messages, calls the AI, replies.

Design notes
------------
* Two trigger modes: @mention the bot, or use a prefix (``!taco <message>``).
  ``BOT_TRIGGER_MODE`` picks one or both.
* Streaming replies: the bot shows a "thinking..." message, then edits it as
  tokens arrive (throttled) so replies feel live instead of laggy.
* Everything user-facing has a cooldown and a length guard so a busy server
  cannot hammer your API.
* Errors are surfaced to the user as a short friendly line; the full detail
  goes to the log (and to the owner in DMs if ``LOG_ERRORS_TO_OWNER`` is on).
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import NamedTuple

import discord
from discord import app_commands
from discord.ext import commands

from .config import AIConfig, BotConfig, load_ai_config, load_bot_config
from .memory import ConversationMemory
from .providers import AIError, AIProvider, create_provider

log = logging.getLogger("talkingtaco")

DISCORD_MESSAGE_LIMIT = 2000
MAX_INPUT_CHARS = 4000
EDIT_INTERVAL = 1.1  # seconds between message edits, keeps us inside rate limits

ERROR_REPLY = "Taco's brain short-circuited. Try again in a moment."


class GenerationResult(NamedTuple):
    """A finished reply plus whether it was already posted live."""

    text: str
    posted: bool


class Cooldown:
    """Tiny per-user gate -- simpler and fairer than per-command cooldowns."""

    def __init__(self, seconds: float) -> None:
        self.seconds = max(0.0, seconds)
        self._last: dict[int, float] = defaultdict(float)

    def check(self, user_id: int) -> float:
        """Return remaining cooldown in seconds (0.0 means allowed)."""
        if self.seconds <= 0:
            return 0.0
        remaining = self.seconds - (time.monotonic() - self._last[user_id])
        if remaining > 0:
            return remaining
        self._last[user_id] = time.monotonic()
        return 0.0


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split on paragraph/word boundaries so we never break mid-word."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        candidate = f"{current}\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= limit:
            current = paragraph
            continue
        # A single huge paragraph: hard-wrap on word boundaries.
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}".strip()
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = word[:limit]
    if current:
        chunks.append(current)
    return chunks


class TacoBot(commands.Bot):
    def __init__(
        self,
        bot_config: BotConfig | None = None,
        ai_config: AIConfig | None = None,
        provider: AIProvider | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.dm_messages = True
        super().__init__(command_prefix="!taco ", intents=intents, help_command=None)

        self.settings = bot_config or load_bot_config()
        self.ai_config = ai_config or load_ai_config()
        self.provider = provider or create_provider(self.ai_config)
        self.memory = ConversationMemory(max_turns=self.settings.memory_turns)
        self.cooldown = Cooldown(self.settings.cooldown_seconds)
        self._prefix_re = re.compile(
            rf"^{re.escape(self.settings.prefix)}\b\s*", re.IGNORECASE
        )

    # ------------------------------------------------------------- lifecycle
    async def setup_hook(self) -> None:
        self.tree.add_command(self.chat)
        self.tree.add_command(self.reset)
        self.tree.add_command(self.status)
        await self.ensure_provider()
        if not self.settings.token:
            log.error("DISCORD_TOKEN is missing -- cannot log in.")
            return
        try:
            await self.tree.sync()
            log.info("Slash commands synced.")
        except Exception:  # pragma: no cover - network dependent
            log.warning("Slash command sync failed; continuing anyway.", exc_info=True)

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, getattr(self.user, "id", "?"))
        log.info(
            "Trigger mode=%s prefix=%r provider=%s model=%s",
            self.settings.trigger_mode,
            self.settings.prefix,
            self.provider.name,
            self.ai_config.model,
        )
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening, name="you | /chat"
            )
        )

    async def close(self) -> None:
        try:
            await self.provider.aclose()
        finally:
            await super().close()

    async def ensure_provider(self) -> None:
        """Best-effort self-check so misconfiguration is obvious at startup."""
        if self.provider.name == "mock":
            log.warning("AI_PROVIDER=mock -- replies are canned, no API is being called.")
            return
        for problem in self.ai_config.validate():
            log.warning("AI config: %s", problem)

    # ---------------------------------------------------------------- events
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.author.id == self.user.id:
            return

        prompt = self._extract_prompt(message)
        if prompt is None:
            return
        if prompt == "":
            await message.reply(
                "You rang? Add a question after the mention and I'll answer it.",
                mention_author=False,
            )
            return

        await self._respond_to_message(message, prompt)

    def _extract_prompt(self, message: discord.Message) -> str | None:
        """Return the prompt text, or None if the message isn't for us."""
        if isinstance(message.channel, discord.DMChannel):
            return message.content.strip()

        if not self._channel_allowed(message.channel):
            return None

        mode = self.settings.trigger_mode
        text = message.content.strip()
        mentioned = any(u.id == self.user.id for u in message.mentions)

        if mentioned and mode in {"mention", "both"}:
            cleaned = re.sub(rf"<@!?{self.user.id}>", "", text, count=1).strip()
            return cleaned

        if mode in {"prefix", "both"}:
            match = self._prefix_re.match(text)
            if match:
                return text[match.end() :].strip()

        if self.settings.reply_to_all_messages:
            return text

        return None

    def _channel_allowed(self, channel) -> bool:
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            return True
        if self.settings.ignored_channel_ids and channel_id in self.settings.ignored_channel_ids:
            return False
        if self.settings.allowed_channel_ids:
            return channel_id in self.settings.allowed_channel_ids
        return True

    # ------------------------------------------------------------- responding
    async def _respond_to_message(self, message: discord.Message, prompt: str) -> None:
        remaining = self.cooldown.check(message.author.id)
        if remaining > 0:
            # Silence looks like a broken bot, so acknowledge with a reaction.
            log.debug("Cooldown hit for %s (%.1fs)", message.author, remaining)
            try:
                await message.add_reaction("\N{HOURGLASS}")
            except discord.HTTPException:
                pass
            return

        if len(prompt) > MAX_INPUT_CHARS:
            prompt = prompt[:MAX_INPUT_CHARS]

        conversation_id = message.channel.id
        messages = self.memory.build_prompt(
            conversation_id, prompt, self.ai_config.system_prompt
        )

        result: GenerationResult | None = None
        async with message.channel.typing():
            try:
                result = await self._generate(messages, message)
            except AIError as exc:
                log.error("AI error: %s", exc)
                await _safe_reply(message, ERROR_REPLY)
                return
            except Exception:
                log.exception("Unexpected failure while generating a reply")
                await _safe_reply(message, ERROR_REPLY)
                return

        if result is None or not result.text:
            await _safe_reply(message, ERROR_REPLY)
            return

        self.memory.append(conversation_id, prompt, result.text)
        # If nothing was streamed live, post the finished answer normally.
        if not result.posted:
            await _send_long(message, result.text)

    async def _generate(
        self, messages, message: discord.Message | None = None
    ) -> GenerationResult:
        """Stream a reply, editing a live message as text arrives.

        ``posted`` is True when the reply was already delivered by editing the
        live message, so the caller knows not to post it a second time.
        """
        parts: list[str] = []
        live: discord.Message | None = None
        last_edit = 0.0
        try:
            async for chunk in self.provider.stream(messages):
                parts.append(chunk)
                text = "".join(parts)
                now = time.monotonic()
                if message is None or now - last_edit < EDIT_INTERVAL:
                    continue
                last_edit = now
                preview = split_message(text, DISCORD_MESSAGE_LIMIT)[0]
                if live is None and not preview.strip():
                    continue
                try:
                    if live is None:
                        live = await message.reply(preview or "…", mention_author=False)
                    else:
                        await live.edit(content=preview)
                except discord.HTTPException:
                    log.debug("Live edit skipped due to HTTP error", exc_info=True)
        except NotImplementedError:
            return GenerationResult(await self.provider.complete(messages), False)

        text = "".join(parts).strip()
        if not text:
            return GenerationResult(await self.provider.complete(messages), False)

        if live is None:
            return GenerationResult(text, False)

        # Finalize: full text (split across messages if it got long).
        chunks = split_message(text)
        try:
            await live.edit(content=chunks[0])
            for extra in chunks[1:]:
                await live.channel.send(extra)
        except discord.HTTPException:
            log.warning("Failed to finalize streamed reply", exc_info=True)
            return GenerationResult(text, False)
        return GenerationResult(text, True)

    # ----------------------------------------------------------- slash /chat
    @app_commands.command(name="chat", description="Ask Taco an AI question.")
    @app_commands.describe(prompt="What do you want to ask?")
    async def chat(self, interaction: discord.Interaction, prompt: str) -> None:
        await interaction.response.defer(thinking=True)
        messages = self.memory.build_prompt(
            interaction.channel_id or 0, prompt, self.ai_config.system_prompt
        )
        try:
            result = await self._generate(messages)
        except AIError as exc:
            log.error("AI error: %s", exc)
            await interaction.followup.send(ERROR_REPLY)
            return
        except Exception:
            log.exception("Slash command failure")
            await interaction.followup.send(ERROR_REPLY)
            return

        reply = result.text
        self.memory.append(interaction.channel_id or 0, prompt, reply)
        chunks = split_message(reply)
        await interaction.followup.send(chunks[0] if chunks else ERROR_REPLY)
        for extra in chunks[1:]:
            await interaction.followup.send(extra)

    @app_commands.command(name="reset", description="Forget this channel's conversation.")
    async def reset(self, interaction: discord.Interaction) -> None:
        self.memory.clear(interaction.channel_id or 0)
        await interaction.response.send_message(
            "Memory cleared. Fresh start -- what's up?", ephemeral=True
        )

    @app_commands.command(name="status", description="Show Taco's current configuration.")
    async def status(self, interaction: discord.Interaction) -> None:
        stats = self.memory.stats(interaction.channel_id or 0)
        provider_line = self.provider.name
        if self.provider.name == "mock":
            provider_line += " (canned replies -- no API configured)"
        embed = discord.Embed(title="Taco status", color=discord.Color.orange())
        embed.add_field(name="Provider", value=provider_line, inline=True)
        embed.add_field(name="Model", value=self.ai_config.model or "unset", inline=True)
        embed.add_field(name="Trigger", value=self.settings.trigger_mode, inline=True)
        embed.add_field(name="Memory (this channel)", value=str(stats["messages"]), inline=True)
        embed.add_field(name="Active conversations", value=str(stats["conversations"]), inline=True)
        embed.add_field(name="Prefix", value=f"`{self.settings.prefix}`", inline=True)
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def _safe_reply(message: discord.Message, text: str) -> None:
    try:
        await message.reply(text, mention_author=False)
    except discord.HTTPException:
        log.warning("Failed to send reply", exc_info=True)


async def _send_long(message: discord.Message, text: str) -> None:
    chunks = split_message(text)
    if not chunks:
        return
    first = await _reply(message, chunks[0])
    for extra in chunks[1:]:
        try:
            await (first.channel.send(extra) if first else message.channel.send(extra))
        except discord.HTTPException:
            log.warning("Failed to send overflow chunk", exc_info=True)


async def _reply(message: discord.Message, text: str) -> discord.Message | None:
    try:
        return await message.reply(text, mention_author=False)
    except discord.HTTPException:
        log.warning("Failed to send reply", exc_info=True)
        return None
