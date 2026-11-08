"""Console entrypoint: ``python -m talkingtaco`` or the ``talkingtaco`` script."""

from __future__ import annotations

import asyncio
import logging
import sys

from aiohttp import ClientConnectorError as _ClientConnectionError
from discord import LoginFailure as _LoginFailure

from .bot import TacoBot
from .config import load_ai_config, load_bot_config
from .providers import AIError, create_provider


def _read_flag(argv: list[str], flag: str) -> str | None:
    """Return the value after ``flag`` from argv, if present."""
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _setup_logging()
    log = logging.getLogger("talkingtaco")

    if "--check" in argv or "--doctor" in argv:
        from .doctor import main as doctor_main

        return doctor_main()
    env_file = _read_flag(argv, "--env-file")
    if env_file:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=True)
        log.info("Loaded configuration from %s", env_file)

    if "--version" in argv:
        from . import __version__

        print(f"talkingtaco {__version__}")
        return 0
    if "--help" in argv or "-h" in argv:
        print(
            "Usage: python -m talkingtaco [options]\n\n"
            "  (no args)            start the bot using .env\n"
            "  --check              verify Discord token + AI endpoint, then exit\n"
            "  --env-file PATH      load configuration from PATH (run several bots!)\n"
            "  --version            print the version\n"
            "  --help               show this message\n\n"
            "Run several bots side by side:\n"
            "  python -m talkingtaco --env-file .env.taco\n"
            "  python -m talkingtaco --env-file .env.taco2\n"
        )
        return 0

    bot_config = load_bot_config()
    ai_config = load_ai_config()

    problems = bot_config.validate()
    if problems:
        for problem in problems:
            log.error("Config error: %s", problem)
        log.error("Copy .env.example to .env and fill in DISCORD_TOKEN.")
        return 1

    try:
        provider = create_provider(ai_config)
    except AIError as exc:
        log.error("%s", exc)
        return 1

    if provider.name != "mock":
        for problem in ai_config.validate():
            log.warning("AI config: %s", problem)

    bot = TacoBot(bot_config=bot_config, ai_config=ai_config, provider=provider)
    try:
        asyncio.run(_run(bot, bot_config.token, log))
    except KeyboardInterrupt:
        log.info("Shutting down.")
        return 0
    except _LoginFailure:
        log.error(
            "Discord rejected the token. Reset it in the Developer Portal "
            "(Bot -> Reset Token) and update DISCORD_TOKEN in .env."
        )
        return 2
    except (OSError, _ClientConnectionError) as exc:
        log.error(
            "Could not connect to discord.com (%s).\n"
            "  - Check your internet connection, DNS, or firewall/proxy.\n"
            "  - On a server, make sure outbound HTTPS (port 443) is allowed.\n"
            "  - Some sandboxes/CI environments block Discord entirely.",
            exc,
        )
        return 3
    except Exception:
        log.exception("Fatal error while running the bot")
        return 1
    return 0


async def _run(bot: TacoBot, token: str, log) -> None:
    try:
        await bot.start(token)
    finally:
        if not bot.is_closed():
            await bot.close()


if __name__ == "__main__":
    sys.exit(main())
