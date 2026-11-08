# Talking Taco 🌮

An AI-powered Discord chat bot. Mention it or use a prefix, and it answers with
your own AI API. Built so you can **plug in your API later** without touching a
single line of code.

```
You:    @Taco what's the best way to learn Python?
Taco:   Start with small scripts you actually want to exist...
```

## Features

- **Bring your own AI API** — any OpenAI-compatible endpoint (OpenAI, OpenRouter,
  Groq, Together, Ollama, LM Studio, vLLM, or your own server). Swap it with three
  env vars.
- **Live streaming replies** — the bot posts a message and edits it as tokens
  arrive, so replies feel instant instead of frozen.
- **Per-channel memory** — remembers the last N exchanges in each channel,
  isolated from every other channel. Bounded by turn count *and* character budget.
- **Two trigger modes** — `@mention`, `!taco <message>` prefix, or both.
- **Slash commands** — `/chat`, `/reset`, `/status`.
- **Safe by default** — per-user cooldown, channel allow/ignore lists, message
  splitting at Discord's 2000-char limit, friendly error messages with full
  detail in logs.
- **Runs offline** — `AI_PROVIDER=mock` gives canned replies so you can test the
  Discord wiring before your API exists.
- **Preflight doctor** — `--check` validates your token and API endpoint.
- **33 tests** covering memory, trigger detection, chunking, streaming, and the
  HTTP contract.

## Quick start

```bash
git clone <this-repo>
cd Talking-taco
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill it in
python -m talkingtaco --check    # verify everything
python -m talkingtaco            # run the bot
```

### 1. Create the Discord application

1. Open the [Discord Developer Portal](https://discord.com/developers/applications)
   → **New Application**.
2. **Bot** tab → **Reset Token** → copy it into `DISCORD_TOKEN` in `.env`.
   Treat this like a password. Never commit `.env` (it's gitignored).
3. **Bot** tab → enable **Message Content Intent**. Without it the bot sees empty
   message text and will never reply.
4. **OAuth2 → URL Generator**: scopes `bot` + `applications.commands`;
   permissions **Send Messages**, **Read Message History**, **Embed Links**.
   Open the generated URL to invite the bot.

Running `python -m talkingtaco --check` prints your exact invite link once the
token is valid.

### 2. Point it at your API

You said you'll provide your API later — so for now leave it on mock mode and
everything still works:

```env
AI_PROVIDER=mock
```

When your API is ready, change **three lines** in `.env` and restart:

```env
AI_PROVIDER=openai
AI_BASE_URL=https://your-api.example.com/v1
AI_API_KEY=your-key-here
AI_MODEL=your-model-name
```

Your endpoint must accept the standard OpenAI chat request:

```http
POST {AI_BASE_URL}/chat/completions
Authorization: Bearer {AI_API_KEY}
Content-Type: application/json

{"model": "...", "messages": [{"role": "user", "content": "hi"}], "stream": true}
```

and answer with `choices[0].message.content` (or SSE deltas when `stream` is
true). If your API uses a different shape, see
[Adding a custom provider](#adding-a-custom-provider).

### 3. Test without Discord

Want to check the AI side only? There's a local fake API:

```bash
python scripts/fake_api_server.py        # terminal 1
AI_BASE_URL=http://127.0.0.1:8000/v1 AI_MODEL=fake AI_API_KEY=x \
  python -m talkingtaco --check          # terminal 2
```

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `DISCORD_TOKEN` | — | **Required.** Bot token from the dev portal |
| `BOT_TRIGGER_MODE` | `both` | `mention`, `prefix`, or `both` |
| `BOT_PREFIX` | `!taco` | Prefix used in prefix mode |
| `REPLY_TO_ALL_MESSAGES` | `false` | Answer every message in allowed channels |
| `ALLOWED_CHANNEL_IDS` | all | Comma-separated allowlist of channel IDs |
| `IGNORED_CHANNEL_IDS` | none | Comma-separated blocklist of channel IDs |
| `MEMORY_TURNS` | `12` | Past exchanges remembered per channel (0 = amnesia) |
| `COOLDOWN_SECONDS` | `3` | Minimum gap between requests from one user |
| `AI_BASE_URL` | — | Your API root, e.g. `https://api.openai.com/v1` |
| `AI_API_KEY` | — | Bearer token for your API |
| `AI_MODEL` | `gpt-4o-mini` | Model name your endpoint expects |
| `AI_PROVIDER` | `openai` | `openai` (compatible) or `mock` |
| `AI_MAX_TOKENS` | `600` | Max reply length |
| `AI_TEMPERATURE` | `0.8` | Creativity, 0–2 |
| `AI_TIMEOUT` | `60` | Request timeout in seconds |
| `AI_SYSTEM_PROMPT` | built-in | Override the personality |

## Commands

| Command | What it does |
|---|---|
| `@Taco <question>` | Chat with the bot |
| `!taco <question>` | Same, via prefix |
| `/chat prompt:<text>` | Slash command version |
| `/reset` | Forget this channel's conversation |
| `/status` | Show provider, model, and memory stats |

## Running several bots

Each bot is one env file. This is the easiest way to run a whole fleet:

```bash
python -m talkingtaco --env-file .env.taco &
python -m talkingtaco --env-file .env.taco2 &
```

Give each bot its own Discord application/token, and its own `AI_SYSTEM_PROMPT`
so they have distinct personalities.

## Deploying

**Docker**

```bash
docker build -t talkingtaco .
docker run --env-file .env --restart unless-stopped --name taco talkingtaco
```

**systemd** (on a VPS)

```ini
[Unit]
Description=Talking Taco bot
After=network-online.target

[Service]
WorkingDirectory=/opt/Talking-taco
ExecStart=/opt/Talking-taco/.venv/bin/python -m talkingtaco
Restart=always
RestartSec=5
User=taco

[Install]
WantedBy=multi-user.target
```

**Free hosts:** Railway, Render, or Fly.io all run this fine as a worker
process (no inbound port needed).

## Adding a custom provider

If your API isn't OpenAI-shaped, implement the interface — the bot needs only
`complete`, with `stream` optional:

```python
# talkingtaco/providers/my_api.py
from ..providers.base import AIProvider, AIError, ChatMessage

class MyAPI(AIProvider):
    name = "myapi"

    async def complete(self, messages: list[ChatMessage]) -> str:
        # call your API, return the reply text
        return "hello!"

# register it in providers/__init__.py
from .my_api import MyAPI
_REGISTRY["myapi"] = lambda config: MyAPI()
```

Then set `AI_PROVIDER=myapi`. Nothing else changes.

## Project layout

```
talkingtaco/
  bot.py                    Discord client, triggers, streaming, chunking
  config.py                 env-driven settings + validation
  memory.py                 per-channel bounded conversation windows
  doctor.py                 --check preflight (token + API + config)
  __main__.py               CLI entrypoint
  providers/
    base.py                 AIProvider interface
    openai_compatible.py    any OpenAI-style API (HTTP + SSE)
    mock.py                 offline canned replies
    __init__.py             provider registry
scripts/fake_api_server.py  local OpenAI-compatible server for testing
tests/                      unit + fake-Discord integration tests
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Logs in but never replies | **Message Content Intent** is off in the dev portal |
| `401 Unauthorized` from Discord | Token is wrong or was reset — regenerate it |
| Replies with "short-circuited" | Check logs; usually bad `AI_BASE_URL` / `AI_API_KEY` / model name |
| `Unknown AI_PROVIDER` | Must be `openai` or `mock` (or one you registered) |
| Slash commands missing | Wait a minute, or kick/re-invite with `applications.commands` scope |
| Every reply is a canned sentence | `AI_PROVIDER=mock` — set it to `openai` |
| Bot replies to everything | `REPLY_TO_ALL_MESSAGES=true` or it's in `allowed_channel_ids` |

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 33 tests
ruff check .
```

## License

MIT
