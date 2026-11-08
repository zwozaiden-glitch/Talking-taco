# Talking Taco 🌮

An AI-powered Discord chat bot. Mention it or use a prefix, and it answers with
your own AI API — **Google Gemini** or any OpenAI-compatible endpoint. Swapping
backends never requires touching code.

```
You:    @Taco what's the best way to learn Python?
Taco:   Start with small scripts you actually want to exist...
```

## Features

- **Bring your own AI API** — Google Gemini (AI Studio keys) plus any
  OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq, Together, Ollama, LM Studio,
  vLLM, or your own server). Swap it with three env vars.
- **Automatic setup** — paste an `AIza...` key and the bot detects Gemini on its
  own; `AI_MODEL=auto` asks the API which models your key can use.
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
- **53 tests** covering memory, trigger detection, chunking, streaming, and the
  HTTP contract.

## Quick start

```bash
git clone <this-repo>
cd Talking-taco
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in DISCORD_TOKEN and AI_API_KEY
./check.sh                       # verify token + AI key
./run.sh                         # start the bot
```

Shortcuts: `./run.sh` bootstraps the venv and `.env` for you, and `./check.sh`
runs the preflight. Both accept an optional env file (`./run.sh .env.bot2`).

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

**Google Gemini** (keys start with `AIzaSy`, from
[AI Studio](https://aistudio.google.com/apikey)):

```env
AI_PROVIDER=gemini
AI_API_KEY=AIzaSy...
AI_MODEL=auto            # or gemini-2.5-flash, gemini-2.5-pro, ...
```

`AI_MODEL=auto` queries the API for the models your key can use and picks a fast
flash model, so you don't have to guess names. Gemini's "thinking" is off by
default (`AI_THINKING_BUDGET=0`) for snappy Discord replies; raise it if you want
the bot to reason harder before answering.

**Any OpenAI-compatible API:**

```env
AI_PROVIDER=openai
AI_BASE_URL=https://your-api.example.com/v1
AI_API_KEY=your-key-here
AI_MODEL=your-model-name
```

It must accept the standard OpenAI chat request:

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
| `AI_PROVIDER` | auto-detect | `gemini`, `openai` (compatible), or `mock` |
| `AI_API_KEY` | — | Your key. `AIza...` is auto-detected as Gemini |
| `AI_BASE_URL` | Gemini default | API root, e.g. `https://api.openai.com/v1` |
| `AI_MODEL` | `auto` | `auto`, `gemini-2.5-flash`, `gpt-4o-mini`, ... |
| `AI_THINKING_BUDGET` | `0` | Gemini 2.5+ reasoning tokens (0 = off, fastest) |
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
    gemini.py               Google Gemini / AI Studio (with model auto-detect)
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
| `Unknown AI_PROVIDER` | Must be `gemini`, `openai`, or `mock` (or one you registered) |
| `API key not valid` from Gemini | Wrong/expired key — make one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| Gemini `429` / quota exceeded | Free-tier limits hit — wait, or check your plan |
| Gemini model not found | Set `AI_MODEL=auto`, or run `--check` to list usable models |
| Gemini replies are slow | Set `AI_THINKING_BUDGET=0` and prefer a `flash` model |
| Slash commands missing | Wait a minute, or kick/re-invite with `applications.commands` scope |
| Every reply is a canned sentence | `AI_PROVIDER=mock` — set it to `openai` |
| Bot replies to everything | `REPLY_TO_ALL_MESSAGES=true` or it's in `allowed_channel_ids` |

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 53 tests
ruff check .
```

## Security notes

- `.env` is gitignored and never committed. Only `.env.example` (with empty
  values) is tracked.
- **If a token or key is ever pasted into a chat, ticket, screenshot, or commit,
  treat it as compromised** and rotate it: Discord Developer Portal → Bot →
  Reset Token, and AI Studio → delete/create a new API key. Rotating takes 30
  seconds and makes the old value useless.
- The bot only needs **Send Messages**, **Read Message History**, and
  **Embed Links**. Avoid granting Administrator.

## License

MIT
