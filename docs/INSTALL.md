# Installation

Product overview: [../README.md](../README.md). Index: [INDEX.md](INDEX.md).

## Prerequisites

| Tool | Notes |
|------|--------|
| Python | 3.10+ |
| [uv](https://docs.astral.sh/uv/) | package runner |
| Git | clone |
| Node 18+ | optional, for the web UI |
| Keys | none required; see below |

## Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
git clone https://github.com/sarv-projects/providence.git
cd providence
bash scripts/install.sh
cp .env.example .env
# optional: GEMINI_API_KEY, EXA_API_KEY

uv run python main.py doctor
uv run python main.py research "your topic" --mode standard
```

UI:

```bash
bash scripts/start-dev.sh
# API :8001 · UI :3000
```

## Windows

```powershell
git clone https://github.com/sarv-projects/providence.git
cd providence
.\scripts\install.ps1
copy .env.example .env

uv run python main.py doctor
uv run python main.py research "your topic" --mode standard
```

## Environment

| Variable | Role |
|----------|------|
| *(none)* | Zen free workhorse + Wikipedia / builtin scrape |
| `GEMINI_API_KEY` | Thinker / scout |
| `EXA_API_KEY` | Primary search |
| `FIRECRAWL_API_KEY` / `TAVILY_API_KEY` / `NEWSDATA_API_KEY` | Extra retrieval |
| `EMBEDDING_API_KEY` or `USE_CHAT_KEY_FOR_EMBEDDINGS=1` | Better vectors |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | Not on the default tier list |

Full catalog: [PROVIDERS.md](PROVIDERS.md) · `config/providers.yaml`.

## Troubleshooting

| Issue | Check |
|-------|--------|
| No search results | Set `EXA_API_KEY` |
| Slow / 429 on Zen free | Wait, or let failover hit the next Zen ID |
| Weak scout / plan | Set `GEMINI_API_KEY` |
| Gemini 429 | Scout is 3 parallel calls; free RPM is tight |
| UI can’t reach API | Backend on :8001; Next rewrites `/api/*` |
