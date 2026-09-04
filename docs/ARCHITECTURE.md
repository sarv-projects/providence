# Architecture

Live graph: **A4** in `src/graph.py`. Overview: [../README.md](../README.md).

Writing and extraction use OpenCode Zen free (`fast` / `strong`). Thinking uses Gemini Flash. Search uses Exa when `EXA_API_KEY` is set. RAG is LanceDB + FTS5, isolated per `run_id`.

---

## Surfaces

- CLI — `uv run python main.py`
- Web — Next.js 14 (`frontend/`) + FastAPI (`src/web/`)
- Ops dashboard — `uv run python -m src.dashboard` (see [GATEWAY.md](GATEWAY.md))

Modes: `chat` · research depth `standard` · `deep` (+ legacy `quick`, `ultra-long`)
Lenses (combinable toggles): `recency` · `academic` · `compare`
(legacy `--mode recency|academic|compare` resolve to `standard` + their lens)
Quality dials (from `config/modes.yaml`): ultra-fast · balanced · accurate · comprehensive  
Autonomy: L1 (run) · L2 (approve plan) · L3 (unattended, tighter $)

Thinker hops after scout run only when the mode's dial has `thinker_enabled` (`deep`, `ultra-long`, legacy `academic` → deep). Scout always runs and uses Gemini if `GEMINI_API_KEY` is set.

---

## Pipeline (A4)

```
scout (Gemini ×3 + web)
  → planner (Zen)
  → thinker_plan_refine (Gemini, if enabled)
  → gather → analyze → contradiction (Gemini, if enabled) → critic → search_strategy ↺
  → devil_advocate
  → claim_adjudicator  [optional one Socratic re-gather]
  → triangulator (accurate / comprehensive dials)
  → synthesizer_outline → parallel section write (Zen strong)
  → compiler: Inference + Evidence Bedrock + Research Debt + Sources
```

Abort path: off-topic / no evidence → `abort_passthrough` → compiler (error note, no fake sources).

Entry: `src/graph.py` → `run_research()`.

---

## Agents

| Agent | File | LLM | Tools |
|-------|------|-----|-------|
| Scout / plan refine / contradiction / strategy | `engine/agents/thinker.py` | **Gemini only** | none (scout does a light web peek) |
| Planner | `engine/agents/planner.py` | fast (Zen) | none |
| Researcher gather / analyze | `engine/agents/researcher.py` | task/fast (Zen) | tool bus + RAG |
| Critic | `engine/agents/critic.py` | fast (Zen) | none |
| Devil’s advocate / adjudicator | `engine/agents/adversary.py` | thinker for debt notes | search |
| Triangulator | `engine/agents/triangulator.py` | fast + strong | none |
| Synthesizer | `engine/agents/synthesizer.py` | strong (Zen) | RAG retrieve |
| Compiler | `engine/agents/compiler.py` | rules + no LLM for Sources | export |

Retriever Guard is **not** an LLM agent — `src/rag/guard.py` scores domains and topicality.  
Factoids (`src/rag/factoid.py`) run only on the **comprehensive** dial (`ultra-long`). They do not replace raw chunks.

---

## Tool bus

`src/tools/registry.py` — capability tags, TTL search cache, parallel extract, optional `TOOL_FUSE_SEARCH=1`.

| Tool | When |
|------|------|
| Exa | `EXA_API_KEY` — primary neural search + full text |
| Firecrawl | cloud key or localhost:3002; else native scrape fallback |
| Wikipedia | always |
| GDELT | always (often 429 under concurrency) |
| NewsData | `NEWSDATA_API_KEY` |
| Tavily | `TAVILY_API_KEY` |
| Builtin scraper | always (DuckDuckGo + trafilatura) |
| MinerU / Nougat / LlamaParse | optional PDF adapters (PyPDF fallback) |

---

## RAG

`src/rag/` — ingest: chunk (parent-child when `RAG_PARENT_CHILD=1`) → embed → LanceDB + FTS5.  
Retrieve: dense + keyword RRF, filtered by `run_id`.  
Embed: OpenAI if `EMBEDDING_API_KEY` / `USE_CHAT_KEY_FOR_EMBEDDINGS=1`; else local bag-of-words.  
Vault: `data/vault.db` — reuse on-topic past sources.

---

## LLM gateway

Every `call_llm` goes through `src/gateway/` (circuits, retries, rate limits, failover).

Tiers in `config/providers.yaml`:

- `fast` / `strong` — Zen free first (`nemotron-3-ultra-free`, …)
- `thinker` — Gemini Flash IDs only

Empty `base_url` → `https://opencode.ai/zen/v1`. Empty key → no `Authorization` header.  
See [PROVIDERS.md](PROVIDERS.md).

---

## Ship-gate (compiler)

- Sources section last; URLs from this run only (`run_corpus` / search / extract)
- Drop writer “References” / “Sources & Bibliography” sections
- Remap inline `[n]` to the final Sources order; drop unmapped numbers
- Claim–evidence check on adjudicated claims
- Warn if many leftover `[n]` do not map to fetched URLs
- L3 can block export on remaining issues

---

## Layout

```
main.py                 CLI
config/modes.yaml
config/providers.yaml
src/graph.py            live LangGraph
src/state.py
src/llm.py              call_llm → gateway
src/engine/agents/      planner, researcher, thinker, critic, …
src/engine/modes.py     budgets + dials
src/gateway/            router, circuits, providers
src/tools/              registry + adapters
src/rag/                pipeline, hybrid, guard, vault, factoid
src/web/                FastAPI
src/dashboard/          ops UI
src/engine/temporal/    optional ultra-long (not a wrap of A4)
frontend/               Next.js
reports/                markdown + html
```

`src/nodes.py` is kept for older tests. The live pipeline is `src/graph.py` plus `src/engine/agents/`.

Temporal (`main.py worker`) is optional. If a cluster is up, `ultra-long` can run there; otherwise it runs in-process and will not survive a restart.

Agents return prompted JSON (fences stripped). Factoid extraction is on for `ultra-long` only. The thinker tier does not fall through to Zen or Groq.
