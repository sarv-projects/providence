"""
Researcher agent — executes the research loop: search, extract, ingest, retrieve, analyze.

Uses the modular tool bus for search and extraction. Tools are auto-discovered
import logging
from the registry: Firecrawl (primary), Wikipedia (free), Built-in Scraper, Exa.
"""

import hashlib
import json
import re
import time

from src.llm import call_llm
from src.jsonutil import parse_json_dict, parse_json_list
from src.tools import execute_searches, extract_pages as tool_extract
from src.tools.registry import get_registry
from src.urlutil import canonical_url
from src.rag.pipeline import ingest_documents, retrieve_chunks
from src.rag.factoid import extract_from_pages, token_reduction_stats
from src.rag.guard import filter_results, retry_pyramid_filter
from src.rag.hybrid import hybrid_retrieve
from src.rag.vault import Vault
from src.state import ResearchState
from .registry import register

RESEARCHER_SYSTEM = (
    "You are a thorough research analyst. Extract factual claims from sources, "
    "identify supporting evidence, and flag contradictions. Return valid JSON."
)


def _progress(stage: str, status: str = "", **kwargs) -> None:
    try:
        from src.engine.progress import get_progress
        get_progress().update(stage=stage, status=status or stage, **kwargs)
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)


@register("researcher_gather")
def researcher_gather(state: ResearchState) -> ResearchState:
    """Search the web and extract page content."""
    from src.engine.budget import (
        budget_status_line,
        check_budgets,
        force_complete,
        record_tool_calls,
        sync_cost_from_metrics,
    )

    state["iteration"] = int(state.get("iteration") or 0) + 1
    sync_cost_from_metrics(state)
    ok, reason = check_budgets(state)
    if not ok:
        print(f"\n🔍 [Researcher] Budget stop: {reason}")
        return force_complete(state, reason)

    # ── Fruitless-action gate (Jina node-DeepResearch) ──
    # If the last search/visit yielded zero new URLs/content, temporarily
    # disable that action so the loop cannot spin on the same fruitless path.
    fruitless = dict(state.get("fruitless") or {})
    search_disabled = bool(fruitless.get("search_disabled"))
    visit_disabled = bool(fruitless.get("visit_disabled"))

    quality = state.get("quality") or {}
    flags = state.get("mode_flags") or {}
    max_results = int(quality.get("max_search_results") or 10)
    max_extract = int(quality.get("max_extract_pages") or 5)
    factoid_on = bool(quality.get("factoid_enabled", False))
    import os

    # Exa: modest boost, not 200-page firehoses (that was killing wall time)
    # Quality = top-N Exa with full text, not volume for volume's sake
    if os.getenv("EXA_API_KEY"):
        if state.get("mode") == "quick":
            max_results = min(max(max_results, 8), 10)
            max_extract = min(max(max_extract, 6), 8)
        elif state.get("mode") in ("deep", "academic", "ultra-long"):
            max_results = min(max(max_results, 10), 12)
            max_extract = min(max(max_extract, 10), 12)
        else:
            max_results = min(max(max_results, 8), 12)
            max_extract = min(max(max_extract, 8), 10)

    state["status"] = "Searching and extracting content..."
    _progress("researching", state["status"], iteration=state["iteration"])
    print(f"\n🔍 [Researcher] Iteration {state['iteration']} — gathering "
          f"(max_results={max_results}, extract={max_extract})")

    queries = list(state.get("search_queries") or [state["query"]])
    if not queries:
        queries = [state["query"]]

    # Cap query fan-out: each query = 1 Exa call; 8 queries × 22 = disaster
    queries = queries[:4]

    # Mode bias: rewrite queries for recency / academic focus (don't explode count)
    if flags.get("recency_bias"):
        queries = [f"{q} 2024 OR 2025 OR 2026 latest" for q in queries]
    if flags.get("academic_bias") or flags.get("force_arxiv"):
        # One arXiv-biased query only (not one per query)
        base = state.get("query") or queries[0]
        arxiv_q = f"{base[:100]} site:arxiv.org survey"
        if arxiv_q not in queries:
            queries = queries[:3] + [arxiv_q]

    # ── Vault reuse: seed results from past high-quality sources ──
    # Only keep vault hits that share topic keywords with the query (anti-contamination)
    vault_hits: list[dict] = []
    if flags.get("vault_rag", True):
        try:
            import re as _re
            q_words = {
                w for w in _re.findall(r"[a-zA-Z]{4,}", (state.get("query") or "").lower())
                if w not in {
                    "does", "what", "with", "from", "that", "this", "into", "about",
                    "cover", "methods", "best", "practices", "large", "language",
                }
            }
            vault = Vault()
            for q in queries[:3]:
                for hit in vault.search(q, k=min(5, max_results)):
                    blob = f"{hit.get('title','')} {hit.get('snippet','')} {hit.get('url','')}".lower()
                    if q_words and sum(1 for w in q_words if w in blob) < min(2, len(q_words)):
                        continue  # off-topic vault residue
                    vault_hits.append({
                        "title": hit.get("title", ""),
                        "url": hit.get("url", ""),
                        "content": hit.get("snippet", ""),
                        "raw_content": hit.get("snippet", ""),
                        "score": float(hit.get("quality_score", 5)) / 10.0,
                        "source": "vault",
                        "guard_score": float(hit.get("quality_score", 5)),
                    })
            if vault_hits:
                print(f"  Vault reuse: {len(vault_hits)} on-topic cached sources")
        except Exception:
            vault_hits = []

    registry = get_registry()
    available = [t.name for t in registry.list_all()]
    print(f"  Tools available: {available}")

    # Live web search (tool bus) — skipped when the fruitless gate fired
    results: list[dict] = []
    if search_disabled:
        print("  🚫 Fruitless gate: live search disabled (previous round added nothing new)")
        results = list(vault_hits)
    else:
        # Search-mode routing (WebSwarm): scale per-query breadth by mode so an
        # entity-collect query pulls a wide net while an atom query stays tight.
        modes = state.get("search_modes") or {}
        buckets: dict[str, list[str]] = {}
        for q in queries:
            m = modes.get(q) or modes.get(q.lower()) or "deep"
            buckets.setdefault(m, []).append(q)
        for mode, qs in buckets.items():
            k = max_results
            if mode == "atom":
                k = max(3, min(k, 5))
            elif mode == "wide":
                k = min(k + 4, 14)
            elif mode == "entity_collect":
                k = min(k + 6, 16)
            elif mode == "web_structure":
                k = max(4, min(k, 8))
            print(f"  Mode {mode}: {len(qs)} queries @ max_results={k}")
            results.extend(execute_searches(qs, max_results=k))
        record_tool_calls(state, n=len(queries), kind="search")

        # ── Newswire pass (GDELT always + NewsData when keyed) ──
        # Tier-1 press (FT/Reuters/Caixin…) is the rubric's "source diversity"
        # category and generic web search rarely surfaces it. GDELT needs no
        # key and has no quota — always runs; NewsData supplements when the
        # env key is present. Both are cheap (snippets, no full-page fetch),
        # so this never blocks the main search chain.
        #
        # One hit-per-run guard: GDELT throttles hard under concurrency
        # (45s cross-process cooldown inside the adapter). Once we already
        # have newswire hits from an earlier iteration, stop calling — there
        # is no need to re-fetch the wire every loop, and it keeps concurrent
        # benchmark processes from 429-storming the free API.
        news_done = bool(state.get("news_hits"))
        if not news_done:
            try:
                news_hits: list[dict] = []
                from concurrent.futures import ThreadPoolExecutor
                from src.tools.adapters.gdelt import gdelt_search
                from src.tools.adapters.newsdata import newsdata_search
                news_q = state.get("query", "")[:200]
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_gdelt = ex.submit(gdelt_search, news_q, 8)
                    f_news = ex.submit(newsdata_search, news_q, 6)
                    try:
                        news_hits.extend(f_gdelt.result(timeout=25))
                    except Exception as e:
                        print(f"  [gdelt] timed out: {e}")
                    try:
                        news_hits.extend(f_news.result(timeout=20))
                    except Exception as e:
                        print(f"  [newsdata] timed out: {e}")
                if news_hits:
                    state["news_hits"] = news_hits  # once-per-run: stop re-calling
                    results.extend(news_hits)
                    print(f"  📰 Newswire: +{len(news_hits)} hits "
                          f"({sum(1 for r in news_hits if r.get('source') == 'gdelt')} gdelt, "
                          f"{sum(1 for r in news_hits if r.get('source') == 'newsdata')} newsdata)")
                    record_tool_calls(state, n=1, kind="search")
                else:
                    print("  📰 Newswire: no hits this round (cooldown/rate-limit) — will retry")
            except Exception as e:
                print(f"  Newswire pass skipped: {e}")
        else:
            print(f"  📰 Newswire: already fetched {len(state.get('news_hits') or [])} hits — skipping")


    # Deep/academic: one Exa arXiv pass only (skip slow/flaky mineru when Exa works)
    if flags.get("academic_bias") or flags.get("force_arxiv"):
        if os.getenv("EXA_API_KEY"):
            try:
                from src.tools.adapters.exa import exa_search
                aq = f"{state['query'][:120]} site:arxiv.org"
                extra = exa_search(aq, max_results=min(6, max_results))
                if extra:
                    print(f"  Exa arXiv: +{len(extra)} hits")
                    results = list(results) + extra
                    record_tool_calls(state, n=1)
            except Exception as e:
                print(f"  Exa arXiv skipped: {e}")
        else:
            try:
                from src.tools.adapters.mineru import mineru_search
                arxiv = mineru_search(state["query"], max_results=min(5, max_results))
                if arxiv:
                    print(f"  Academic: +{len(arxiv)} arXiv hits")
                    results = list(results) + arxiv
                    record_tool_calls(state, n=1)
            except Exception as e:
                print(f"  Academic arXiv search skipped: {e}")

    # Hard cap total search hits kept (Exa can dump 100+ across queries).
    # Newswire hits are exempt: GDELT/NewsData snippets carry a low raw
    # `score` (0.85) that would lose the sort against Exa's full-text hits
    # and get culled before the guard even runs — dropping the very source
    # category the newswire pass exists for. Keep them, cap the rest.
    hard_cap = max(max_results * 2, 20)
    if len(results) > hard_cap:
        news, rest = [], []
        for r in results:
            (news if r.get("source") in ("gdelt", "newsdata") else rest).append(r)
        rest = sorted(rest, key=lambda r: float(r.get("score") or 0), reverse=True)
        results = news + rest[:max(hard_cap - len(news), 0)]
        print(f"  Capped search pool to {len(results)} "
              f"(kept {len(news)} newswire hits, top {max(hard_cap - len(news), 0)} by score)")

    # Merge vault + live (URL dedup, live wins on conflict for freshness)
    seen_urls: set[str] = set()
    merged: list[dict] = []
    for r in results + vault_hits:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            merged.append(r)
    results = merged

    # ── Retriever Guard ──
    if results:
        print(f"  Found {len(results)} raw results via {results[0].get('source', 'unknown')}")
        before_count = len(results)
        # P0.4: pass the query so the guard can block real-but-irrelevant hits
        results, guard_stats = filter_results(results, min_score=3.0, topic=state.get("query", ""))
        state["guard_stats"] = guard_stats
        print(f"  Guard: {before_count} → {len(results)} passed"
              f" ({guard_stats['blocked']} blocked, avg score {guard_stats['avg_score']})")
        if guard_stats.get("off_topic_blocked"):
            print(f"    Off-topic blocked: {guard_stats['off_topic_blocked']}")
        if guard_stats.get("domains", {}).get("blocked"):
            print(f"    Blocked: {', '.join(guard_stats['domains']['blocked'])}")
    else:
        print("  ⚠️  No search results from any tool")
        state["guard_stats"] = {"total": 0, "passed": 0, "blocked": 0, "avg_score": 0}

    # Store live results in vault for future reuse
    if results:
        try:
            Vault().store_results(
                [r for r in results if r.get("source") != "vault"],
                queries=queries,
            )
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)

    state["search_results"] = results
    for r in results[:3]:
        print(f"    • [{r.get('guard_score', '?')}] {r.get('title', '')[:60]}")

    pages_scanned = len(results)
    _progress(
        "researching",
        f"Found {len(results)} sources",
        iteration=state["iteration"],
        pages_scanned=pages_scanned,
        sources_count=len(results),
    )
    try:
        from src.engine.progress import get_progress
        get_progress().think("next", f"Extracting up to {max_extract} pages")
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)

    # Prefer Exa full text already in results — no re-extract round-trip
    top = results[:max_extract]
    already_full = {
        r["url"] for r in top
        if r.get("url") and len(r.get("raw_content") or "") > 800
    }
    need_extract = [r["url"] for r in top if r.get("url") and r["url"] not in already_full]
    if visit_disabled:
        print("  🚫 Fruitless gate: extraction disabled (previous extraction added nothing new)")
        need_extract = []
    extracted = tool_extract(need_extract) if need_extract else []
    for r in top:
        if r.get("url") in already_full:
            extracted.append({
                "url": r["url"],
                "content": r.get("raw_content") or r.get("content", ""),
                "title": r.get("title", ""),
                "source": r.get("source", "exa"),
            })
    record_tool_calls(state, n=1 if need_extract else 0, kind="extract")
    state["extracted_pages"] = extracted

    # ── Fetched-source ledger (citation ship-gate input) ──
    # Only pages whose content was ACTUALLY retrieved become ledger entries.
    # Search hits that were never opened are deliberately NOT recorded, so
    # the compiler's Sources list cannot cite unverified URLs.
    ledger = dict(state.get("fetched_sources") or {})

    def _ledger_add(url: str, title: str, content: str) -> None:
        u = canonical_url(url or "")
        if not u:
            return
        prev = ledger.get(u) if isinstance(ledger.get(u), dict) else {}
        ledger[u] = {
            "url": u,
            "title": title or prev.get("title") or u,
            "status": "fetched",
            "content_hash": hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:16],
            "chars": len(content or ""),
            "fetched_at": time.time(),
        }

    for p in extracted:
        _ledger_add(p.get("url"), p.get("title"), p.get("content"))
    for c in state.get("run_corpus") or []:
        _ledger_add(c.get("url"), c.get("title") or c.get("url"), c.get("text") or c.get("content"))
    for c in state.get("retrieved_chunks") or []:
        _ledger_add(c.get("url"), c.get("title"), c.get("text") or c.get("content"))
    state["fetched_sources"] = ledger
    print(f"  Content pages: {len(extracted)} ({len(already_full)} from Exa text, "
          f"{len(need_extract)} extracted), ledger={len(ledger)}")

    # Full-text corpus (Tier-2 #13): pages actually fetched this run become
    # searchable vault entries — "research once, search web".
    # (Runs AFTER extraction — `extracted` must be populated first.)
    if extracted:
        try:
            Vault().store_pages(extracted, queries=queries)
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)

    # ── Fruitless gate bookkeeping ──
    # Search: did this round surface any URL we haven't already consulted?
    consulted = set((state.get("research_memory") or {}).get("consulted_sources") or [])
    new_search_urls = [
        canonical_url(r.get("url") or "")
        for r in results
        if canonical_url(r.get("url") or "") and canonical_url(r.get("url") or "") not in consulted
    ]
    if search_disabled:
        pass  # keep disabled until a round with new content clears it
    elif new_search_urls:
        fruitless["search_disabled"] = False
        fruitless["search_streak"] = 0  # reset so re-disable needs 2 bad rounds again
    else:
        streak = int(fruitless.get("search_streak") or 0) + 1
        fruitless["search_streak"] = streak
        if streak >= 2:
            fruitless["search_disabled"] = True
            print(f"  🚫 Fruitless gate: {streak} rounds with no new URLs — disabling live search")

    new_extract_urls = [
        canonical_url(p.get("url") or "")
        for p in extracted
        if canonical_url(p.get("url") or "") and canonical_url(p.get("url") or "") not in consulted
    ]
    if visit_disabled:
        pass
    elif new_extract_urls:
        fruitless["visit_disabled"] = False
        fruitless["visit_streak"] = 0  # reset so re-disable needs 2 bad rounds again
    else:
        vstreak = int(fruitless.get("visit_streak") or 0) + 1
        fruitless["visit_streak"] = vstreak
        if vstreak >= 2:
            fruitless["visit_disabled"] = True
            print(f"  🚫 Fruitless gate: {vstreak} rounds with no new content — disabling extraction")

    state["fruitless"] = fruitless
    _progress("researching", f"Pages ready: {len(extracted)}",
              pages_scanned=max(pages_scanned, len(extracted)),
              iteration=state["iteration"])

    # Ingest only top pages (not entire 100+ result dump)
    pages = []
    seen_page_urls: set[str] = set()
    for r in top:
        raw = r.get("raw_content", "") or r.get("content", "")
        url = r.get("url", "")
        if raw and url not in seen_page_urls:
            seen_page_urls.add(url)
            pages.append({
                "url": url, "title": r.get("title", ""),
                "content": raw[:12000], "source_type": "web",
            })
    for p in extracted:
        content = p.get("content", "")
        url = p.get("url", "")
        if content and url not in seen_page_urls:
            seen_page_urls.add(url)
            pages.append({
                "url": url, "title": p.get("title", ""),
                "content": content[:12000], "source_type": "web_extracted",
            })

    if pages:
        # Factoids: skip after iter 1 and keep tiny (each call is a free-model stall)
        if not factoid_on or int(state.get("iteration") or 1) > 1:
            if not factoid_on:
                print("  Skipping factoid extraction (disabled by quality dial)")
            else:
                print("  Skipping factoids after iter 1 (speed)")
            all_factoids = []
        else:
            sample = pages[:3]
            print(f"  Extracting factoids from {len(sample)} pages (speed-capped)...")
            all_factoids = extract_from_pages(sample, max_pages=3, max_llm_calls=1)

        from src.rag.factoid import deduplicate_factoids
        existing_factoids = state.get("factoids") or []
        combined = deduplicate_factoids(list(existing_factoids) + all_factoids)
        state["factoids"] = combined
        state["factoid_stats"] = token_reduction_stats(pages, all_factoids)
        stats = state["factoid_stats"]
        print(f"  Factoids: {len(all_factoids)} new → {len(combined)} total, "
              f"{stats['factoid_tokens']} tokens ({stats['reduction_pct']:.0f}% reduction"
              f" vs {stats['raw_tokens']} raw)")
        _progress("researching", f"Factoids={len(combined)}",
                  factoids_count=len(combined), iteration=state["iteration"])

        ingested = ingest_documents(pages, run_id=state.get("run_id", "default"))
        if all_factoids:
            factoid_pages = [
                {
                    "url": (f.get("source_urls") or [f.get("source_url", "factoid://")])[0]
                           or f"factoid://{f.get('id', '')}",
                    "title": f"[Factoid: {f.get('type', '')}] {f.get('value', '')[:80]}",
                    "content": f.get("value", ""),
                    "source_type": "factoid",
                }
                for f in all_factoids
            ]
            # Same run_id so hybrid retrieve isolation still sees factoids
            ingested += ingest_documents(
                factoid_pages, run_id=state.get("run_id", "default")
            )
        state["chunks_ingested"] = int(state.get("chunks_ingested") or 0) + ingested
        print(f"  Ingested {ingested} chunks (total: {state['chunks_ingested']})")
        state["clean_content"] = [p["content"][:500] for p in pages[:5] if p.get("content")]
    else:
        state["status"] = "No content found"
        print("  ⚠️  No content to ingest")

    sync_cost_from_metrics(state)
    return state


@register("researcher_analyze")
def researcher_analyze(state: ResearchState) -> ResearchState:
    """Retrieve from RAG and extract claims from the retrieved chunks."""
    state["status"] = "Retrieving and analyzing..."
    print(f"\n🔍 [Researcher] Analyzing")

    # Retrieve from RAG
    query = state["query"]
    if state.get("findings"):
        query = query + " " + " ".join(state["findings"][-3:])
    if state.get("gaps"):
        query = query + " " + " ".join(state["gaps"][-3:])

    # ── Hybrid Retrieval (Phase H): dense + sparse + factoid fusion ──
    # P0.1: always filter by run_id to prevent cross-run contamination
    factoids = state.get("factoids", [])
    run_id = state.get("run_id", "")
    results = hybrid_retrieve(query, k=12, factoids=factoids, run_id=run_id)
    state["retrieved_chunks"] = results

    # Accumulate run-wide corpus so later adjudication can verify claims
    # extracted in EARLIER iterations too (chunks are overwritten each loop).
    # Dedup by canonical URL so html/pdf/abs variants of one paper count once.
    run_corpus = list(state.get("run_corpus") or [])
    seen_chunk = {canonical_url(c.get("url") or "") for c in run_corpus}
    for r in results:
        u = canonical_url(r.get("url") or "")
        if u and u not in seen_chunk:
            seen_chunk.add(u)
            run_corpus.append(r)
    state["run_corpus"] = run_corpus

    # Keep the fetched-source ledger in sync with newly retrieved chunks
    # (retrieval happens every iteration; the gather-side ledger merge above
    # only saw the previous iteration's corpus).
    ledger = dict(state.get("fetched_sources") or {})
    for r in results:
        u = canonical_url(r.get("url") or "")
        if not u:
            continue
        prev = ledger.get(u) if isinstance(ledger.get(u), dict) else {}
        ledger[u] = {
            "url": u,
            "title": r.get("title") or prev.get("title") or u,
            "status": "fetched",
            "content_hash": hashlib.sha256(
                (r.get("text") or "").encode("utf-8")
            ).hexdigest()[:16],
            "chars": len(r.get("text") or ""),
            "fetched_at": time.time(),
        }
    state["fetched_sources"] = ledger

    if results:
        retrieved_tokens = sum(len(r.get("text", "").split()) * 1.3 for r in results)
        raw_est = sum(len(p.get("content", "").split()) * 1.3 for p in state.get("extracted_pages", []))
        raw_est += sum(len((r2.get("raw_content") or r2.get("content", "")).split()) * 1.3
                       for r2 in state.get("search_results", []))
        reduction = (1 - retrieved_tokens / max(raw_est, 1)) * 100
        print(f"  Retrieved {len(results)} chunks ({retrieved_tokens:.0f} tokens, {reduction:.0f}% vs raw)")

        # Extract claims
        content_text = "\n\n".join(
            f"[{r.get('title','') or r.get('url','')}]\n{r.get('text','')[:600]}"
            for r in results[:10]
        )
    else:
        content_text = "\n".join(state.get("clean_content", ["No content."]))
        print("  ⚠️  No RAG chunks — using raw fallback")


    # Keep analyze prompt small — free models stall on 30k+ contexts
    if len(content_text) > 12000:
        content_text = content_text[:12000]

    from src.engine.budget import budget_status_line
    budget_line = budget_status_line(state)

    prompt = f"""Extract key findings and claims from this research content.

Query: "{state['query']}"
{budget_line}

Content:
{content_text}

Return a JSON object with:
  - "findings": list of 5-10 key findings (each a string)
  - "claims": list of claim objects: {{"text": "...", "atoms": ["atomic fact 1"], "evidence": [{{"url": "fetched URL", "quote": "verbatim contiguous quote"}}], "evidence_ids": ["url1"], "confidence": "high"|"medium"|"low"}}
    Every supported atom MUST have a verbatim quote copied from Content. Do not
    invent quotes or use a search-result URL without fetched text; use an empty
    evidence list when the content does not support the claim.
  - "gaps": list of unanswered questions or missing information
  - "confidence": overall confidence: "high", "medium", or "low\""""

    # Task-tier (Tier-2 #18): extraction is high-throughput, not deep reasoning
    result = call_llm(RESEARCHER_SYSTEM, prompt, model="task")
    analysis = parse_json_dict(result, default=None)
    if not analysis:
        analysis = {"findings": [content_text[:500]], "claims": [], "gaps": [], "confidence": "low"}

    findings = analysis.get("findings", [])
    gaps = analysis.get("gaps", [])
    claims = analysis.get("claims", [])

    # Merge findings
    existing = set(state.get("findings", []))
    for f in findings:
        if f not in existing:
            state["findings"].append(f)
            existing.add(f)

    # ── Task-id ledger (langgraph-deep-research) ──
    # Bind each new finding to the plan section it best serves (mechanical
    # keyword overlap on section titles). Lets the critic see per-section
    # coverage and kills cross-section contamination in later steps.
    section_titles = [str(s.get("title") or "") for s in (state.get("outline") or [])]
    task_ids = [
        str(s.get("task_id") or f"T{i+1}")
        for i, s in enumerate(state.get("outline") or [])
    ]
    ledger = list(state.get("task_ledger") or [])
    for f in findings:
        fl = f.lower()
        best, best_score = -1, 0
        for i, t in enumerate(section_titles):
            words = re.findall(r"[a-zA-Z]{3,}", t.lower())
            if not words:
                continue
            score = sum(1 for w in words if w in fl)
            if score > best_score:
                best_score, best = score, i
        ledger.append({
            "finding": str(f)[:300],
            "task_id": task_ids[best] if best >= 0 else "",
            "section_title": section_titles[best] if best >= 0 else "",
            "iteration": int(state.get("iteration") or 0),
        })
    state["task_ledger"] = ledger[-400:]

    # Store claims with evidence mapping — only URLs actually retrieved this run.
    # LLM-invented evidence IDs (real-looking but never fetched) are dropped here
    # so they cannot reach evidence_map, Bedrock, or Sources (P0.4 fix).
    known_urls: set[str] = set()
    for c in state.get("retrieved_chunks") or []:
        u = canonical_url(c.get("url") or "")
        if u:
            known_urls.add(u)
    for p in state.get("extracted_pages") or []:
        u = canonical_url(p.get("url") or "")
        if u and (p.get("content") or ""):
            known_urls.add(u)
    for c in state.get("run_corpus") or []:
        u = canonical_url(c.get("url") or "")
        if u and (c.get("text") or c.get("content") or ""):
            known_urls.add(u)
    for c in claims:
        evidence = c.get("evidence") or []
        if isinstance(evidence, dict):
            evidence = [evidence]
        candidate_urls = list(c.get("evidence_ids", []))
        candidate_urls.extend(item.get("url") for item in evidence if isinstance(item, dict))
        for url in candidate_urls:
            cu = canonical_url(url)
            if cu and cu in known_urls:
                state["evidence_map"].setdefault(cu, []).append(c.get("text", "")[:100])
    # Seed the map from URLs actually retrieved this run so critic/progress
    # counts real sources even when the LLM invents evidence_ids.
    for u in known_urls:
        state["evidence_map"].setdefault(u, [])

    state["claims"] = state.get("claims", []) + claims
    state["gaps"] = gaps

    # ── RE-TRAC structured state (answers / consulted sources / open hypotheses) ──
    # Persist across iterations so the critic sees explicit gaps and the search
    # strategy avoids re-searching already-consulted sources.
    memory = dict(state.get("research_memory") or {})
    consulted = list(memory.get("consulted_sources") or [])
    seen_consulted = set(consulted)
    for c in (state.get("run_corpus") or []):
        u = canonical_url(c.get("url") or "")
        if u and u not in seen_consulted:
            seen_consulted.add(u)
            consulted.append(u)
    for r in (state.get("search_results") or []):
        u = canonical_url(r.get("url") or "")
        if u and u not in seen_consulted:
            seen_consulted.add(u)
            consulted.append(u)
    memory["consulted_sources"] = consulted[-500:]  # cap size

    answers = list(memory.get("answers") or [])
    for f in findings[:10]:
        if f and f not in answers:
            answers.append(f)
    memory["answers"] = answers[-40:]

    open_hypotheses = list(memory.get("open_hypotheses") or [])
    for g in gaps[:10]:
        if g and g not in open_hypotheses:
            open_hypotheses.append(g)
    memory["open_hypotheses"] = open_hypotheses[-30:]
    state["research_memory"] = memory

    state["status"] = f"Extracted {len(findings)} findings, {len(claims)} claims, {len(gaps)} gaps"
    print(f"  Findings: {len(findings)}, Claims: {len(claims)}, Gaps: {len(gaps)}")
    try:
        from src.engine.progress import get_progress
        p = get_progress()
        for f in findings[:3]:
            p.think("learned", str(f)[:200])
        for g in gaps[:3]:
            p.think("gap", str(g)[:200])
        p.update(
            findings_count=len(state.get("findings") or []),
            sources_count=len(state.get("evidence_map") or {}),
            status=state["status"],
        )
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)
    return state
