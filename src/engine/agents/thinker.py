"""
Thinker agent — large-context reasoning with no side effects (except state).

Uses model tier "thinker" (Gemini key allowed only on this tier).

Nodes:
  thinker_query_scout       — BEFORE planner: web scout + 2–3 parallel Gemini
  thinker_plan_refine       — after planner
  thinker_contradiction_check — after researcher analyze
  thinker_search_strategy   — after critic, crafts better web/Exa queries
"""

from __future__ import annotations

import logging

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm import call_llm as _call_llm
from src.state import ResearchState
from .registry import register

THINKER_SYSTEM = (
    "You are a deep reasoning engine for a Deep Research agent. "
    "No tools. Work only with provided packs. Output valid JSON. Be precise."
)

_last_thinker_call = [0.0]
_thinker_lock = threading.RLock()
MIN_THINKER_INTERVAL = 2.0
# Scout uses 3 parallel Gemini outside this counter; leave room for later hops
MAX_THINKER_CALLS_PER_RUN = 10
_thinker_call_count = [0]

# Gemini free Flash-Lite-class is ~15 RPM; 3 parallel scout calls are fine
_GEMINI_SCOUT_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)


def _should_invoke_thinker(state: ResearchState) -> bool:
    with _thinker_lock:
        if _thinker_call_count[0] >= MAX_THINKER_CALLS_PER_RUN:
            return False
        now = time.time()
        if now - _last_thinker_call[0] < MIN_THINKER_INTERVAL:
            return False
        _last_thinker_call[0] = now
        _thinker_call_count[0] += 1
        return True


def _invoke_thinker(context_pack: str, purpose: str) -> dict:
    prompt = f"""Analyze for: {purpose}

{context_pack}

Return a JSON object with your analysis."""
    result = _call_llm(THINKER_SYSTEM, prompt, model="thinker")
    try:
        cleaned = result.strip()
        for pfx in ("```json", "```"):
            if cleaned.startswith(pfx):
                cleaned = cleaned[len(pfx):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Failed to parse Thinker output", "raw": result[:500]}


def _parse_json_loose(raw: str) -> dict:
    try:
        cleaned = (raw or "").strip()
        for pfx in ("```json", "```"):
            if cleaned.startswith(pfx):
                cleaned = cleaned[len(pfx):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        # Strip leading CoT before first {
        if "{" in cleaned and not cleaned.startswith("{"):
            cleaned = cleaned[cleaned.find("{"):]
        if "}" in cleaned:
            cleaned = cleaned[: cleaned.rfind("}") + 1]
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {"error": "parse_failed", "raw": (raw or "")[:400]}


def _gemini_direct(system: str, user: str, max_tokens: int = 1200) -> str:
    """Call Gemini OpenAI-compatible endpoint directly (for parallel scout).

    Falls back to thinker tier (Groq/Zen) if Gemini unavailable.
    """
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        try:
            from src.gateway.providers import OpenAICompatibleProvider
            base = os.getenv(
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            )
            p = OpenAICompatibleProvider("gemini", base, key)
            last_err = None
            for model in _GEMINI_SCOUT_MODELS:
                try:
                    out = p.complete(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        model=model,
                        max_tokens=max_tokens,
                        temperature=0.2,
                        timeout=45.0,
                    )
                    text = getattr(out, "text", "") or ""
                    if text.strip():
                        return text
                except Exception as e:
                    last_err = e
                    continue
            if last_err:
                print(f"  [scout] Gemini direct failed ({last_err}); fallback thinker tier")
        except Exception as e:
            print(f"  [scout] Gemini setup failed ({e}); fallback thinker tier")
    return _call_llm(system, user, model="thinker", max_tokens=max_tokens)


def _web_scout_snippets(query: str, k: int = 5) -> list[dict]:
    """Lightweight Exa/web scout for query understanding (not full research)."""
    hits: list[dict] = []
    if os.getenv("EXA_API_KEY"):
        try:
            from src.tools.adapters.exa import exa_search
            for r in exa_search(query, max_results=k)[:k]:
                hits.append({
                    "title": (r.get("title") or "")[:120],
                    "url": r.get("url") or "",
                    "snippet": (r.get("content") or r.get("raw_content") or "")[:400],
                })
        except Exception as e:
            print(f"  [scout] Exa scout failed: {e}")
    if not hits:
        try:
            from src.tools import execute_searches
            for r in execute_searches([query], max_results=k)[:k]:
                hits.append({
                    "title": (r.get("title") or "")[:120],
                    "url": r.get("url") or "",
                    "snippet": (r.get("content") or "")[:400],
                })
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)
    return hits


@register("thinker_query_scout")
def thinker_query_scout(state: ResearchState) -> ResearchState:
    """Start-of-run thinker: web scout + 2–3 parallel Gemini analyses.

    Improves plan quality by seeding:
      - refined_query / sub_questions
      - must_cover systems/papers
      - search_queries
      - eval_axes / failure_modes
    Within Gemini free RPM (~15 for Flash-Lite): 3 parallel requests OK.
    """
    query = state.get("query") or ""
    state["status"] = "Scout: web + parallel thinker..."
    print(f"\n💭 [Thinker] Query scout (web + 3× parallel Gemini/thinker)")
    try:
        from src.engine.progress import get_progress
        get_progress().update(stage="scouting", status=state["status"])
        get_progress().think("next", "Scout query with web + parallel Gemini")
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)

    hits = _web_scout_snippets(query, k=5)
    scout_blob = "\n".join(
        f"- {h['title']} | {h['url']}\n  {h['snippet'][:280]}"
        for h in hits
    ) or "(no web hits)"
    print(f"  Scout web hits: {len(hits)}")

    base_ctx = f"USER QUERY:\n{query}\n\nWEB SCOUT HITS:\n{scout_blob}\n"

    tasks = {
        "intent": (
            "Research intent analyst. Return JSON only.",
            base_ctx
            + "Return JSON: {\n"
            '  "refined_query": "clearer research question",\n'
            '  "sub_questions": ["...", "..."],\n'
            '  "scope": "what to include/exclude",\n'
            '  "assumptions": ["..."]\n'
            "}",
        ),
        "systems": (
            "Domain systems/papers scout. Return JSON only. Prefer real names.",
            base_ctx
            + "Return JSON: {\n"
            '  "must_cover_systems": ["Self-RAG","CRAG","RAPTOR","ColBERT","DPR","HyDE","GraphRAG", ... real ones],\n'
            '  "must_cover_papers": ["Author Year title or arxiv id if known"],\n'
            '  "search_queries": ["high precision Exa/web queries", "..."]\n'
            "}",
        ),
        "eval": (
            "Evaluation and failure-mode planner. Return JSON only.",
            base_ctx
            + "Return JSON: {\n"
            '  "eval_axes": ["context relevance","faithfulness","answer relevance", ...],\n'
            '  "failure_modes": ["lost-in-middle","hallucination-on-hallucination", ...],\n'
            '  "production_topics": ["chunking","reranking","monitoring", ...],\n'
            '  "outline_hints": ["section titles for deep report"]\n'
            "}",
        ),
    }

    results: dict[str, dict] = {}

    def _one(name: str, system: str, user: str) -> tuple[str, dict]:
        raw = _gemini_direct(system, user, max_tokens=1400)
        return name, _parse_json_loose(raw)

    # 3 parallel Gemini calls — within free ~15 RPM Flash-Lite
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [
            pool.submit(_one, name, sys, usr)
            for name, (sys, usr) in tasks.items()
        ]
        for fut in as_completed(futs):
            try:
                name, data = fut.result()
                results[name] = data
                ok = "error" not in data
                print(f"  Scout/{name}: {'ok' if ok else 'weak'} keys={list(data.keys())[:6]}")
            except Exception as e:
                print(f"  Scout parallel task failed: {e}")

    intent = results.get("intent") or {}
    systems = results.get("systems") or {}
    evalp = results.get("eval") or {}

    # Merge into state for planner / researcher
    scout = {
        "web_hits": hits,
        "refined_query": intent.get("refined_query") or query,
        "sub_questions": intent.get("sub_questions") or [],
        "scope": intent.get("scope") or "",
        "assumptions": intent.get("assumptions") or [],
        "must_cover_systems": systems.get("must_cover_systems") or [],
        "must_cover_papers": systems.get("must_cover_papers") or [],
        "eval_axes": evalp.get("eval_axes") or [],
        "failure_modes": evalp.get("failure_modes") or [],
        "production_topics": evalp.get("production_topics") or [],
        "outline_hints": evalp.get("outline_hints") or [],
    }
    state["scout"] = scout
    state.setdefault("plan", {})
    state["plan"]["scout"] = {
        k: scout[k]
        for k in (
            "refined_query", "sub_questions", "must_cover_systems",
            "must_cover_papers", "eval_axes", "failure_modes", "outline_hints",
        )
    }

    # Seed search queries (planner may refine)
    seeded: list[str] = []
    for q in systems.get("search_queries") or []:
        if q and str(q) not in seeded:
            seeded.append(str(q))
    for q in intent.get("sub_questions") or []:
        if q and str(q) not in seeded:
            seeded.append(str(q))
    if not seeded:
        seeded = [query]
    state["search_queries"] = seeded[:8]

    # Findings seed so critic has on-topic anchors early
    seeds = []
    if scout["must_cover_systems"]:
        seeds.append(
            "Must cover systems: " + ", ".join(str(s) for s in scout["must_cover_systems"][:12])
        )
    if scout["eval_axes"]:
        seeds.append("Eval axes: " + ", ".join(str(s) for s in scout["eval_axes"][:8]))
    if scout["failure_modes"]:
        seeds.append("Failure modes: " + ", ".join(str(s) for s in scout["failure_modes"][:8]))
    state["findings"] = list(state.get("findings") or []) + seeds

    try:
        from src.engine.progress import get_progress
        get_progress().think(
            "learned",
            f"Scout systems: {', '.join(str(s) for s in (scout['must_cover_systems'] or [])[:6])}",
        )
        get_progress().think("next", f"Plan with {len(state['search_queries'])} seeded queries")
        get_progress().update(plan=state.get("plan") or {})
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)

    state["status"] = (
        f"Scout done: {len(hits)} web hits, "
        f"{len(scout['must_cover_systems'])} systems, "
        f"{len(state['search_queries'])} queries"
    )
    print(f"  {state['status']}")
    return state


def _build_plan_pack(state: ResearchState) -> str:
    plan = state.get("plan", {})
    scout = state.get("scout") or plan.get("scout") or {}
    return "\n".join([
        f"QUERY: {state.get('query', '')}",
        f"SCOUT_REFINED: {scout.get('refined_query', '')}",
        f"MUST_COVER_SYSTEMS: {json.dumps(scout.get('must_cover_systems', [])[:15])}",
        f"MUST_COVER_PAPERS: {json.dumps(scout.get('must_cover_papers', [])[:10])}",
        f"EVAL_AXES: {json.dumps(scout.get('eval_axes', [])[:10])}",
        f"FAILURE_MODES: {json.dumps(scout.get('failure_modes', [])[:10])}",
        f"OUTLINE_HINTS: {json.dumps(scout.get('outline_hints', [])[:12])}",
        f"TOPIC: {plan.get('topic', '')}",
        f"SUBTOPICS: {json.dumps(plan.get('subtopics', []))}",
        f"OUTLINE: {json.dumps(plan.get('outline', []))}",
        f"FINDINGS: {json.dumps(state.get('findings', [])[:10])}",
    ])


def _build_contradiction_pack(state: ResearchState) -> str:
    return "\n".join([
        f"QUERY: {state.get('query', '')}",
        "CLAIMS:",
        json.dumps(state.get("claims", [])[:15], indent=2),
        "FINDINGS:",
        json.dumps(state.get("findings", [])[:15], indent=2),
    ])


@register("thinker_plan_refine")
def thinker_plan_refine(state: ResearchState) -> ResearchState:
    if not _should_invoke_thinker(state):
        return state
    plan_sections = state.get("plan", {}).get("outline", [])
    if len(plan_sections) < 3:
        print(f"\n💭 [Thinker] Skipped plan refine — simple plan ({len(plan_sections)} sections)")
        return state

    print(f"\n💭 [Thinker] Refining research plan ({len(plan_sections)} sections)")
    try:
        from src.engine.progress import get_progress
        get_progress().think("next", "Thinker refining research plan")
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)

    analysis = _invoke_thinker(_build_plan_pack(state), "plan refinement for deep research")
    if "error" not in analysis:
        if analysis.get("refined_outline") and isinstance(analysis["refined_outline"], list):
            refined = analysis["refined_outline"]
            if len(refined) >= len(plan_sections):
                state["plan"]["outline"] = refined
                state["outline"] = [
                    {"title": s.get("title", f"Section {i+1}") if isinstance(s, dict) else str(s), "order": i}
                    for i, s in enumerate(refined)
                ]
                print(f"  Refined outline: {len(refined)} sections")
        if analysis.get("refined_queries") and isinstance(analysis["refined_queries"], list):
            state["search_queries"] = analysis["refined_queries"][:6]
            print(f"  Refined queries: {state['search_queries']}")
        try:
            from src.engine.progress import get_progress
            get_progress().think("learned", f"Plan refined to {len(state.get('outline', []))} sections")
            get_progress().update(plan=state.get("plan") or {})
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)
    return state


@register("thinker_contradiction_check")
def thinker_contradiction_check(state: ResearchState) -> ResearchState:
    if not _should_invoke_thinker(state):
        return state
    claims = state.get("claims", [])
    if len(claims) < 3 and len(state.get("findings") or []) < 5:
        print(f"\n💭 [Thinker] Skipped contradiction check — sparse claims")
        return state

    print(f"\n💭 [Thinker] Checking contradictions ({len(claims)} claims)")
    try:
        from src.engine.progress import get_progress
        get_progress().think("next", "Thinker checking contradictions across sources")
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)

    analysis = _invoke_thinker(
        _build_contradiction_pack(state),
        "contradiction detection and multi-source reasoning",
    )
    if "error" not in analysis:
        contradictions = analysis.get("contradictions", [])
        # Free models sometimes return dict / str instead of list
        if isinstance(contradictions, dict):
            contradictions = list(contradictions.values())
        elif isinstance(contradictions, str):
            contradictions = [contradictions] if contradictions.strip() else []
        elif not isinstance(contradictions, list):
            contradictions = []
        if contradictions:
            print(f"  Found {len(contradictions)} potential contradictions")
            state["gaps"] = list(state.get("gaps") or []) + [
                f"Contradiction: {c.get('description', str(c))[:200]}"
                if isinstance(c, dict) else f"Contradiction: {str(c)[:200]}"
                for c in contradictions[:5]
            ]
            try:
                from src.engine.progress import get_progress
                for c in contradictions[:3]:
                    get_progress().think(
                        "gap",
                        c.get("description", str(c))[:200] if isinstance(c, dict) else str(c)[:200],
                    )
            except Exception:
                logging.getLogger(__name__).debug("ignored error", exc_info=True)
        follow = analysis.get("follow_up_queries")
        if isinstance(follow, list) and state.get("needs_more_research"):
            state["search_queries"] = [str(q) for q in follow[:6] if q]
    return state


@register("thinker_search_strategy")
def thinker_search_strategy(state: ResearchState) -> ResearchState:
    """After critic: craft high-quality Exa/web queries when more research needed.

    Always runs lightly when needs_more_research or replan/off_topic.
    Uses thinker tier (Gemini allowed).
    """
    if not state.get("needs_more_research") and not state.get("replan") and not state.get("off_topic"):
        return state

    # Prefer thinker even if rate-limited — fall back to fast craft
    use_thinker = _should_invoke_thinker(state)
    print(f"\n💭 [Thinker] Search strategy (thinker={'yes' if use_thinker else 'fast-fallback'})")

    scout = state.get("scout") or {}
    memory = state.get("research_memory") or {}
    consulted = list(memory.get("consulted_sources") or [])[-25:]
    open_hyp = list(memory.get("open_hypotheses") or [])[:8]
    missing_facts = state.get("missing_facts") or []
    from src.engine.budget import budget_status_line
    budget_line = budget_status_line(state)
    pack = "\n".join([
        f"QUERY: {state.get('query', '')}",
        f"{budget_line}",
        f"OFF_TOPIC: {state.get('off_topic', False)}",
        f"REPLAN: {state.get('replan', False)}",
        f"GAPS: {json.dumps((state.get('gaps') or [])[:10])}",
        f"OPEN_HYPOTHESES: {json.dumps(open_hyp)}",
        f"MISSING_FACTS: {json.dumps(missing_facts[:8])}",
        f"FINDINGS: {json.dumps((state.get('findings') or [])[:8])}",
        f"MUST_COVER_SYSTEMS: {json.dumps((scout.get('must_cover_systems') or [])[:12])}",
        f"CURRENT_QUERIES: {json.dumps(state.get('search_queries') or [])}",
        f"MODE: {state.get('mode', 'standard')}",
        "Produce SHORT high-precision search queries (not the full user query repeated).",
        "Each query should target a named system, benchmark, or failure mode from gaps/missing_facts/must_cover.",
        "Cover the missing_facts: every fact without evidence needs a targeted query.",
        "AVOID re-searching these already-consulted sources/domains if possible.",
        f"ALREADY_CONSULTED ({len(consulted)}): {json.dumps(consulted[:12])}",
        "Optimized for Exa neural search and arXiv.",
    ])

    purpose = (
        "web search strategy for deep research recovery: return JSON with "
        "search_queries (list of 4-6 strings), arxiv_queries (list), "
        "search_modes (dict mapping each search_query to one of: atom | deep | "
        "wide | entity_collect | web_structure), "
        "rationale (string), learned (list of short strings)"
    )

    if use_thinker:
        analysis = _invoke_thinker(pack, purpose)
    else:
        raw = _call_llm(THINKER_SYSTEM, f"{purpose}\n\n{pack}", model="fast")
        try:
            cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
            analysis = json.loads(cleaned)
        except Exception:
            analysis = {}

    queries = analysis.get("search_queries") if isinstance(analysis, dict) else None
    arxiv_q = analysis.get("arxiv_queries") if isinstance(analysis, dict) else None
    merged: list[str] = []
    if isinstance(queries, list):
        merged.extend(str(q) for q in queries if q)
    if isinstance(arxiv_q, list):
        merged.extend(f"{q} site:arxiv.org" if "arxiv" not in str(q).lower() else str(q) for q in arxiv_q if q)
    # Targeted re-search from structured missing facts (r1-reasoning-rag)
    for mf in missing_facts[:4]:
        for sq in (mf.get("suggested_queries") or []):
            merged.append(str(sq))
    if not merged:
        q = state.get("query", "")
        systems = (scout.get("must_cover_systems") or [])[:4]
        merged = [f"{s} RAG hallucination evaluation" for s in systems]
        merged += [
            f"{q[:80]} site:arxiv.org",
            f"{q[:80]} survey faithfulness",
            f"RAG production best practices hallucination 2025",
        ]

    # Dedup preserve order
    seen = set()
    final = []
    for q in merged:
        ql = q.lower().strip()
        if ql and ql not in seen:
            seen.add(ql)
            final.append(q.strip())
    state["search_queries"] = final[:8]
    print(f"  Search strategy queries ({len(state['search_queries'])}): {state['search_queries'][:4]}...")

    # ── Search-mode routing (WebSwarm): tag each query with an execution mode ──
    modes_from_llm = analysis.get("search_modes") if isinstance(analysis, dict) else None
    search_modes: dict[str, str] = {}
    for q in state["search_queries"]:
        mode = ""
        if isinstance(modes_from_llm, dict):
            mode = str(modes_from_llm.get(q) or modes_from_llm.get(q.lower()) or "")
        search_modes[q] = (
            mode
            if mode in ("atom", "deep", "wide", "entity_collect", "web_structure")
            else _default_search_mode(q)
        )
    state["search_modes"] = search_modes
    print(f"  Search modes: {json.dumps(search_modes)}")

    try:
        from src.engine.progress import get_progress
        p = get_progress()
        p.think("next", f"Exa search: {state['search_queries'][0][:100]}")
        for item in (analysis.get("learned") or [])[:3]:
            p.think("learned", str(item))
        if analysis.get("rationale"):
            p.think("next", str(analysis["rationale"])[:200])
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)

    # Optional replan outline
    if state.get("replan") and isinstance(analysis, dict) and analysis.get("refined_outline"):
        refined = analysis["refined_outline"]
        if isinstance(refined, list) and refined:
            state["outline"] = [
                {"title": (s.get("title") if isinstance(s, dict) else str(s)), "order": i}
                for i, s in enumerate(refined)
            ]
            state.setdefault("plan", {})["outline"] = refined
            print(f"  Re-planned outline: {[s['title'] for s in state['outline']]}")

    return state


def _default_search_mode(q: str) -> str:
    """Deterministic mode assignment when the LLM returns none (WebSwarm)."""
    ql = (q or "").lower()
    if "site:" in ql or "filetype:" in ql:
        return "web_structure"
    if any(w in ql for w in (
        "list of", "top ", "best ", "compare", "versus", " vs ",
        "taxonomy", "overview", "survey",
    )):
        return "wide"
    if any(w in ql for w in (
        "who ", "when ", "where ", "how many", "what is",
        "define ", "what year",
    )):
        return "atom"
    if any(w in ql for w in (
        "systems", "papers", "benchmarks", "models", "companies", "tools",
    )):
        return "entity_collect"
    return "deep"


def reset_thinker() -> None:
    global _last_thinker_call, _thinker_call_count
    with _thinker_lock:
        _last_thinker_call[0] = 0.0
        _thinker_call_count[0] = 0


def disable_thinker() -> None:
    global _thinker_call_count
    with _thinker_lock:
        _thinker_call_count[0] = MAX_THINKER_CALLS_PER_RUN
