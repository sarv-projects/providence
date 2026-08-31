"""
Build the multi-agent LangGraph research agent graph.

Architecture (A3 + Ultra steals):
  Scout → Planner → Thinker(plan) → Research loop
    → Devil's advocate → Claim adjudicator (Socratic hop 0–1)
    → Triangulator → Synth → Compiler (Bedrock / Debt / Sources)

Integrity:
  - per-run RAG isolation
  - off-topic hard fail / re-search
  - CoVe-lite adjudication + claim–evidence ship-gate
  - Exa-primary search when key present
"""

from __future__ import annotations

import logging

import threading
import time

from langgraph.graph import StateGraph

from src.state import ResearchState, initial_state

import src.engine.agents  # noqa: F401
from src.engine.agents.registry import get_agent


def should_continue_research(state: ResearchState) -> str:
    if state.get("abort_synthesis"):
        return "compile_abort"
    if state.get("needs_more_research", False):
        return "research_again"
    return "adversary"


def after_adjudicator(state: ResearchState) -> str:
    """Socratic tree: one optional re-gather, else triangulate."""
    if state.get("abort_synthesis"):
        return "compile_abort"
    if state.get("socratic_reopen"):
        return "socratic_again"
    return "triangulate"


def build_graph() -> StateGraph:
    builder = StateGraph(ResearchState)

    def cancellable(agent):
        """Check the job between graph nodes for cooperative cancellation."""
        def wrapped(state: ResearchState) -> ResearchState:
            job_id = state.get("job_id")
            if job_id:
                from src.engine.jobs import get_jobs
                if get_jobs().is_cancelled(job_id):
                    raise RuntimeError("Research cancelled by user")
            return agent(state)
        return wrapped

    # Scout first (web + parallel Gemini), then plan
    builder.add_node("thinker_query_scout", cancellable(get_agent("thinker_query_scout")))
    builder.add_node("planner", cancellable(get_agent("planner")))
    builder.add_node("thinker_plan_refine", cancellable(get_agent("thinker_plan_refine")))
    builder.add_node("researcher_gather", cancellable(get_agent("researcher_gather")))
    builder.add_node("researcher_analyze", cancellable(get_agent("researcher_analyze")))
    builder.add_node("thinker_contradiction_check", cancellable(get_agent("thinker_contradiction_check")))
    builder.add_node("critic", cancellable(get_agent("critic")))
    builder.add_node("thinker_search_strategy", cancellable(get_agent("thinker_search_strategy")))
    # Ultra steals
    builder.add_node("devil_advocate_gather", cancellable(get_agent("devil_advocate_gather")))
    builder.add_node("claim_adjudicator", cancellable(get_agent("claim_adjudicator")))
    builder.add_node("triangulator", cancellable(get_agent("triangulator")))
    builder.add_node("synthesizer_outline", cancellable(get_agent("synthesizer_outline")))
    builder.add_node("synthesizer_write", cancellable(get_agent("synthesizer_write")))
    builder.add_node("compiler", cancellable(get_agent("compiler")))

    def _abort_passthrough(state: ResearchState) -> ResearchState:
        """Skip synth when aborted — go straight to compiler with error note."""
        if not state.get("sections"):
            state["sections"] = [
                {
                    "title": "Research Status",
                    "content": (
                        f"Research could not complete with on-topic evidence.\n\n"
                        f"**Error:** {state.get('error', 'aborted')}\n\n"
                        f"Query: {state.get('query', '')}\n"
                    ),
                    "sources": [],
                }
            ]
        return state

    builder.add_node("abort_passthrough", _abort_passthrough)

    builder.set_entry_point("thinker_query_scout")
    builder.add_edge("thinker_query_scout", "planner")
    builder.add_edge("planner", "thinker_plan_refine")
    builder.add_edge("thinker_plan_refine", "researcher_gather")
    builder.add_edge("researcher_gather", "researcher_analyze")
    builder.add_edge("researcher_analyze", "thinker_contradiction_check")
    builder.add_edge("thinker_contradiction_check", "critic")
    builder.add_edge("critic", "thinker_search_strategy")

    builder.add_conditional_edges(
        "thinker_search_strategy",
        should_continue_research,
        {
            "research_again": "researcher_gather",
            "adversary": "devil_advocate_gather",
            "compile_abort": "abort_passthrough",
        },
    )

    builder.add_edge("devil_advocate_gather", "claim_adjudicator")
    builder.add_conditional_edges(
        "claim_adjudicator",
        after_adjudicator,
        {
            "socratic_again": "researcher_gather",
            "triangulate": "triangulator",
            "compile_abort": "abort_passthrough",
        },
    )

    builder.add_edge("abort_passthrough", "compiler")
    builder.add_edge("triangulator", "synthesizer_outline")
    builder.add_edge("synthesizer_outline", "synthesizer_write")
    builder.add_edge("synthesizer_write", "compiler")
    builder.set_finish_point("compiler")

    return builder.compile()


def create_research_plan(
    query: str,
    mode: str = "standard",
    autonomy: str = "L1",
    clarifications: dict | None = None,
) -> dict:
    """Generate an editable research plan (and clarifying questions if needed).

    Does not run gather/synth. Returns plan store payload.
    """
    from src.engine.plan_store import get_plans
    from src.engine.clarify import generate_clarifying_questions, apply_clarifications, is_ambiguous
    from src.engine.agents.planner import planner
    from src.engine.agents.thinker import thinker_plan_refine, reset_thinker

    store = get_plans()
    entry = store.create(query, mode=mode, autonomy=autonomy)

    # Clarifying prelude (ChatGPT-style)
    clar = generate_clarifying_questions(query)
    entry.clarifying_questions = list(clar.get("questions") or [])
    entry.needs_clarification = bool(clar.get("ambiguous")) and not clarifications

    enriched = query
    if clarifications:
        enriched = apply_clarifications(query, clarifications, clar.get("assumptions"))
        entry.clarifications = dict(clarifications)
        entry.needs_clarification = False
    elif clar.get("ambiguous") and autonomy.upper() == "L2":
        # L2: stop for answers before plan is final
        store.update(
            entry.plan_id,
            status="awaiting_clarification",
            clarifying_questions=entry.clarifying_questions,
            needs_clarification=True,
            plan={
                "topic": query,
                "assumptions": clar.get("assumptions") or [],
                "refined_query_hint": clar.get("refined_query_hint") or query,
            },
        )
        return store.get(entry.plan_id).to_dict()  # type: ignore

    reset_thinker()
    state = initial_state(enriched)
    state["mode"] = mode
    state["autonomy"] = autonomy
    state["clarifications"] = clarifications or {}
    state["mode_flags"] = {
        "structured_output": mode == "compare",
        "academic_bias": mode == "academic",
        "force_arxiv": mode in ("deep", "academic", "ultra-long", "standard"),
        "vault_rag": True,
        "recency_bias": mode == "recency",
        "requires_temporal": False,
    }
    state = planner(state)
    try:
        state = thinker_plan_refine(state)
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)

    plan = state.get("plan") or {}
    outline = state.get("outline") or []
    queries = state.get("search_queries") or []
    store.update(
        entry.plan_id,
        status="awaiting_approval" if autonomy.upper() in ("L2", "L3") else "draft",
        plan=plan,
        outline=outline,
        search_queries=queries,
        clarifying_questions=entry.clarifying_questions,
        needs_clarification=False,
        query=enriched if clarifications else query,
    )
    return store.get(entry.plan_id).to_dict()  # type: ignore


def run_research(
    query: str,
    mode: str = "standard",
    autonomy: str = "L1",
    job_id: str = "",
    background: bool = False,
    plan_only: bool = False,
    approved_plan: dict | None = None,
    plan_id: str = "",
    clarifications: dict | None = None,
    skip_clarify: bool = False,
    model: str | None = None,
    max_cost_usd: float | None = None,
    max_iterations_override: int | None = None,
    max_tokens: int | None = None,
) -> ResearchState:
    """Run multi-agent research.

    background=True: daemon thread + job_id.
    plan_only=True: generate editable plan and return (no gather).
    approved_plan: skip planning; use this plan dict + search_queries/outline.
    L2 without approved_plan: auto plan_only (require approval).
    """
    if plan_only or (autonomy.upper() == "L2" and not approved_plan and not background):
        # Synchronous plan generation path
        payload = create_research_plan(
            query, mode=mode, autonomy=autonomy, clarifications=clarifications
        )
        st = initial_state(query)
        st["plan"] = payload.get("plan") or {}
        st["outline"] = payload.get("outline") or []
        st["search_queries"] = payload.get("search_queries") or []
        st["plan_id"] = payload.get("plan_id", "")
        st["clarifying_questions"] = payload.get("clarifying_questions") or []
        st["status"] = payload.get("status", "draft")
        st["mode"] = mode
        st["autonomy"] = autonomy
        return st

    if background:
        from src.engine.jobs import get_jobs
        job = get_jobs().create(query, mode=mode, autonomy=autonomy)

        def _worker() -> None:
            try:
                if get_jobs().is_cancelled(job.job_id):
                    return
                get_jobs().update(job.job_id, status="running", started_at=time.time())
                if get_jobs().is_cancelled(job.job_id):
                    return
                # L2 background without approved plan → generate plan first, pause
                if autonomy.upper() == "L2" and not approved_plan:
                    payload = create_research_plan(
                        query, mode=mode, autonomy=autonomy, clarifications=clarifications
                    )
                    get_jobs().update(
                        job.job_id,
                        status="awaiting_plan",
                        stage="plan_review",
                        plan=payload.get("plan") or {},
                        plan_id=payload.get("plan_id") or "",
                        next_action="Edit/approve research plan",
                        run_id="",
                    )
                    # Attach plan_id into job thoughts
                    get_jobs().add_thought(
                        job.job_id, "next", f"plan_id={payload.get('plan_id')} awaiting approval"
                    )
                    from src.engine.plan_store import get_plans
                    get_plans().update(payload["plan_id"], job_id=job.job_id)
                    return
                run_research(
                    query,
                    mode=mode,
                    autonomy=autonomy,
                    job_id=job.job_id,
                    background=False,
                    approved_plan=approved_plan,
                    plan_id=plan_id,
                    clarifications=clarifications,
                    skip_clarify=skip_clarify,
                    model=model,
                    max_cost_usd=max_cost_usd,
                    max_iterations_override=max_iterations_override,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                if not get_jobs().is_cancelled(job.job_id):
                    get_jobs().update(
                        job.job_id,
                        status="error",
                        error="Research failed; see server logs",
                        finished_at=time.time(),
                    )

        threading.Thread(target=_worker, daemon=True).start()
        st = initial_state(query)
        st["job_id"] = job.job_id
        st["status"] = "queued"
        return st

    from src.engine.modes import load_modes, get_mode
    from src.engine.progress import start_run_progress, end_run_progress

    registry = load_modes()
    mode_config = get_mode(registry, mode)
    dial = mode_config.quality
    budgets = mode_config.budgets

    # Intensity: breadth with Exa, but bounded wall-time (free LLMs are the bottleneck)
    intensity = {
        "quick": {"max_search_results": 6, "max_extract_pages": 5, "max_iterations": 2},
        "standard": {"max_search_results": 8, "max_extract_pages": 8, "max_iterations": min(budgets.max_iterations, 4)},
        "deep": {"max_search_results": 10, "max_extract_pages": 10, "max_iterations": min(budgets.max_iterations, 4)},
        "academic": {"max_search_results": 10, "max_extract_pages": 10, "max_iterations": min(budgets.max_iterations, 4)},
        "ultra-long": {"max_search_results": 12, "max_extract_pages": 12, "max_iterations": max(budgets.max_iterations, 8)},
        "compare": {"max_search_results": 8, "max_extract_pages": 8, "max_iterations": min(budgets.max_iterations, 4)},
        "recency": {"max_search_results": 8, "max_extract_pages": 8, "max_iterations": min(budgets.max_iterations, 3)},
    }.get(mode, {"max_search_results": 8, "max_extract_pages": 8})

    max_iters = intensity.get("max_iterations", budgets.max_iterations)
    if max_iterations_override is not None and int(max_iterations_override) > 0:
        max_iters = max(1, min(int(max_iterations_override), 100))

    # Per-run progress isolation: bind a run-scoped progress object to this
    # thread and register it under job_id — concurrent runs each mutate their
    # own instance instead of clobbering one global (web endpoints read it
    # back via get_progress_by_job). The old archive-based fallback in
    # ResearchProgress remains for non-registered runs.
    progress = start_run_progress(job_id)
    try:
        return _run_research_inner(
            query, mode=mode, autonomy=autonomy, job_id=job_id, plan_id=plan_id,
            approved_plan=approved_plan, clarifications=clarifications,
            skip_clarify=skip_clarify, mode_config=mode_config, registry=registry,
            dial=dial, budgets=budgets, intensity=intensity, max_iters=max_iters,
            progress=progress, model=model, max_cost_usd=max_cost_usd,
            max_tokens=max_tokens,
        )
    finally:
        end_run_progress()


def _run_research_inner(
    query, *, mode, autonomy, job_id, plan_id, approved_plan, clarifications,
    skip_clarify, mode_config, registry, dial, budgets, intensity, max_iters,
    progress, model=None, max_cost_usd=None, max_tokens=None,
):
    from src.engine.agents.thinker import (
        disable_thinker as _disable_thinker,
        reset_thinker as _reset_thinker,
    )
    from src.engine.agents.triangulator import (
        disable_triangulator as _disable_triangulator,
        reset_triangulator as _reset_triangulator,
    )
    from src.rag.pipeline import begin_run
    from src.engine.clarify import apply_clarifications, generate_clarifying_questions

    _reset_thinker()
    _reset_triangulator()
    if not dial.thinker_enabled:
        _disable_thinker()
    if not dial.triangulation_enabled:
        _disable_triangulator()

    # Optional clarify enrichment for L1 when answers provided
    work_query = query
    if clarifications:
        work_query = apply_clarifications(query, clarifications)
    elif not skip_clarify and not approved_plan:
        # Soft: record questions in progress but continue for L1
        try:
            c = generate_clarifying_questions(query)
            if c.get("questions"):
                progress.think("gap", f"Ambiguity notes: {c['questions'][0][:160]}")
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)

    state = initial_state(work_query, max_iterations=max_iters)
    run_id = state["run_id"]
    begin_run(run_id)

    progress.start(work_query, run_id=run_id, max_iterations=max_iters, job_id=job_id, mode=mode)
    progress.update(stage="starting", status=f"Mode={mode} autonomy={autonomy} (Exa+Zen)")
    progress.think("next", f"Planning research for: {work_query[:120]}")
    if job_id:
        try:
            from src.engine.jobs import get_jobs
            get_jobs().update(
                job_id,
                status="running",
                started_at=time.time(),
                run_id=run_id,
                stage="starting",
            )
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)

    max_cost = budgets.max_cost_usd
    if max_cost_usd is not None and float(max_cost_usd) > 0:
        max_cost = min(float(max_cost_usd), 1000.0)
    max_time = budgets.max_time_s
    if autonomy == "L3":
        max_cost = min(max_cost, max_cost * 0.8 if max_cost > 0 else 0.4)

    if mode_config.requires_temporal or mode == "ultra-long":
        try:
            from src.engine.temporal.client import try_run_temporal_research
            report = try_run_temporal_research(
                work_query, mode=mode, autonomy=autonomy,
                config={"mode": mode, "autonomy": autonomy, "run_id": run_id},
            )
            if report:
                state["report"] = report
                state["status"] = "Research complete (Temporal)"
                progress.update(stage="complete", finished=True, report=report, status="complete")
                return state
        except Exception as e:
            print(f"  [ultra-long] Temporal fallback: {e}")

    state["mode"] = mode
    state["autonomy"] = autonomy
    state["job_id"] = job_id
    state["plan_id"] = plan_id
    state["clarifications"] = clarifications or {}
    state["quality"] = {
        "max_tokens_per_call": int(max_tokens or dial.max_tokens_per_call),
        "max_search_results": intensity.get("max_search_results", dial.max_search_results),
        "max_extract_pages": intensity.get("max_extract_pages", dial.max_extract_pages),
        "thinker_enabled": dial.thinker_enabled,
        "triangulation_enabled": dial.triangulation_enabled,
        # Factoids cost free-LLM wall time; only if dial enables them (not forced by mode)
        "factoid_enabled": bool(dial.factoid_enabled),
    }
    state["budgets"] = {
        "max_tokens": int(budgets.max_tokens),
        "max_cost_usd": max_cost,
        "max_time_s": max_time,
        # Honor the configured limit; only fall back to a mode default when
        # the config leaves it unset/zero (previously max() silently raised
        # configured limits to a forced 25/40 minimum).
        "max_tool_calls": budgets.max_tool_calls or (40 if mode in ("deep", "ultra-long") else 25),
        "max_iterations": max_iters,
        "started_at": time.time(),
        "tool_calls": 0,
        "spent_usd": 0.0,
        "tokens_used": 0,
    }
    state["mode_flags"] = {
        "recency_bias": mode_config.recency_bias,
        "academic_bias": mode_config.academic_bias or mode == "academic",
        "structured_output": mode_config.structured_output or mode == "compare",
        "vault_rag": mode_config.vault_rag,
        "requires_temporal": mode_config.requires_temporal,
        "force_arxiv": mode in ("deep", "academic", "ultra-long", "standard"),
    }

    # Inject approved / edited plan
    if approved_plan:
        state["plan"] = approved_plan.get("plan") or approved_plan
        state["outline"] = approved_plan.get("outline") or state.get("outline") or []
        state["search_queries"] = approved_plan.get("search_queries") or state.get("search_queries") or []
        # outline may live inside plan
        if not state["outline"] and isinstance(state["plan"], dict):
            state["outline"] = [
                {"title": s.get("title", f"Section {i+1}"), "order": i}
                for i, s in enumerate(state["plan"].get("outline") or [])
            ]
        if not state["search_queries"] and isinstance(state["plan"], dict):
            state["search_queries"] = list(state["plan"].get("search_queries") or [work_query])[:8]
        state["plan_approved"] = True
        progress.update(plan=state["plan"] if isinstance(state["plan"], dict) else {})
        progress.think("next", "Running with user-approved research plan")

    if autonomy == "L2" and not approved_plan:
        try:
            from src.engine.temporal.activities import register_approval_request
            register_approval_request("plan", {"query": query, "mode": mode, "autonomy": autonomy})
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)

    graph = build_graph()

    # Per-run cost/token accounting: route every LLM call made on this thread
    # straight into this run's budgets (replaces the race-prone global-metrics
    # baseline approach for runs that go through here).
    from src.llm import set_run_cost_sink, clear_run_cost_sink

    def _run_cost_sink(prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        b = state.setdefault("budgets", {})
        b["tokens_used"] = int(b.get("tokens_used") or 0) + int(prompt_tokens) + int(completion_tokens)
        b["spent_usd"] = float(b.get("spent_usd") or 0) + float(cost_usd)

    set_run_cost_sink(_run_cost_sink)
    from src.llm import set_run_request_context, clear_run_request_context
    set_run_request_context(model=model, max_tokens=int(max_tokens or dial.max_tokens_per_call))
    try:
        result = graph.invoke(state)
        if mode_config.structured_output or mode == "compare":
            result = _ensure_compare_structure(result)
        result = _ensure_sources_at_end(result)
        progress.update(
            stage="complete",
            finished=True,
            status="Research complete",
            findings_count=len(result.get("findings", [])),
            factoids_count=len(result.get("factoids", [])),
            sources_count=len(result.get("evidence_map") or {}),
            sections=result.get("sections", []),
            report=result.get("report", ""),
            markdown_path=result.get("markdown_path", ""),
        )
        progress.think("learned", f"Report complete: {len(result.get('report') or '')} chars")
        if job_id:
            from src.engine.jobs import get_jobs
            get_jobs().update(
                job_id,
                status="complete",
                finished_at=time.time(),
                report=result.get("report", ""),
                markdown_path=result.get("markdown_path", ""),
                findings_count=len(result.get("findings") or []),
                sources_count=len(result.get("evidence_map") or {}),
                iterations=result.get("iteration", 0),
            )
        if plan_id:
            try:
                from src.engine.plan_store import get_plans
                get_plans().update(plan_id, status="complete", job_id=job_id)
            except Exception:
                logging.getLogger(__name__).debug("ignored error", exc_info=True)
        return result
    except Exception as e:
        logging.getLogger(__name__).exception("Research graph failed: %s", e)
        progress.update(stage="error", finished=True, error="Research failed; see server logs")
        if job_id:
            from src.engine.jobs import get_jobs
            if not get_jobs().is_cancelled(job_id):
                get_jobs().update(
                    job_id, status="error", error="Research failed; see server logs",
                    finished_at=time.time(),
                )
        raise
    finally:
        clear_run_cost_sink()
        clear_run_request_context()


def _ensure_compare_structure(state: ResearchState) -> ResearchState:
    sections = list(state.get("sections") or [])
    titles = {s.get("title", "").lower() for s in sections}
    if any("compar" in t for t in titles):
        return state
    findings = state.get("findings") or []
    rows = []
    for i, f in enumerate(findings[:8]):
        rows.append(f"| Aspect {i+1} | {str(f)[:120].replace('|', '/')} | medium |")
    table = (
        "## Comparison Matrix\n\n"
        "| Aspect | Summary | Confidence |\n"
        "|--------|---------|------------|\n"
        + ("\n".join(rows) if rows else "| — | No findings | low |\n")
        + "\n"
    )
    insert_at = len(sections)
    for i, s in enumerate(sections):
        if s.get("title", "").lower() in ("sources", "references"):
            insert_at = i
            break
    sections.insert(insert_at, {"title": "Comparison Matrix", "content": table, "sources": []})
    state["sections"] = sections
    if state.get("report"):
        state["report"] = state["report"] + "\n\n" + table
    return state


def _ensure_sources_at_end(state: ResearchState) -> ResearchState:
    """Guarantee a Sources section is last and this-run URLs are listed.

    Uses the same ordering as the compiler's `_collect_run_urls` so inline
    citation numbers stay consistent with the final Sources list.
    """
    from src.engine.agents.compiler import _collect_run_urls

    sections = list(state.get("sections") or [])
    run_urls = _collect_run_urls(state)
    urls = [u for u, _ in run_urls]

    # Remove existing sources sections then re-append clean one
    body = [s for s in sections if s.get("title", "").lower() not in ("sources", "references")]
    lines = ["# Sources\n"]
    if urls:
        for i, (url, title) in enumerate(run_urls[:50], 1):
            lines.append(f"[{i}] [{title or url}]({url})")
    else:
        lines.append("_No external sources were retained for this run._")
    body.append({"title": "Sources", "content": "\n".join(lines), "sources": urls})
    state["sections"] = body

    # Rebuild report ending with sources if report exists
    if state.get("report"):
        # strip trailing sources-ish and append
        report = state["report"]
        # simple append if Sources not last
        if "## Sources" not in report[-2000:]:
            state["report"] = report.rstrip() + "\n\n" + "\n".join(lines) + "\n"
    return state
