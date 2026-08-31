"""
Planner agent — decomposes the user query into a structured research plan.

Outputs:
  - Topic identification
  - Subtopics
  - Report outline (section titles + initial search queries)
  - Source type preferences
  - First wave of search queries
"""

import logging
import json
import re

from src.llm import call_llm
from src.jsonutil import parse_json_dict
from src.state import ResearchState
from .registry import register

PLANNER_SYSTEM = (
    "You are an expert research planner. Your job is to decompose a research query "
    "into a structured plan with an outline, subtopics, and search queries. "
    "Always return valid JSON. Be thorough and specific."
)


def classify_query_type(query: str, mode: str = "") -> str:
    """Classify a query into depth_first / breadth_first / straightforward.

    Anthropic's multi-agent research finding: 80% of quality variance is
    token budget — matching the plan shape to the query type is the lever.
    Deterministic (no LLM call): comparisons/surveys are breadth-first,
    simple fact lookups are straightforward, everything else is a deep dive.
    """
    q = (query or "").lower()
    if mode == "compare" or any(
        w in q
        for w in (
            "compare", "versus", " vs ", "differences between",
            "which is better", "pros and cons", "alternatives to",
        )
    ):
        return "breadth_first"
    if any(
        w in q
        for w in (
            "survey", "overview", "state of", "landscape", "list of",
            "taxonomy", "comprehensive", "top 10", " best", "trends in",
        )
    ):
        return "breadth_first"
    # Short single-fact lookups ("what is X", "who invented Y") are cheap
    if re.match(
        r"^(what is|who is|who invented|when did|when was|where is|"
        r"how many|what year|define)\s+",
        q,
    ) and len(q) < 90:
        return "straightforward"
    if q.endswith("?") and len(q.split()) <= 7 and not any(
        w in q for w in ("how does", "why does", "mechanism", "works")
    ):
        return "straightforward"
    return "depth_first"


_TYPE_HINTS = {
    "depth_first": (
        "QUERY TYPE: depth-first — investigate a narrow topic deeply. "
        "Prefer FEW subtopics with heavy mechanisms, named systems/papers, and "
        "evidence per section."
    ),
    "breadth_first": (
        "QUERY TYPE: breadth-first — cover many aspects broadly. "
        "Prefer MANY subtopics, broad coverage, and a comparison/taxonomy structure."
    ),
    "straightforward": (
        "QUERY TYPE: straightforward — answer directly. "
        "Keep the plan compact (2-3 sections) and factual."
    ),
}


@register("planner")
def planner(state: ResearchState) -> ResearchState:
    """Analyze the query and generate a research plan with outline and queries.

    If state already has an approved plan (plan_approved / skip_planning), reuse it.
    """
    # Resume from user-edited / approved plan
    if state.get("plan_approved") and state.get("plan"):
        plan = state["plan"]
        state["search_queries"] = list(
            state.get("search_queries") or plan.get("search_queries") or [state["query"]]
        )[:8]
        if not state.get("outline"):
            state["outline"] = [
                {
                    "title": s.get("title", f"Section {i+1}"),
                    "order": i,
                    "task_id": f"T{i+1}",
                }
                for i, s in enumerate(plan.get("outline", []))
            ]
        else:
            # Ensure task_ids on pre-existing outline entries (approved-plan path)
            for i, s in enumerate(state["outline"]):
                if isinstance(s, dict) and not s.get("task_id"):
                    s["task_id"] = f"T{i+1}"
        if not state.get("query_type"):
            state["query_type"] = classify_query_type(
                state.get("query", ""), mode=state.get("mode", "")
            )
        state["status"] = f"Using approved plan: {len(state.get('outline') or [])} sections"
        print(f"\n🧠 [Planner] Using approved plan ({len(state['search_queries'])} queries)")
        try:
            from src.engine.progress import get_progress
            get_progress().update(stage="planning", status=state["status"], plan=plan)
            get_progress().think("next", "Approved plan — starting research gather")
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)
        return state

    state["status"] = "Planning research..."
    try:
        from src.engine.progress import get_progress
        get_progress().update(stage="planning", status=state["status"])
        get_progress().think("next", "Decomposing query into research plan")
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)
    print(f"\n🧠 [Planner] Analyzing query: {state['query'][:80]}")

    # ── Query-type classification (Anthropic) → drives plan shape + budgets ──
    qtype = classify_query_type(state.get("query", ""), mode=state.get("mode", ""))
    state["query_type"] = qtype
    print(f"  Query type: {qtype}")

    flags = state.get("mode_flags") or {}
    structured = bool(flags.get("structured_output")) or state.get("mode") == "compare"
    deep = (state.get("mode") or "") in ("deep", "academic", "ultra-long")
    compare_extra = ""
    if structured:
        compare_extra = """
This is a COMPARE / structured mode query. Outline MUST include:
  - Criteria / Evaluation Framework
  - Option A deep dive
  - Option B deep dive (and C if relevant)
  - Head-to-head Comparison Matrix
  - Recommendation / Trade-offs
  - Sources
Search queries should target each option and direct comparisons (A vs B).
"""
    deep_extra = ""
    if deep:
        deep_extra = """
DEEP mode: outline should name real systems/papers where possible, and include
Evaluation Matrix + Failure-Mode Taxonomy sections before Sources.
"""
    clarif = ""
    if state.get("clarifications"):
        clarif = f"\nUser clarifications: {json.dumps(state.get('clarifications'))}\n"

    scout = state.get("scout") or {}
    scout_extra = ""
    if scout:
        scout_extra = f"""
SCOUT HINTS (from pre-research thinker + web):
  refined_query: {scout.get('refined_query', '')}
  must_cover_systems: {json.dumps(scout.get('must_cover_systems') or [])}
  must_cover_papers: {json.dumps((scout.get('must_cover_papers') or [])[:8])}
  eval_axes: {json.dumps(scout.get('eval_axes') or [])}
  failure_modes: {json.dumps(scout.get('failure_modes') or [])}
  outline_hints: {json.dumps(scout.get('outline_hints') or [])}
  seeded_queries: {json.dumps((state.get('search_queries') or [])[:6])}
REQUIRE: outline/sections should name key systems from must_cover_systems when relevant.
REQUIRE: include Evaluation Matrix and Failure-Mode Taxonomy for deep/academic modes.
"""

    type_extra = _TYPE_HINTS.get(qtype, "")

    from src.engine.budget import budget_status_line
    budget_line = budget_status_line(state)
    prompt = f"""Analyze this research query and create a structured plan.

Query: "{state['query']}"
{budget_line}
{type_extra}
{clarif}{scout_extra}{compare_extra}{deep_extra}
Return a JSON object with:
  - "topic": main topic (string)
  - "subtopics": 3-5 key subtopics to investigate (list of strings)
  - "outline": list of section objects with "title" and "queries" (list of search queries for that section)
  - "source_types": recommended source types (e.g. "academic", "news", "documentation")
  - "search_queries": first wave of 3-5 specific search queries
  - "rationale": brief why this plan covers the query

Example outline entry:
  {{"title": "Historical Context", "queries": ["history of X", "origins of X"]}}"""

    result = call_llm(PLANNER_SYSTEM, prompt)
    plan = parse_json_dict(result, default=None)
    if not plan:
        plan = {
            "topic": state["query"],
            "subtopics": [],
            "outline": [{"title": "Overview", "queries": [state["query"]]}],
            "source_types": ["web"],
            "search_queries": [state["query"]],
        }

    state["plan"] = plan
    plan["query_type"] = qtype
    # Prefer plan queries; keep scout seeds if plan weak
    planned_q = plan.get("search_queries") or []
    if planned_q:
        state["search_queries"] = planned_q[:6]
    elif not state.get("search_queries"):
        state["search_queries"] = [state["query"]]
    else:
        state["search_queries"] = list(state.get("search_queries") or [state["query"]])[:6]
    # Task-id ledger (langgraph-deep-research): every plan section gets a stable id
    state["outline"] = [
        {
            "title": s.get("title", f"Section {i+1}"),
            "order": i,
            "task_id": f"T{i+1}",
        }
        for i, s in enumerate(plan.get("outline", []))
    ]
    state["findings"] = [f"Research topic: {plan.get('topic', state['query'])}"]

    # ── Per-type budget dials (Anthropic: 80% of variance is token budget) ──
    budgets = state.setdefault("budgets", {})
    quality = state.setdefault("quality", {})
    if qtype == "straightforward":
        budgets["max_iterations"] = min(int(budgets.get("max_iterations") or 6), 2)
        state["max_iterations"] = min(int(state.get("max_iterations") or 6), 2)
        budgets["max_tool_calls"] = min(int(budgets.get("max_tool_calls") or 25), 12)
        quality["max_search_results"] = min(int(quality.get("max_search_results") or 10), 6)
        quality["max_extract_pages"] = min(int(quality.get("max_extract_pages") or 8), 4)
        print("  Budgets (straightforward): 2 iters max, slim search/extract")
    elif qtype == "breadth_first":
        budgets["max_iterations"] = min(int(budgets.get("max_iterations") or 6), 3)
        state["max_iterations"] = min(int(state.get("max_iterations") or 6), 3)
        budgets["max_tool_calls"] = max(int(budgets.get("max_tool_calls") or 25), 30)
        print("  Budgets (breadth-first): 3 iters max, wider tool budget")
    else:
        print("  Budgets (depth-first): keep mode defaults (fewer queries, deeper iterations)")
    state["status"] = f"Plan: {len(state['outline'])} sections, {len(state['search_queries'])} queries"
    print(f"  Outline: {[s['title'] for s in state['outline']]}")
    print(f"  Initial queries: {state['search_queries']}")
    try:
        from src.engine.progress import get_progress
        get_progress().update(stage="planning", status=state["status"], plan=plan)
        get_progress().think("learned", f"Plan topic: {plan.get('topic', '')[:120]}")
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)
    return state
