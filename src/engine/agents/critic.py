"""
Critic agent — research quality gate + off-topic hard fail + re-plan signals.

After critic, graph runs thinker_search_strategy to propose better web queries
when more research is needed.
"""

from __future__ import annotations

import logging

import json
import re

from src.llm import call_llm
from src.jsonutil import parse_json_dict
from src.state import ResearchState
from .registry import register

CRITIC_SYSTEM = (
    "You are a strict research quality evaluator for a Deep Research agent. "
    "Detect off-topic contamination. Prefer evidence-backed findings. "
    "Return valid JSON only."
)


def _topic_keywords(query: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "how",
        "does", "what", "with", "as", "by", "is", "are", "be", "from", "that",
        "this", "into", "about", "cover", "methods", "best", "practices",
    }
    words = re.findall(r"[a-zA-Z]{3,}", query.lower())
    return {w for w in words if w not in stop}


def _run_source_urls(state: ResearchState) -> list[str]:
    """URLs actually fetched/read this run — not LLM-invented evidence_map keys."""
    from src.urlutil import canonical_url

    seen: set[str] = set()
    out: list[str] = []
    for bag in (
        state.get("run_corpus") or [],
        state.get("retrieved_chunks") or [],
        state.get("extracted_pages") or [],
        state.get("search_results") or [],
    ):
        for row in bag:
            u = canonical_url((row or {}).get("url") or "")
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    if not out:
        for u in (state.get("evidence_map") or {}):
            cu = canonical_url(u)
            if cu and cu not in seen:
                seen.add(cu)
                out.append(cu)
    return out


def _findings_on_topic(query: str, findings: list[str]) -> tuple[bool, float]:
    """Heuristic: fraction of findings that share topic keywords with query."""
    kws = _topic_keywords(query)
    if not findings or not kws:
        return True, 1.0
    hits = 0
    for f in findings:
        fl = f.lower()
        if sum(1 for k in kws if k in fl) >= min(2, len(kws)):
            hits += 1
    ratio = hits / max(len(findings), 1)
    # Off-topic if almost no findings match core query terms
    return ratio >= 0.25, ratio


@register("critic")
def critic(state: ResearchState) -> ResearchState:
    """Evaluate research completeness; hard-fail off-topic junk."""
    from src.engine.budget import check_budgets, force_complete, sync_cost_from_metrics

    sync_cost_from_metrics(state)
    ok, reason = check_budgets(state)
    if not ok:
        print(f"\n🔎 [Critic] Budget force-complete: {reason}")
        return force_complete(state, reason)

    max_iter = state.get("max_iterations", 6)
    iteration = int(state.get("iteration") or 0)
    findings = list(state.get("findings") or [])
    gaps = list(state.get("gaps") or [])
    query = state.get("query", "")
    urls = _run_source_urls(state)

    state["status"] = f"Evaluating research ({iteration}/{max_iter})..."
    try:
        from src.engine.progress import get_progress
        p = get_progress()
        p.update(
            stage="evaluating",
            status=state["status"],
            iteration=iteration,
            findings_count=len(findings),
            sources_count=len(urls),
        )
        p.think("next", "Critic reviewing findings vs query")
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)
    print(f"\n🔎 [Critic] Evaluating iteration {iteration}/{max_iter}")

    # ── Hard off-topic gate (P0.2) ──
    # Use findings + titles + urls (arxiv.org/abs/… has no topic words in path)
    on_topic, ratio = _findings_on_topic(query, findings)
    query_kws = _topic_keywords(query)
    source_blob_parts: list[str] = list(urls)
    for c in state.get("retrieved_chunks") or []:
        source_blob_parts.append(str(c.get("title") or ""))
        source_blob_parts.append(str(c.get("url") or ""))
        source_blob_parts.append(str(c.get("text") or "")[:400])
    for r in state.get("search_results") or []:
        source_blob_parts.append(str(r.get("title") or ""))
        source_blob_parts.append(str(r.get("url") or ""))
    source_blob = " ".join(source_blob_parts).lower()
    url_hits = sum(1 for k in query_kws if k in source_blob)
    # Need a few keyword hits across corpus, not just bare URL paths
    sources_on_topic = url_hits >= min(2, max(1, len(query_kws) // 4)) or not (
        urls or state.get("retrieved_chunks") or state.get("search_results")
    )

    # Only hard-fail when findings are clearly off AND sources also look wrong
    hard_off_topic = bool(findings) and (not on_topic) and (not sources_on_topic)
    soft_off_topic = bool(findings) and (not on_topic) and sources_on_topic
    if hard_off_topic:
        print(f"  ⛔ Off-topic detected (findings_ratio={ratio:.2f}, source_kw_hits={url_hits})")
        state["off_topic"] = True
        state["needs_more_research"] = True
        state["replan"] = True
        # Drop contaminated findings so synthesizer cannot use them
        state["findings"] = [f for f in findings if _findings_on_topic(query, [f])[0]]
        state["claims"] = []
        # Reset marginal-value counters: claims were just cleared, so the next
        # critic round must start from a clean slate — otherwise saturation
        # would compute new_claims=0 instantly and force a premature
        # completion on a single fresh iteration of on-topic content.
        state["_marginal_prev_claims"] = 0
        state["_marginal_prev_urls"] = len(urls)
        core = " ".join(list(query_kws)[:8])
        state["search_queries"] = [
            f"{core} site:arxiv.org",
            f"{query[:120]} mechanisms evaluation production",
            f"{core} survey review 2024 2025 2026",
            f"{core} best practices limitations",
        ]
        try:
            from src.engine.progress import get_progress
            get_progress().update(off_topic=True, next_action="Hard re-search: off-topic contamination")
            get_progress().think("gap", f"Off-topic contamination ratio={ratio:.2f}")
            get_progress().think("next", "Re-search with arXiv + focused queries")
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)
        if iteration >= max_iter:
            state["needs_more_research"] = False
            state["abort_synthesis"] = True
            state["error"] = "Research aborted: could not gather on-topic evidence"
            state["status"] = "Aborted: off-topic evidence only"
            print("  🛑 Abort synthesis — max iters with off-topic evidence")
            return state
        state["status"] = "Off-topic — forcing re-search"
        return state

    if soft_off_topic:
        print(f"  ⚠️  Findings weak on topic (ratio={ratio:.2f}) but sources look relevant — continue")
        state["needs_more_research"] = True if iteration < max_iter else False

    # ── Marginal-value stop (OverSearchQA/SAAS) ──
    # If this iteration added almost nothing new over the previous one, the
    # research has saturated — force completion instead of burning budget on
    # over-searching. Only fires once we already have a REASONABLE evidence
    # base: a minimum number of findings and URLs, and at least 3 iterations.
    # This guards against the thin-run failure seen in the 15-topic benchmark
    # (T10/T11/T14 exited with 5-6 sections / ~50K chars because saturation
    # fired at iteration 2 on a 3-URL corpus — especially when the claims
    # extractor returned 0 claims, making new_claims < 2 trivially true).
    prev_claims = int(state.get("_marginal_prev_claims") or 0)
    prev_urls = int(state.get("_marginal_prev_urls") or 0)
    new_claims = max(0, len(state.get("claims") or []) - prev_claims)
    new_urls = max(0, len(urls) - prev_urls)
    has_evidence_base = len(findings) >= 8 and len(urls) >= 6
    saturated = (
        iteration >= 3
        and has_evidence_base
        and new_claims < 2
        and new_urls < 2
        and not state.get("replan")  # never force-complete during a replan cycle
    )
    # If we have almost no evidence yet, saturation must NOT fire — the run
    # needs to keep searching regardless of marginal gains (over-searching is
    # the lesser failure than shipping a 5-finding report).
    if not has_evidence_base and iteration >= 3:
        print(f"  ⚠️  Low evidence base (findings={len(findings)}, urls={len(urls)}) "
              f"— saturation suppressed, continuing research")
    state["_marginal_prev_claims"] = len(state.get("claims") or [])
    state["_marginal_prev_urls"] = len(urls)

    state["off_topic"] = False
    findings_text = "\n".join(f"- {f}" for f in findings)
    gaps_text = "\n".join(f"- {g}" for g in gaps)
    outline_titles = [s.get("title", "") for s in state.get("outline", [])]

    # RE-TRAC context: open hypotheses from earlier iterations
    memory = state.get("research_memory") or {}
    open_hyp = list(memory.get("open_hypotheses") or [])[:5]
    open_hyp_text = "\n".join(f"- {h}" for h in open_hyp) if open_hyp else "(none)"

    # Task-ledger coverage summary (langgraph-deep-research): which plan
    # sections already have findings, so the critic can target the bare ones.
    ledger = state.get("task_ledger") or []
    led: dict[str, int] = {}
    for e in ledger:
        tid = str(e.get("task_id") or "")
        led[tid] = int(led.get(tid, 0)) + 1
    coverage = ", ".join(f"{k}:{v}" for k, v in led.items()) or "(no ledger yet)"

    from src.engine.budget import budget_status_line
    budget_line = budget_status_line(state)

    prompt = f"""Evaluate if the research is complete enough to write a publication-grade report.

Query: "{query}"
{budget_line}
Iteration: {iteration}/{max_iter}
Expected sections: {outline_titles}
Section coverage (task_id:findings): {coverage}
Evidence URLs: {len(urls)}
Findings ({len(findings)}):
{findings_text[:3000]}

Gaps:
{gaps_text[:800]}

Open hypotheses (still to resolve):
{open_hyp_text}

Return JSON:
  - "complete": true/false
  - "reason": brief explanation
  - "confidence": "high"|"medium"|"low"
  - "off_topic": true if findings are unrelated to the query
  - "replan": true if the research plan/outline itself should change
  - "gap_queries": 2-5 NEW search queries if not complete
  - "missing_facts": list of {{ "fact": what's missing, "sub_topic": which section/sub-question, "suggested_queries": [1-3 new queries] }}
  - "learned": 1-3 short bullets of what we now know
  - "missing": 1-3 short bullets of what is still missing
"""

    if saturated:
        reason = (
            f"Marginal-value saturation (iter {iteration}: +{new_claims} claims, "
            f"+{new_urls} URLs over previous round)"
        )
        print(f"  ⏹️  {reason} — forcing completion")
        state["needs_more_research"] = False
        state["replan"] = False
        state["status"] = f"Evaluation: complete ({reason[:80]})"
        try:
            from src.engine.progress import get_progress
            get_progress().think("learned", reason[:160])
            get_progress().think("next", "Proceed to triangulation and synthesis")
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)
        return state

    result = call_llm(CRITIC_SYSTEM, prompt, model="fast")
    evaluation = parse_json_dict(result)
    if not evaluation:
        evaluation = {
            "complete": len(findings) >= 5 and len(urls) >= 3,
            "reason": "JSON parse failed — heuristic complete",
            "confidence": "low",
            "gap_queries": [],
            "missing_facts": [],
            "off_topic": False,
            "replan": False,
        }

    if evaluation.get("off_topic"):
        state["off_topic"] = True
        state["needs_more_research"] = True
        state["replan"] = True
        state["search_queries"] = evaluation.get("gap_queries") or [
            f"{query} site:arxiv.org",
            query,
        ]
        print(f"  ⛔ LLM critic marked off-topic: {evaluation.get('reason')}")
        if iteration >= max_iter:
            # Reconcile the LLM flag (which only sees the LAST iteration's
            # findings) with the deterministic source-topicality signal. If the
            # gathered SOURCES are clearly on-topic, the LLM flag is a weak
            # final draw — complete with the real evidence instead of aborting.
            if sources_on_topic:
                state["off_topic"] = False
                state["needs_more_research"] = False
                state["replan"] = False
                print(
                    "  ✅ Sources on-topic — completing with gathered evidence "
                    "(LLM findings-only flag overridden)"
                )
            else:
                state["needs_more_research"] = False
                state["abort_synthesis"] = True
                state["error"] = "Research aborted: off-topic"
        return state

    is_complete = bool(evaluation.get("complete", False))
    reason = evaluation.get("reason", "")

    if iteration >= max_iter:
        is_complete = True
        reason = f"Reached max iterations ({max_iter})"

    # Need minimum evidence to complete
    if is_complete and len(urls) < 2 and iteration < max_iter:
        is_complete = False
        reason = "Too few evidence URLs — continue search"
        evaluation["gap_queries"] = evaluation.get("gap_queries") or [query]

    state["needs_more_research"] = not is_complete
    state["replan"] = bool(evaluation.get("replan", False))

    # ── Structured missing_facts (r1-reasoning-rag: COMPLETE/INCOMPLETE) ──
    # Explicit "what's missing → which sub-topic → suggested queries" so the
    # search strategy re-searches targeted, not generically.
    missing_facts: list[dict] = []
    for mf in (evaluation.get("missing_facts") or []):
        if isinstance(mf, dict) and (mf.get("fact") or mf.get("sub_topic")):
            missing_facts.append({
                "fact": str(mf.get("fact") or mf.get("sub_topic") or "")[:200],
                "sub_topic": str(mf.get("sub_topic") or "")[:120],
                "suggested_queries": [
                    str(q) for q in (mf.get("suggested_queries") or [])[:4] if q
                ],
            })
    # Fallback: mechanically derive from gaps if the LLM gave nothing structured
    if not missing_facts and gaps:
        for g in gaps[:6]:
            missing_facts.append({"fact": str(g)[:200], "sub_topic": "", "suggested_queries": []})
    # Per-section coverage: sections with zero findings are gaps by construction
    covered_tasks = {str(e.get("task_id") or "") for e in ledger}
    for s in (state.get("outline") or []):
        tid = str(s.get("task_id") or "")
        title = str(s.get("title") or f"Section {tid}")
        if (
            tid and tid not in covered_tasks
            and title.lower() not in ("sources", "references")
            and not any(mf.get("sub_topic") == title for mf in missing_facts)
        ):
            missing_facts.append({
                "fact": f"No evidence gathered yet for section: {title}",
                "sub_topic": title[:120],
                "suggested_queries": [f"{query} {title}"],
            })
    state["missing_facts"] = missing_facts[:12]

    if not is_complete:
        next_queries = list(evaluation.get("gap_queries") or [])
        for mf in missing_facts[:4]:
            next_queries.extend(mf.get("suggested_queries") or [])
        state["search_queries"] = (next_queries or [query])[:6]
        print(f"  🔄 More needed: {reason}")
        print(f"  Missing facts: {len(missing_facts)} | Next queries: {state['search_queries']}")
    else:
        print(f"  ✅ Complete: {reason} | missing_facts={len(missing_facts)}")

    # Thinking panel
    try:
        from src.engine.progress import get_progress
        p = get_progress()
        for item in evaluation.get("learned") or []:
            p.think("learned", str(item))
        for item in evaluation.get("missing") or gaps[:3]:
            p.think("gap", str(item))
        if not is_complete:
            p.think("next", f"Search: {state.get('search_queries', [''])[0][:120]}")
        else:
            p.think("next", "Proceed to triangulation and synthesis")
        p.update(
            status=f"Evaluation: {'complete' if is_complete else 'needs more'}",
            findings_count=len(state.get("findings") or []),
            sources_count=len(urls),
        )
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)

    state["status"] = f"Evaluation: {'complete' if is_complete else 'needs more'}"
    return state
