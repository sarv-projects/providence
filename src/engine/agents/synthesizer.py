"""
Synthesizer agent — writes the report section by section using retrieved RAG claims.

Progressive output pattern:
  1. Generate final outline from findings + plan
  2. Write sections IN PARALLEL for maximum speed AND maximum section depth/token allocation
  3. Verification & audit pass over assembled draft
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re

from src.llm import call_llm_strong, call_llm
from src.jsonutil import parse_json_dict, parse_json_list
from src.rag.pipeline import retrieve_chunks
from src.rag.hybrid import hybrid_retrieve
from src.state import ResearchState, Section
from .registry import register

SYNTH_SYSTEM = (
    "You are a principal research scientist and technical report author. "
    "Write exhaustive, highly detailed, evidence-backed research report sections. "
    "Always include ASCII architecture flowcharts/diagrams, Markdown comparative evaluation tables, "
    "and LaTeX mathematical equations ($...$) where relevant. Use inline citations like [1], [2] "
    "referencing sources. Ensure technical rigor, structural depth, and clarity."
)


@register("synthesizer_outline")
def synthesizer_outline(state: ResearchState) -> ResearchState:
    """Generate the final report outline from findings and plan."""
    state["status"] = "Generating report outline..."
    print(f"\n✍️ [Synthesizer] Creating outline")

    findings_text = "\n".join(f"- {f}" for f in state.get("findings", [])[:30])
    plan_outline = state.get("plan", {}).get("outline", [])
    plan_titles = [s.get("title", "") for s in plan_outline]
    flags = state.get("mode_flags") or {}
    mode = state.get("mode") or "standard"
    deep = mode in ("deep", "academic", "ultra-long")
    structured = bool(flags.get("structured_output")) or mode == "compare"
    scout = state.get("scout") or {}
    must_sys = scout.get("must_cover_systems") or []
    must_papers = scout.get("must_cover_papers") or []
    compare_hint = ""
    if structured:
        compare_hint = (
            "COMPARE MODE: include Criteria, Option A, Option B, Comparison Matrix, "
            "Recommendation, Sources.\n"
        )
    deep_hint = ""
    if deep:
        deep_hint = (
            "DEEP MODE (ChatGPT/Gemini RAG report style):\n"
            "- Start with Executive Summary\n"
            "- Body sections MUST name specific systems/papers (e.g. Dense Passage Retrieval, "
            "ColBERT, RAPTOR, Self-RAG, GraphRAG, HyDE, CRAG)\n"
            "- Include Evaluation Matrix (RAG triad: context relevance, faithfulness, answer "
            "relevance + tools)\n"
            "- Include Failure-Mode Taxonomy (hallucination-on-hallucination, lost-in-middle, "
            "retrieval cascade failures, citation fabrication, etc.)\n"
            "- End with Conclusion then Sources\n"
        )

    from src.engine.budget import budget_status_line
    budget_line = budget_status_line(state)
    prompt = f"""Create an exhaustive report outline based on the research findings.

Query: "{state['query']}"
{budget_line}
Planned sections: {plan_titles}
{compare_hint}{deep_hint}
Key findings:
{findings_text[:4000]}

MUST COVER SYSTEMS (from scout): {json.dumps(must_sys[:15])}
MUST COVER PAPERS (from scout): {json.dumps(must_papers[:10])}

Return a JSON list of section objects with "title" and "order".
Include: Executive Summary (if deep), Introduction, 4-6 detailed body sections naming real systems/papers,
Evaluation Matrix (if deep), Failure Modes (if deep), Conclusion, Sources.
Example: [{{"title": "Introduction", "order": 0}}, ...]"""

    result = call_llm_strong(SYNTH_SYSTEM, prompt)
    outline = parse_json_list(result, default=None)
    if not isinstance(outline, list):
        outline = [{"title": "Overview", "order": 0}, {"title": "Findings", "order": 1},
                   {"title": "Sources", "order": 2}]

    # Ensure required deep sections exist
    titles_l = {str(s.get("title", "")).lower() for s in outline}
    if deep:
        required = [
            ("Executive Summary", 0),
            ("Evaluation Matrix", 50),
            ("Failure-Mode Taxonomy", 51),
            ("Conclusion", 90),
            ("Sources", 99),
        ]
        for title, order in required:
            if not any(title.lower() in t for t in titles_l):
                outline.append({"title": title, "order": order})
                titles_l.add(title.lower())
        outline = sorted(outline, key=lambda s: s.get("order", 0))
    elif len(outline) < 8 and not state.get("mode") == "quick":
        # Deterministic minimum-section floor (thin-run fix). The outline LLM
        # sometimes returns a minimal 4-5 section list for standard mode — that
        # produced the 5-6-section / ~50K-char thin reports in the benchmark
        # (T10/T11/T14). Backfill from the planner's outline so every standard
        # run ships a full report regardless of LLM variance.
        plan_outline = state.get("plan", {}).get("outline", [])
        for s in plan_outline:
            t = str(s.get("title") or "").strip()
            tl = t.lower()
            if (
                t
                and len(outline) >= 8
            ):
                break
            if t and tl not in titles_l and tl not in ("sources", "references"):
                outline.append({"title": t, "order": 10 + len(outline)})
                titles_l.add(tl)
        if "introduction" not in titles_l:
            outline.insert(0, {"title": "Introduction", "order": 1})
            titles_l.add("introduction")
        if "conclusion" not in titles_l:
            outline.append({"title": "Conclusion", "order": 95})
            titles_l.add("conclusion")
        outline = sorted(outline, key=lambda s: s.get("order", 0))

    # Compiler appends Bedrock + Research Debt + Sources; avoid duplicate Sources here only
    # Still put Sources last in outline for write-path auto-sources
    drop = {"sources", "references", "research debt", "evidence bedrock", "bedrock"}
    outline = [s for s in outline if str(s.get("title", "")).lower().strip() not in drop]
    outline.append({"title": "Sources", "order": 999})

    state["outline"] = outline
    print(f"  Outline: {[s['title'] for s in outline]}")

    state["sections"] = [
        {"title": s["title"], "content": "", "sources": []}
        for s in outline
    ]
    _update_progress(state, "synthesizing_outline", sections=state["sections"])
    return state


@register("synthesizer_write")
def synthesizer_write(state: ResearchState) -> ResearchState:
    """Write all body sections using parallel multi-agent section synthesis + audit verification."""
    outline = state.get("outline", [])
    if not outline:
        return state

    body_sections = [(i, s) for i, s in enumerate(outline)
                     if s["title"].lower() not in ("sources", "references")]
    source_sections = [(i, s) for i, s in enumerate(outline)
                       if s["title"].lower() in ("sources", "references")]

    print(f"\n✍️ [Synthesizer] Writing {len(outline)} sections ({len(body_sections)} in parallel + {len(source_sections)} sources)")

    factoids = state.get("factoids", [])

    # ── Auto-generate Sources section (no LLM call) ──
    for idx, section_def in source_sections:
        all_urls: list[str] = []
        seen: set[str] = set()
        for c in state.get("retrieved_chunks", []):
            url = c.get("url", "")
            if url and url not in seen:
                all_urls.append(url)
                seen.add(url)
        for c in state.get("claims", []):
            for url in c.get("evidence_ids", []):
                if url and url not in seen:
                    all_urls.append(url)
                    seen.add(url)

        sources_content = "# Sources\n\n"
        for j, url in enumerate(all_urls[:40]):
            title_str = next((
                c.get("title", url)
                for c in state.get("retrieved_chunks", [])
                if c.get("url") == url
            ), url)
            sources_content += f"[{j+1}] [{title_str}]({url})\n"
        if not all_urls:
            sources_content += "No external sources available.\n"

        state["sections"][idx]["content"] = sources_content
        state["sections"][idx]["sources"] = all_urls
        print(f"  [Sources] auto-generated ({len(all_urls)} URLs, 0 LLM calls)")

    # Execute Parallel Section Synthesis (High Depth + 15s Total Speed)
    if body_sections:
        _write_parallel_sections(state, body_sections, factoids)

    state["status"] = f"Wrote {len(outline)} sections"
    _update_progress(state, "synthesizing_done", sections=state["sections"])
    return state


def _write_single_section(
    state: ResearchState,
    idx: int,
    section_def: dict,
    factoids: list[dict],
) -> tuple[int, str, list[str]]:
    """Draft a single section in isolation with dedicated full token budget."""
    title = section_def["title"]
    section_query = f"{state['query']} {title}"
    run_id = state.get("run_id", "")
    chunks = hybrid_retrieve(section_query, k=8, factoids=factoids, run_id=run_id)
    chunk_text = "\n\n".join(
        f"[Source: {c.get('title','') or c.get('url','')}]\n{c.get('text','')[:1500]}"
        for c in chunks[:8]
    ) if chunks else "No specific sources found."

    # Preserve retrieval order so inline [k] markers map 1:1 to section sources
    all_urls: list[str] = []
    for c in chunks:
        u = c.get("url", "")
        if u and u not in all_urls:
            all_urls.append(u)

    findings_context = "\n".join(f"- {f}" for f in state.get("findings", [])[:15])
    claims_context = "\n".join(
        f"- {c.get('text','')[:250]}" for c in state.get("claims", [])[:10]
    )
    scout = state.get("scout") or {}
    must_sys = ", ".join(str(s) for s in (scout.get("must_cover_systems") or [])[:12])
    deep = (state.get("mode") or "") in ("deep", "academic", "ultra-long")
    title_l = title.lower()
    extra = ""
    if must_sys:
        extra += f"\nName these systems when relevant and supported: {must_sys}\n"
    if "executive" in title_l:
        extra = (
            "\nWrite a crisp executive summary (bullets + short paras). "
            "Cover: problem, key systems, main findings, recommendations.\n"
        )
    elif "evaluation" in title_l or "matrix" in title_l:
        extra = (
            "\nInclude a Markdown table for the RAG triad "
            "(Context Relevance | Faithfulness | Answer Relevance) "
            "plus tool/retrieval dimensions. Name real systems per cell.\n"
        )
    elif "failure" in title_l:
        extra = (
            "\nTaxonomy of failure modes: hallucination-on-hallucination, lost-in-middle, "
            "retrieval cascade, citation fabrication, domain shift, chunking artifacts. "
            "Use a Markdown table: Failure Mode | Symptom | Mitigation.\n"
        )
    elif deep:
        extra = (
            "\nName specific papers/systems with years where known. "
            "Prefer evidence from the provided sources; do not invent monograph titles.\n"
        )

    from src.engine.budget import budget_status_line
    budget_line = budget_status_line(state)
    prompt = f"""Write the "{title}" section of an exhaustive, publication-grade research report.

Query: "{state['query']}"
{budget_line}

Available source materials:
{chunk_text[:6000]}

Relevant findings:
{findings_context[:2500]}

Key claims:
{claims_context[:2000]}
{extra}
INSTRUCTIONS:
- Write an exhaustive, deep technical passage (5-8 detailed paragraphs for body sections; shorter OK for exec summary).
- Include ASCII architecture flowcharts/diagrams where relevant.
- Include Markdown comparison tables and LaTeX mathematical equations ($...$) where applicable.
- Use inline citations like [1], [2] matching the numbered source materials above.
- Do NOT invent a References / Bibliography / Sources section — the compiler adds Sources.
- Do NOT invent arXiv IDs, venues, or paper titles that are not in the source materials.
- Provide rigorous analysis, specific parameters, and real-world implementations.
- Only name systems/papers that appear in the source materials.
"""

    max_tok = None
    try:
        max_tok = int((state.get("quality") or {}).get("max_tokens_per_call") or 0) or None
    except Exception:
        max_tok = None
    try:
        content = call_llm_strong(SYNTH_SYSTEM, prompt, max_tokens=max_tok)
    except Exception as e:
        try:
            content = call_llm(SYNTH_SYSTEM, prompt, model="fast", max_tokens=max_tok)
        except Exception:
            content = f"### {title}\n\nTechnical analysis for {title} based on research findings:\n\n" + findings_context[:1000]

    return idx, content, all_urls


def _write_parallel_sections(
    state: ResearchState,
    body_sections: list[tuple[int, dict]],
    factoids: list[dict],
) -> None:
    """Draft all body sections concurrently in parallel threads, followed by an audit verification pass."""
    state["status"] = f"Writing {len(body_sections)} sections in parallel..."
    print(f"  🚀 Launching {len(body_sections)} parallel section generators...")

    with ThreadPoolExecutor(max_workers=min(len(body_sections), 6)) as executor:
        futures = {
            executor.submit(_write_single_section, state, idx, sd, factoids): (idx, sd["title"])
            for idx, sd in body_sections
        }
        for future in as_completed(futures):
            idx, title = futures[future]
            try:
                res_idx, content, urls = future.result()
                state["sections"][res_idx]["content"] = content
                state["sections"][res_idx]["sources"] = urls
                print(f"  ✅ Section '{title}' completed ({len(content)} chars)")
            except Exception as e:
                print(f"  ⚠️ Section '{title}' drafting failed: {e}")
                state["sections"][idx]["content"] = f"### {title}\n\nContent generation failed for {title}."

    # ── Audit & Verification Pass ──
    _audit_verification_pass(state, body_sections)

    # ── Multi-pass self-critique after full draft (P3.4) ──
    if (state.get("mode") or "") in ("deep", "academic", "ultra-long", "standard"):
        _self_critique_pass(state, body_sections)

    # ── Dynamic outline pruning (Tier-2 #16, ScaffoldAgent/WebWeaver) ──
    # Drop dead sections (no sources, no citations, tiny) and surface
    # discovered topics (findings bound to no plan section).
    pruned = _prune_dead_sections(state, body_sections)
    if pruned:
        print(f"  ✂️  Pruned {pruned} dead sections (no evidence, no citations)")

    # ── Interleaved draft→deepen loop (Tier-2 #15, AgentCPM-Report WARP) ──
    # One bounded deepen pass on the weakest surviving section.
    if (state.get("mode") or "") in ("deep", "academic", "ultra-long"):
        deepened = _deepen_weakest_section(state, body_sections)
        if deepened:
            print(f"  🔍 Deepened section: {deepened}")

    total_chars = sum(
        len(state["sections"][idx].get("content", ""))
        for idx, _ in body_sections
    )
    print(f"  Wrote {len(body_sections)} sections in parallel ({total_chars} chars total)")
    _update_progress(state, "writing_section", sections=state["sections"])


def _audit_verification_pass(state: ResearchState, body_sections: list[tuple[int, dict]]) -> None:
    """Audit section drafts: short content re-draft + citation coverage check."""
    claims = state.get("claims") or []
    claim_texts = [c.get("text", "")[:80] for c in claims if c.get("text")]

    for idx, section_def in body_sections:
        title = section_def["title"]
        content = state["sections"][idx].get("content", "")

        # Short sections → re-draft
        if len(content) < 300:
            print(f"  🔍 Audit Pass: Section '{title}' is short ({len(content)} chars) — re-drafting...")
            try:
                res_idx, new_content, urls = _write_single_section(
                    state, idx, section_def, state.get("factoids") or []
                )
                state["sections"][res_idx]["content"] = new_content
                if urls:
                    state["sections"][res_idx]["sources"] = urls
                content = new_content
            except Exception:
                logging.getLogger(__name__).debug("ignored error", exc_info=True)

        # Citation soft-check: body sections should have at least one [n] or URL
        has_cite = bool(
            "[" in content and "]" in content
        ) or "http" in content.lower()
        if not has_cite and claim_texts and title.lower() not in ("introduction", "overview"):
            # Append a lightweight evidence note from claims (no extra LLM call)
            note = "\n\n**Evidence notes:**\n" + "\n".join(
                f"- {t}" for t in claim_texts[:3]
            )
            state["sections"][idx]["content"] = content + note
            print(f"  🔍 Audit Pass: Section '{title}' missing citations — appended evidence notes")


def _self_critique_pass(state: ResearchState, body_sections: list[tuple[int, dict]]) -> None:
    """After full draft: one LLM pass to flag unsupported claims and tighten weakest section."""
    draft_preview = "\n\n".join(
        f"## {state['sections'][idx].get('title','')}\n{state['sections'][idx].get('content','')[:800]}"
        for idx, _ in body_sections[:6]
    )
    sources = []
    for c in (state.get("retrieved_chunks") or [])[:15]:
        if c.get("url"):
            sources.append(f"- {c.get('title','')}: {c.get('url')}")
    prompt = f"""You are a strict research editor. Review this draft for faithfulness.

Query: {state.get('query','')}
Available sources:
{chr(10).join(sources)[:2500]}

Draft excerpts:
{draft_preview[:5000]}

Return JSON:
  - "issues": list of short issues (unsupported claims, missing systems, thin sections)
  - "patch_section": title of the weakest section or ""
  - "patch_notes": how to improve that section (bullets)
  - "ok": true if draft is acceptable
"""
    try:
        raw = call_llm(
            "You are a research integrity editor. Return valid JSON only.",
            prompt,
            model="fast",
        )
        critique = parse_json_dict(raw)
    except Exception:
        print("  Self-critique skipped (parse/LLM failure)")
        return

    if not critique:
        print("  Self-critique skipped (empty parse)")
        return

    issues = critique.get("issues") or []
    if issues:
        print(f"  🔍 Self-critique: {len(issues)} issues — {issues[:2]}")
    patch_title = (critique.get("patch_section") or "").lower()
    notes = critique.get("patch_notes") or ""
    if patch_title and notes:
        for idx, sd in body_sections:
            if patch_title in (sd.get("title") or "").lower():
                content = state["sections"][idx].get("content", "")
                state["sections"][idx]["content"] = (
                    content
                    + "\n\n**Editor notes (self-critique):**\n"
                    + (notes if isinstance(notes, str) else "\n".join(f"- {n}" for n in notes))
                )
                print(f"  Self-critique patched section: {sd.get('title')}")
                break


def _prune_dead_sections(
    state: ResearchState,
    body_sections: list[tuple[int, dict]],
) -> int:
    """Drop dead sections and append discovered findings (Tier-2 #16).

    Dead = a body section whose written content is tiny and carries no sources
    and no inline citation. Such sections are evidence-empty — keeping them
    ships padding. Discovered topics (findings bound to no plan section) are
    surfaced as an "Additional Findings" section when there are enough of them.
    """
    pruned = 0
    keep: list[tuple[int, dict]] = []
    for idx, sd in body_sections:
        title = (sd.get("title") or "").lower()
        if title in ("sources", "references", "executive summary"):
            keep.append((idx, sd))
            continue
        content = state["sections"][idx].get("content", "") or ""
        sources = state["sections"][idx].get("sources") or []
        has_cite = bool(re.search(r"\[\d+\]", content) or "http" in content.lower())
        if len(content.strip()) < 200 and not sources and not has_cite:
            state["sections"][idx]["content"] = (
                f"### {sd.get('title', '')}\n\n"
                "_Section pruned: no supporting evidence was retrieved for this "
                "topic during research._"
            )
            pruned += 1
        else:
            keep.append((idx, sd))

    # Discovered findings: bound to no plan section → surface as extra section
    ledger = state.get("task_ledger") or []
    unbound = [
        e.get("finding") for e in ledger
        if not e.get("task_id") and e.get("finding")
    ]
    existing_titles = {
        (state["sections"][idx].get("title") or "").lower() for idx, _ in keep
    }
    if len(unbound) >= 3 and "additional findings" not in existing_titles:
        state["sections"].append({
            "title": "Additional Findings",
            "content": (
                "### Additional Findings\n\n"
                + "\n".join(f"- {str(f)[:220]}" for f in unbound[:8])
                + "\n\n_These findings did not map cleanly onto a planned section._"
            ),
            "sources": [],
        })
    return pruned


def _deepen_weakest_section(
    state: ResearchState,
    body_sections: list[tuple[int, dict]],
) -> str:
    """One bounded deepen pass on the weakest section (Tier-2 #15).

    AgentCPM-Report WARP: after the draft, find logical/evidence gaps and go
    deeper — re-retrieve with a wider net and re-write the section with the
    extra evidence integrated. Bound to one section, one pass, so it cannot
    blow up wall time.
    """
    if not body_sections:
        return ""
    # Weakest = shortest content among body sections
    weakest_idx, weakest_def = min(
        body_sections,
        key=lambda t: len(state["sections"][t[0]].get("content", "") or ""),
    )
    title = weakest_def.get("title", "")
    current = state["sections"][weakest_idx].get("content", "") or ""
    if len(current) >= 3000:
        return ""  # already deep enough

    try:
        run_id = state.get("run_id", "")
        factoids = state.get("factoids") or []
        chunks = hybrid_retrieve(
            f"{state.get('query', '')} {title}", k=14, factoids=factoids, run_id=run_id
        )
        chunk_text_extra = "\n\n".join(
            f"[Source: {c.get('title','') or c.get('url','')}]\n{c.get('text','')[:1400]}"
            for c in chunks[:12]
        ) if chunks else "(no additional sources retrieved)"

        prompt = f"""Deepen this section of a research report. The current draft is thin;
use the additional source material to expand it with specific evidence, numbers,
and named systems. Keep all inline citations [n] consistent with source order.

Section: "{title}"
Query: "{state.get('query', '')}"

Current draft (keep its structure, expand where thin):
{current[:2500]}

Additional source material:
{chunk_text_extra[:7000]}

Write the expanded section (markdown, 5-8 paragraphs). Integrate new evidence;
do not repeat the draft verbatim."""
        content = call_llm_strong(SYNTH_SYSTEM, prompt)
        if content and len(content.strip()) > len(current) + 200:
            state["sections"][weakest_idx]["content"] = content.strip()
            urls = [c.get("url") for c in chunks if c.get("url")]
            if urls:
                state["sections"][weakest_idx]["sources"] = urls
            return title
    except Exception as e:
        print(f"  🔍 Deepen pass failed for '{title}': {e}")
    return ""


def _update_progress(state: ResearchState, step: str, **kwargs) -> None:
    """Push progress to the global research progress tracker (SSE/dashboard)."""
    try:
        from src.engine.progress import get_progress
        sections = kwargs.get("sections") or state.get("sections") or []
        get_progress().update(
            stage=step,
            status=state.get("status") or step,
            sections=sections,
            findings_count=len(state.get("findings") or []),
            factoids_count=len(state.get("factoids") or []),
            total_sections=len(sections),
        )
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)
