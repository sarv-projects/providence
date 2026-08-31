"""
Compiler agent — assembles the final report from sections, validates citations,
and exports the result.

Ship gate (P0.3 / P0.4):
  - Sources section present and last
  - Sources must be this run's evidence URLs (not empty / fake monographs)
  - Claim–evidence: claim quotes appear in retrieved text when possible
  - No empty body
"""

from __future__ import annotations

import logging

import re
import time

from src.state import ResearchState
from src.export import save_markdown, save_html
from src.render.math import render_mathjax_html, has_math, detect_math
from src.urlutil import canonical_url
from src.evidence import verify_claims
from .registry import register

# Banned placeholder / fake monograph patterns
_FAKE_SOURCE_PATTERNS = (
    re.compile(r"^about:blank$", re.I),
    re.compile(r"^https?://example\.(com|org)", re.I),
    re.compile(r"placeholder", re.I),
    re.compile(r"lorem ipsum", re.I),
    re.compile(r"^factoid://", re.I),
    re.compile(r"^n/?a$", re.I),
)


def _is_fake_url(url: str) -> bool:
    if not url or len(url) < 8:
        return True
    if any(p.search(url) for p in _FAKE_SOURCE_PATTERNS):
        return True
    # Malformed / LLM-hallucinated URLs
    if " " in url or "[" in url or "]" in url:
        return True
    if "..." in url:
        return True
    if not url.startswith(("http://", "https://")):
        return True
    return False


def _collect_run_urls(state: ResearchState) -> list[tuple[str, str]]:
    """Collect (url, title) pairs from this run only.

    P0.5 hardening: a URL counts as evidence only if it was ACTUALLY fetched
    and read during this run. Fetched evidence = run_corpus + retrieved_chunks
    + extracted_pages + the explicit ``fetched_sources`` ledger the researcher
    maintains (url / status / content hash / timestamp). Bare search-result
    hits whose pages were never opened are NOT evidence and are excluded —
    previously they were cited as if verified.
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    ledger = state.get("fetched_sources") or {}

    def _add(url: str, title: str) -> None:
        url = canonical_url(url)
        if not url or url in seen or _is_fake_url(url):
            return
        if ledger:
            meta = ledger.get(url)
            if not isinstance(meta, dict) or meta.get("status") not in ("fetched", "success", "ok"):
                return
        seen.add(url)
        out.append((url, (title or url).strip()))

    # 1. Sources actually retrieved & read this run (all iterations)
    for c in state.get("run_corpus") or []:
        _add(c.get("url"), c.get("title") or c.get("url"))
    for c in state.get("retrieved_chunks") or []:
        _add(c.get("url"), c.get("title") or c.get("url"))
    for p in state.get("extracted_pages") or []:
        _add(p.get("url"), p.get("title") or p.get("url"))
    # 1b. Explicit fetched-source ledger (researcher-recorded actual fetches:
    #     status, content hash, fetched_at) — authoritative when present.
    for url, meta in ledger.items():
        if isinstance(meta, dict):
            if meta.get("status") not in (None, "fetched", "success", "ok"):
                continue
            _add(url, meta.get("title") or url)
        else:
            _add(url, url)
    # 2. Search hits count ONLY if the page was actually fetched above —
    #    a URL that appeared in a search listing was never opened.
    known = set(seen)
    for r in state.get("search_results") or []:
        u = canonical_url(r.get("url") or "")
        if u and u in known:
            _add(r.get("url"), r.get("title") or r.get("url"))
    # 3. evidence_map keys — only if they were actually retrieved above
    known = set(seen)
    for url in (state.get("evidence_map") or {}):
        if canonical_url(url) in known:
            _add(url, url)
    return out


_MATH_RE = re.compile(r"\$\$[^\n]+?\$\$|\$[^\n$]+\$|\\\([^\n]+?\\\)|\\\[[^\n]+?\\\]")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# Writer-appended reference blocks: "## References (source order)",
# "## References (as mapped from source materials)", "## Sources & Bibliography"…
_REF_BLOCK_RE = re.compile(
    r"^#{1,6}\s+(references|bibliography|works cited|"
    r"sources?\s*(&|and)?\s*bibliography)(\s*\([^)]*\))?\s*$",
    re.I,
)


def _renumber_section_citations(content: str, section_urls: list[str], final_urls: list[str]) -> str:
    """Rewrite inline [N] markers in a section to match the final Sources order.

    The parallel section writers number citations against their OWN retrieved
    chunk order (section_urls). The compiler rebuilds a single Sources list
    (final_urls), so markers must be remapped: [k] -> [m] where m is the index of
    section_urls[k-1] in final_urls. Markers whose URL did not make the final
    list (dropped/fake) are removed, and out-of-range markers (writer invented a
    number larger than any possible source) are dropped too, so no dangling
    citations remain. Math expressions and code fences are left untouched.
    """
    if not section_urls or not final_urls:
        return content
    # Canonicalize section URLs so they match the canonicalized final list
    # (synthesizer stores raw chunk URLs; _collect_run_urls canonicalizes)
    section_urls = [canonical_url(u) for u in section_urls]
    url_to_final: dict[str, int] = {
        u: i + 1 for i, (u, _) in enumerate(final_urls)
    }

    def _sub(m: re.Match) -> str:
        # Handle single AND multi-citation markers: [1], [1,2], [1, 2], [1-3]
        parts = re.split(r"\s*[,–—-]\s*", m.group(1))
        mapped: list[str] = []
        for p in parts:
            k = int(p)
            if 1 <= k <= len(section_urls):
                url = section_urls[k - 1]
                if url in url_to_final:
                    mapped.append(str(url_to_final[url]))
                    continue
                continue  # URL dropped from final list → drop this number
            if k > len(final_urls):
                continue  # writer invented an impossible citation number
            mapped.append(p)  # within-range but unmappable — keep original
        if not mapped:
            return ""  # every number in the marker was dropped
        return "[" + ", ".join(mapped) + "]"

    # Protect math, code fences, and inline code: substitute placeholders first
    protected: dict[str, str] = {}

    def _protect_math(m: re.Match) -> str:
        key = f"\x00M{len(protected)}\x00"
        protected[key] = m.group(0)
        return key

    def _protect_code(m: re.Match) -> str:
        key = f"\x00C{len(protected)}\x00"
        protected[key] = m.group(0)
        return key

    content = _MATH_RE.sub(_protect_math, content)
    content = _CODE_FENCE_RE.sub(_protect_code, content)
    content = _INLINE_CODE_RE.sub(_protect_math, content)
    content = re.sub(r"\[(\d{1,3}(?:\s*[,–—-]\s*\d{1,3})*)\]", _sub, content)
    for key, val in protected.items():
        content = content.replace(key, val)
    return content


def _strip_heading_duplicates(content: str, title: str) -> str:
    """Remove heading lines that duplicate the section title (parallel-writer artifact)."""
    t = (title or "").strip()
    if not t:
        return content
    lines = content.split("\n")
    out: list[str] = []
    seen_title_heading = False
    for ln in lines:
        stripped = ln.strip()
        m = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if m and m.group(1).strip().lower() == t.lower():
            if not seen_title_heading:
                seen_title_heading = True
                continue  # drop the duplicate title heading
        out.append(ln)
    return "\n".join(out)


def _strip_trailing_references(content: str) -> str:
    """Drop a trailing References/Bibliography block a section writer appended.

    The compiler appends its own Sources section; mid-report reference lists
    would duplicate it and can carry unreviewed URLs. Matches headings like
    "References (source order)" / "Sources & Bibliography". Only a block at the
    END of the section (no other heading after) is removed — a legitimately
    placed mid-section heading is left untouched.
    """
    lines = content.split("\n")
    cut = -1
    for i, ln in enumerate(lines):
        if _REF_BLOCK_RE.match(ln.strip()):
            cut = i
    if cut >= 0:
        rest = "\n".join(lines[cut + 1:])
        if re.search(r"^#{1,6}\s+", rest, re.M):
            return content
        return "\n".join(lines[:cut]).rstrip()
    return content


def _is_sources_like(title: str) -> bool:
    """True if a section title is a writer-produced bibliography.

    The compiler appends its own # Sources section, so mid-report lists
    ("References & Sources", "8. References", "Sources & Bibliography")
    are dropped from the body. Does not match ordinary titles that merely
    contain the word "source" (e.g. "Open-source landscape").
    """
    t = re.sub(r"[^a-z ]", " ", (title or "").lower()).strip()
    t = re.sub(r"^\d+\s*", "", t).strip()
    words = set(t.split())
    if any(w in words for w in ("references", "bibliography", "citations")):
        return True
    if t in ("sources", "source list", "works cited", "sources list"):
        return True
    if "sources" in words and any(w in words for w in ("list", "and", "bibliography")):
        return True
    return False


def _claim_evidence_check(state: ResearchState) -> tuple[int, int, list[str]]:
    """Apply the canonical verifier used by the adversary stage.

    This deliberately re-verifies adjudicated output instead of trusting a
    prior label.  It prevents stale or hand-built adjudication rows from
    bypassing the final ship gate.
    """
    result = verify_claims(state)
    rows = result["claims"]
    state["verified_spans"] = result["spans"]
    state["evidence_graph"] = result["graph"]
    state["adjudicated_claims"] = rows
    supported = sum(1 for row in rows if row.get("status") == "supported")
    unsupported = [
        (row.get("text") or "")[:80]
        for row in rows
        if row.get("status") != "supported"
    ][:5]
    return supported, len(rows), unsupported


def _build_bedrock_section(state: ResearchState) -> dict:
    """Layer 1 — Bedrock: quotes / claims with support status (zero-hallucination zone)."""
    lines = [
        "# Evidence Bedrock\n",
        "_Direct evidence from this run's retrieval. Prefer this layer over inference._\n",
    ]
    adj = state.get("adjudicated_claims") or []
    if adj:
        lines.append("## Adjudicated claims\n")
        for i, a in enumerate(adj[:20], 1):
            st = a.get("status", "?")
            sc = a.get("score", "")
            eids = a.get("evidence_ids") or []
            eid = eids[0] if eids else ""
            flag = {
                "supported": "✅",
                "contradicted": "❌",
                "contested": "⚠️",
                "synthetic": "🧪",
                "uncertain": "⚠️",
            }.get(st, "•")
            lines.append(f"{flag} **[{i}] ({st}** score={sc}) {a.get('text', '')}")
            if eid and not _is_fake_url(str(eid)):
                lines.append(f"   - evidence: {eid}")
            lines.append("")
        # Compact evidence-graph view (Argus): claims ↔ sources with relation
        graph = state.get("evidence_graph") or []
        if graph:
            lines.append("## Evidence graph (claims ↔ sources)\n")
            lines.append("| Claim | Relation | Evidence |")
            lines.append("|-------|----------|----------|")
            rel_label = {
                "support": "✅ support",
                "contradiction": "⚠️ contradiction",
                "unsupported": "🧪 unsupported",
            }
            for e in graph[:12]:
                claim = str(e.get("claim") or "")[:90].replace("|", "/")
                rel = rel_label.get(e.get("relation"), str(e.get("relation")))
                ev = str(e.get("evidence_url") or "—")[:60]
                lines.append(f"| {claim} | {rel} | {ev} |")
            lines.append("")
    else:
        # Fallback path (no adjudicator run): only surface evidence URLs that
        # were actually retrieved this run — never raw LLM-provided IDs.
        run_urls = {u for u, _ in _collect_run_urls(state)}
        claims = state.get("claims") or []
        for i, c in enumerate(claims[:15], 1):
            lines.append(f"{i}. {c.get('text', '')[:300]}")
            shown: set[str] = set()
            for u in (c.get("evidence_ids") or [])[:4]:
                cu = canonical_url(u)
                if cu and cu in run_urls and cu not in shown:
                    shown.add(cu)
                    lines.append(f"   - {cu}")
            lines.append("")
    # Top chunk quotes
    chunks = state.get("retrieved_chunks") or []
    if chunks:
        lines.append("## Retrieved source excerpts\n")
        for i, c in enumerate(chunks[:8], 1):
            title = (c.get("title") or c.get("url") or "chunk")[:100]
            url = c.get("url") or ""
            quote = (c.get("text") or "")[:400].replace("\n", " ")
            lines.append(f"**[{i}] {title}**")
            if url and not _is_fake_url(url):
                lines.append(f"- {url}")
            lines.append(f"> {quote}\n")
    if len(lines) < 4:
        lines.append("_No bedrock evidence captured._\n")
    return {"title": "Evidence Bedrock", "content": "\n".join(lines), "sources": []}


def _build_research_debt_section(state: ResearchState) -> dict:
    """Layer 3 — Research Debt: what remains unknown / experiments still needed."""
    debt = list(state.get("research_debt") or [])
    gaps = state.get("gaps") or []
    contested = state.get("contested_claims") or []
    synthetic = state.get("synthetic_claims") or []
    note = state.get("confidence_note") or ""

    lines = [
        "# Research Debt\n",
        "_To be more certain, these items still need work (intellectual honesty layer)._\n",
    ]
    if note:
        lines.append(f"**Confidence note:** {note}\n")
    if debt:
        lines.append("## Outstanding debt\n")
        for d in debt[:10]:
            lines.append(f"- {d}")
        lines.append("")
    if contested:
        lines.append("## Contested claims (need better evidence)\n")
        for c in contested[:6]:
            lines.append(f"- {c.get('text', '')[:200]}")
        lines.append("")
    if synthetic:
        lines.append("## Synthetic inferences (no solid source chunk)\n")
        for c in synthetic[:6]:
            lines.append(f"- {c.get('text', '')[:200]}")
        lines.append("")
    if gaps and not debt:
        lines.append("## Open gaps\n")
        for g in gaps[:8]:
            lines.append(f"- {g}")
        lines.append("")
    if len(lines) < 5:
        lines.append(
            "- No major residual debt flagged; still verify key citations against primary sources.\n"
        )
    lines.append(
        "\n_This section is intentionally incomplete knowledge — not a failure of the agent._\n"
    )
    return {"title": "Research Debt", "content": "\n".join(lines), "sources": []}


def _unbound_inline_citations(state: ResearchState) -> int:
    """Count [n] markers in body sections whose n is outside the final Sources list.

    Writers invent [1] as a generic placeholder. After remapping, leftovers
    that cannot bind to a fetched URL are counted here.
    """
    run_urls = _collect_run_urls(state)
    n_src = len(run_urls)
    if n_src == 0:
        return 0
    count = 0
    for s in state.get("sections") or []:
        if _is_sources_like(s.get("title") or ""):
            continue
        if (s.get("title") or "").lower() in ("evidence bedrock", "research debt", "bedrock"):
            continue
        content = s.get("content") or ""
        for m in re.finditer(r"\[(\d{1,3})\]", content):
            k = int(m.group(1))
            if k < 1 or k > n_src:
                count += 1
    return count


def _validate_ship_gate(state: ResearchState) -> tuple[bool, list[str]]:
    """Validate the report passes the ship gate before export."""
    issues: list[str] = []

    sections = state.get("sections", [])
    if not sections:
        issues.append("No sections written")

    source_section = None
    for s in sections:
        if s.get("title", "").lower() in ("sources", "references"):
            source_section = s
            break
    if not source_section:
        issues.append("Missing Sources/References section")

    total_content = sum(len(s.get("content", "")) for s in sections)
    if total_content < 100:
        issues.append("Report body is too short (<100 chars)")

    run_urls = _collect_run_urls(state)
    if not state.get("abort_synthesis"):
        if not run_urls:
            issues.append("No this-run evidence URLs (ban empty/fake monographs)")
        # Detect fake-only sources
        fake_count = 0
        for s in sections:
            for u in s.get("sources") or []:
                if _is_fake_url(u):
                    fake_count += 1
        if fake_count and not run_urls:
            issues.append("Only empty/fake source URLs present")

    claims = state.get("claims") or []
    if claims and not state.get("abort_synthesis"):
        supported, total, samples = _claim_evidence_check(state)
        if total >= 3 and supported / max(total, 1) < 0.3:
            issues.append(
                f"Claim–evidence check failed ({supported}/{total} supported); "
                f"e.g. {samples[:2]}"
            )

    unbound = _unbound_inline_citations(state)
    if unbound >= 8 and not state.get("abort_synthesis"):
        issues.append(
            f"Body has {unbound} inline [n] citations that do not map to this-run URLs"
        )

    # Evidence-graph gate (Argus): a meaningful graph with ZERO supported edges
    # means every claim is unsupported/contradicted — nothing should ship.
    graph = state.get("evidence_graph") or []
    if len(graph) >= 5 and not state.get("abort_synthesis"):
        support_edges = sum(1 for e in graph if e.get("relation") == "support")
        if support_edges == 0:
            issues.append(
                f"Evidence graph: no supported edges "
                f"({len(graph)} claims all unsupported/contradicted)"
            )

    return len(issues) == 0, issues


def _build_sources_section(state: ResearchState) -> dict:
    urls = _collect_run_urls(state)
    lines = ["# Sources\n"]
    if urls:
        for i, (url, title) in enumerate(urls[:60], 1):
            safe_title = (title or url).replace("\n", " ").strip()[:120]
            lines.append(f"[{i}] [{safe_title}]({url})")
    else:
        lines.append("_No external sources were retained for this run._")
    return {
        "title": "Sources",
        "content": "\n".join(lines) + "\n",
        "sources": [u for u, _ in urls],
    }


@register("compiler")
def compiler(state: ResearchState) -> ResearchState:
    """Assemble report, run ship gate, and export.

    Confidence volcano (Ultra steal):
      Inference body (existing sections)
      + Evidence Bedrock
      + Research Debt
      + Sources (last)
    """
    state["status"] = "Compiling final report..."
    print(f"\n📦 [Compiler] Assembling report")

    # Strip special trailing sections then re-append in order (P0.4: also drop
    # writer-produced "Sources & Bibliography" sections — the compiler appends
    # the single authoritative # Sources at the end)
    skip_titles = {
        "research debt", "evidence bedrock", "bedrock",
    }
    body = [
        s for s in (state.get("sections") or [])
        if not _is_sources_like(s.get("title", ""))
        and s.get("title", "").lower().strip() not in skip_titles
    ]
    # Inference layer label (optional banner section)
    if body and not any("inference" in s.get("title", "").lower() for s in body[:1]):
        pass  # body already is inference layer

    # P0.4 cleanup pass on parallel-written body sections:
    #  1. drop duplicate title headings  2. drop mid-report References blocks
    #  3. renumber inline [N] citations against the final Sources list
    final_urls = _collect_run_urls(state)
    for s in body:
        content = s.get("content") or ""
        content = _strip_heading_duplicates(content, s.get("title") or "")
        content = _strip_trailing_references(content)
        content = _renumber_section_citations(
            content, list(s.get("sources") or []), final_urls
        )
        s["content"] = content

    body.append(_build_bedrock_section(state))
    body.append(_build_research_debt_section(state))
    body.append(_build_sources_section(state))
    state["sections"] = body
    print("  Confidence volcano: Inference body + Bedrock + Research Debt + Sources")

    # Run ship gate
    passed, issues = _validate_ship_gate(state)
    autonomy = str(state.get("autonomy") or "L1").upper()
    if issues:
        print(f"  ⚠️  Ship gate issues: {issues}")
        if any("Sources" in i or "evidence" in i.lower() or "empty" in i.lower() for i in issues):
            body = [
                s for s in state["sections"]
                if s.get("title", "").lower() not in ("sources", "references")
            ]
            body.append(_build_sources_section(state))
            state["sections"] = body
            passed, issues = _validate_ship_gate(state)

        if issues and autonomy in ("L3", "STRICT"):
            state["error"] = f"Ship gate failed: {issues}"
            state["status"] = f"Ship gate blocked: {issues}"
            print(f"  🛑 Ship gate blocked under autonomy={autonomy}: {issues}")
            state["report"] = (
                f"# Research Report (BLOCKED)\n\nShip gate failures: {issues}\n\n"
                + "\n\n".join(
                    f"## {s.get('title','')}\n\n{s.get('content','')}"
                    for s in state.get("sections", [])
                )
            )
            return state

    supported, total, unsup = _claim_evidence_check(state)
    if total:
        print(f"  Claim–evidence (CoVe): {supported}/{total} supported")
        if unsup:
            print(f"  Unsupported samples: {unsup[:2]}")
    graph = state.get("evidence_graph") or []
    if graph:
        g_sup = sum(1 for e in graph if e.get("relation") == "support")
        print(f"  Evidence graph: {len(graph)} edges ({g_sup} support, "
              f"{len(graph) - g_sup} contradiction/unsupported)")

    sections = state.get("sections", [])
    conf = ""
    if total:
        pct = int(100 * supported / max(total, 1))
        conf = f"**Evidence confidence**: {pct}% claims supported ({supported}/{total})"
    if state.get("confidence_note"):
        conf = (conf + f"  \n**Note**: {state['confidence_note']}") if conf else f"**Note**: {state['confidence_note']}"

    report_lines = [
        f"# Research Report: {state['query']}",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Sources**: {len(_collect_run_urls(state))} references",
        f"**Iterations**: {state.get('iteration', 0)}",
        f"**Methodology**: Scout → research loop → devil's advocate → adjudicate → synth",
        f"**Claim–evidence**: {supported}/{total} claims grounded" if total else "",
        conf,
        "",
        "> **Report layers:** (1) Inference body below · (2) Evidence Bedrock · "
        "(3) Research Debt · (4) Sources. Prefer Bedrock over uncited inference.",
        "",
    ]

    for s in sections:
        report_lines.append(f"## {s['title']}")
        report_lines.append("")
        report_lines.append(s.get("content", "").strip())
        report_lines.append("")

    report = "\n".join(line for line in report_lines if line is not None)

    # Math rendering
    if has_math(report):
        math_info = detect_math(report)
        print(f"  Math detected: {math_info['count']} expressions "
              f"({len(math_info['inline'])} inline, {len(math_info['block'])} block)")
        report = render_mathjax_html(report)

    state["report"] = report

    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in state["query"])[:50]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"research_{safe_name}_{timestamp}"

    md_path = save_markdown(report, base)
    state["markdown_path"] = md_path
    html_path = save_html(report, base, title=state["query"])

    ship_status = "✅ passed" if passed else "⚠️  issues noted"
    state["status"] = f"Report compiled ({len(report)} chars, ship gate {ship_status})"
    print(f"  Report: {len(report)} chars, {len(sections)} sections")
    print(f"  Ship gate: {ship_status}")
    if issues:
        print(f"  Remaining issues: {issues}")
    print(f"  Saved: {md_path}")
    print(f"  HTML:  {html_path}")

    try:
        from src.engine.progress import get_progress
        get_progress().update(
            stage="complete",
            status=state["status"],
            sources_count=len(_collect_run_urls(state)),
            report=report,
            markdown_path=md_path,
        )
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)
    return state
