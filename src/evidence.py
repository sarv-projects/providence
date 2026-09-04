"""Canonical, span-based evidence verification.

Evidence is valid only when a claim points at content that this run actually
received and supplies a verbatim quote from that content.  Search-result
metadata is deliberately not a source document.  The verifier is shared by
the adjudicator and compiler so those stages cannot disagree about support.
"""

from __future__ import annotations

import re
from typing import Any

from src.urlutil import canonical_url


_NUMBER_RE = re.compile(r"(?<![\w.-])\d+(?:[.,]\d+)?\s*%?(?![\w.-])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")


def _source_documents(state: dict[str, Any]) -> dict[str, str]:
    """Return canonical URL -> fetched text.

    ``search_results`` is intentionally absent.  The content-bearing state
    collections are the fetch ledger in practice; if an explicit ledger is
    present, only entries marked fetched are accepted.
    """
    docs: dict[str, str] = {}
    ledger = state.get("fetched_sources") or {}
    enforce_ledger = bool(ledger)

    def add(item: dict[str, Any], text_key: str = "text") -> None:
        url = canonical_url(str(item.get("url") or ""))
        text = str(item.get(text_key) or item.get("content") or "").strip()
        if not url or not text:
            return
        meta = ledger.get(url) if isinstance(ledger, dict) else None
        if enforce_ledger and (not isinstance(meta, dict) or meta.get("status") != "fetched"):
            return
        docs[url] = (docs.get(url, "") + "\n" + text).strip()

    for item in state.get("run_corpus") or []:
        if isinstance(item, dict):
            add(item)
    for item in state.get("retrieved_chunks") or []:
        if isinstance(item, dict):
            add(item)
    for item in state.get("extracted_pages") or []:
        if isinstance(item, dict):
            add(item, text_key="content")
    return docs


def _find_verbatim_span(text: str, quote: str) -> dict[str, Any] | None:
    """Find a quote with exact characters or whitespace-only normalization."""
    quote = str(quote or "").strip()
    if not quote:
        return None
    direct = text.lower().find(quote.lower())
    if direct >= 0:
        return {"quote": text[direct:direct + len(quote)], "start": direct, "end": direct + len(quote)}

    # Permit line wrapping/HTML extraction whitespace changes, but no fuzzy
    # substitutions: a quote must still be a contiguous source span.
    # Cap the pattern: unbounded LLM quotes build giant regexes (ReDoS).
    parts = [re.escape(p) for p in re.split(r"\s+", quote) if p][:60]
    if not parts:
        return None
    try:
        match = re.search(r"\s+".join(parts), text, re.IGNORECASE | re.DOTALL)
    except re.error:
        return None
    if not match:
        return None
    return {"quote": text[match.start():match.end()], "start": match.start(), "end": match.end()}


def _atoms(claim: dict[str, Any]) -> list[str]:
    values = claim.get("atoms") or claim.get("atomic_claims") or []
    if isinstance(values, str):
        values = [values]
    values = [str(v).strip() for v in values if str(v).strip()]
    return values or [str(claim.get("text") or "").strip()]


def _evidence_items(claim: dict[str, Any]) -> list[dict[str, Any]]:
    raw = claim.get("evidence") or claim.get("evidence_spans") or []
    if isinstance(raw, dict):
        raw = [raw]
    result = [dict(v) for v in raw if isinstance(v, dict)]
    # Backward-compatible input, but still requires a quote.  A URL by itself
    # is never enough to support a claim.
    quote = claim.get("source_quote") or claim.get("quote")
    if quote and not result:
        result = [{"quote": quote}]
    ids = claim.get("evidence_ids") or []
    for i, url in enumerate(ids):
        if i < len(result):
            result[i].setdefault("url", url)
        elif quote:
            result.append({"url": url, "quote": quote})
    return result


def _anchors_match(atom: str, quote: str) -> bool:
    """Conservative factual-anchor check, not a semantic entailment claim."""
    numbers = {n.replace(",", ".").replace(" ", "") for n in _NUMBER_RE.findall(atom)}
    quote_numbers = {n.replace(",", ".").replace(" ", "") for n in _NUMBER_RE.findall(quote)}
    if numbers and not numbers.issubset(quote_numbers):
        return False
    words = {w.lower() for w in _WORD_RE.findall(atom)}
    quote_words = {w.lower() for w in _WORD_RE.findall(quote)}
    meaningful = {w for w in words if len(w) >= 4}
    # Exact quotes carry the primary proof. This guard catches a quote from a
    # different paragraph attached to a claim with no shared subject.
    return not meaningful or bool(meaningful & quote_words)


def verify_claims(state: dict[str, Any]) -> dict[str, Any]:
    """Verify all claims against fetched source spans.

    Returns normalized adjudication rows, evidence graph edges, and spans.
    Statuses are ``supported``, ``uncertain``, or ``contradicted``.  Existing
    atomic verification can mark a claim contradicted, but cannot turn a URL
    association or lexical overlap into support.
    """
    docs = _source_documents(state)
    claims = [c for c in (state.get("claims") or []) if isinstance(c, dict) and str(c.get("text") or "").strip()]
    refuted = {
        str(item.get("claim") or item.get("text") or "").strip().lower()
        for item in (state.get("atomic_verified") or [])
        if str(item.get("status") or item.get("verdict") or "").upper() in {"REFUTED", "CONTRADICTED"}
    }

    rows: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    graph: list[dict[str, Any]] = []
    for index, claim in enumerate(claims, 1):
        text = str(claim.get("text") or "").strip()
        if text.lower() in refuted:
            status = "contradicted"
        else:
            atoms = _atoms(claim)
            evidence = _evidence_items(claim)
            atom_spans: list[dict[str, Any]] = []
            verified_urls: list[str] = []
            for atom in atoms:
                found = None
                for item in evidence:
                    url = canonical_url(str(item.get("url") or item.get("source_id") or ""))
                    if url not in docs:
                        continue
                    span = _find_verbatim_span(docs[url], str(item.get("quote") or item.get("span") or ""))
                    if span and _anchors_match(atom, span["quote"]):
                        found = {"atom": atom, "url": url, **span}
                        break
                if found:
                    atom_spans.append(found)
                    if found["url"] not in verified_urls:
                        verified_urls.append(found["url"])
            status = "supported" if atom_spans and len(atom_spans) == len(atoms) else "uncertain"
            for span in atom_spans:
                spans.append({"claim": text, **span})
            score = len(atom_spans) / max(len(atoms), 1)
            row = {
                "claim_id": str(claim.get("id") or f"C{index}"),
                "text": text[:400],
                "status": status,
                "score": round(score, 3),
                "evidence_ids": verified_urls[:5],
                "spans": atom_spans[:10],
            }
            rows.append(row)
            for url in (verified_urls[:5] or [""]):
                graph.append({
                    "claim_id": row["claim_id"],
                    "claim": text[:200],
                    "evidence_url": url,
                    "relation": "support" if status == "supported" else "unsupported",
                    "score": row["score"],
                })
            continue

        row = {
            "claim_id": str(claim.get("id") or f"C{index}"),
            "text": text[:400], "status": status, "score": 0.0,
            "evidence_ids": [], "spans": [],
        }
        rows.append(row)
        graph.append({"claim_id": row["claim_id"], "claim": text[:200],
                      "evidence_url": "", "relation": "contradiction", "score": 0.0})

    return {"claims": rows, "spans": spans[:200], "graph": graph}

