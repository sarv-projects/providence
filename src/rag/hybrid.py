"""
Hybrid Retriever — fuses dense (vector), sparse (keyword), and factoid results
using Reciprocal Rank Fusion (RRF) for the current run corpus.

Also supports cross-run vault search for persistent source retrieval.
"""

from __future__ import annotations

import logging

from typing import Optional

from .pipeline import retrieve_chunks as _vector_retrieve
from .store import VectorStore
from .rerank import rerank_results, rerank_available


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score: 1 / (k + rank)."""
    return 1.0 / (k + rank)


def hybrid_retrieve(
    query: str,
    k: int = 10,
    store: Optional[VectorStore] = None,
    embedder=None,
    factoids: Optional[list[dict]] = None,
    run_id: str = "",
    filters: Optional[dict] = None,
) -> list[dict]:
    """Hybrid retrieval: dense vector + sparse keyword + factoid fusion.

    When run_id is set, only current-run chunks are returned (isolation).

    filters: Optional metadata filter dict (Tier-2 #19) applied on both the
    dense (LanceDB .where) and sparse (FTS) streams, e.g. {source_type, acl}.

    When a cross-encoder reranker is available, more candidates are fused
    (over-fetch) and then reranked down to k for higher top-k precision.
    """
    from .pipeline import _get_or_create_store, _get_or_create_embedder

    if store is None:
        store = _get_or_create_store()
    if embedder is None:
        embedder = _get_or_create_embedder()

    # Over-fetch when reranking so the final top-k is chosen from a wider pool.
    use_rerank = rerank_available()
    fetch_k = k * 5 if use_rerank else k * 2

    seen_ids: set[str] = set()
    rrf_scores: dict[str, float] = {}
    merged: dict[str, dict] = {}

    # ── Dense (vector) stream ──
    try:
        vec_results = _vector_retrieve(
            query, k=fetch_k, store=store, embedder=embedder, run_id=run_id,
            filters=filters,
        )
        for rank, r in enumerate(vec_results):
            rid = r.get("id", "")
            if rid not in seen_ids:
                seen_ids.add(rid)
                rrf_scores[rid] = rrf_scores.get(rid, 0) + _rrf_score(rank)
                r["source"] = "dense"
                merged[rid] = r
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)

    # ── Sparse (keyword) stream via FTS ──
    if store._fts:
        try:
            fts_results = store._fts.query(
                query, k=fetch_k * 2 if run_id else fetch_k, filters=filters
            )
            if run_id:
                fts_results = [r for r in fts_results if r.get("run_id") == run_id]
            for rank, r in enumerate(fts_results):
                rid = r.get("id", "")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                rrf_scores[rid] = rrf_scores.get(rid, 0) + _rrf_score(rank)
                r["source"] = "keyword"
                if rid not in merged:
                    merged[rid] = r
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)

    # ── Factoid stream ──
    if factoids:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored_factoids = []
        for f in factoids:
            val = (f.get("value", "") or "").lower()
            # Simple BM25-like: word overlap * confidence
            val_words = set(val.split())
            overlap = len(query_words & val_words) / max(len(query_words), 1)
            score = overlap * f.get("confidence", 0.5)
            if overlap > 0:
                scored_factoids.append((score, f))

        scored_factoids.sort(key=lambda x: x[0], reverse=True)
        for rank, (score, f) in enumerate(scored_factoids[:k]):
            fid = f.get("id", f"factoid_{rank}")
            if fid not in seen_ids:
                seen_ids.add(fid)
                rrf_scores[fid] = rrf_scores.get(fid, 0) + _rrf_score(rank)
                merged[fid] = {
                    "id": fid,
                    "text": f.get("value", ""),
                    "url": f.get("source_url", ""),
                    "title": f"[Factoid: {f.get('type', '')}]",
                    "score": score,
                    "source": "factoid",
                }

    # ── Final ranking by RRF score ──
    for rid, r in merged.items():
        r["rrf_score"] = round(rrf_scores.get(rid, 0), 4)
        # Blend original score with RRF
        raw_score = r.get("score", 0.5)
        r["score"] = round(raw_score * 0.3 + rrf_scores.get(rid, 0) * 10.0, 4)

    sorted_results = sorted(
        merged.values(),
        key=lambda r: r.get("score", 0),
        reverse=True,
    )

    # ── Optional cross-encoder rerank (top-k precision boost) ──
    if use_rerank:
        return rerank_results(query, sorted_results, k=k)

    return sorted_results[:k]


def search_vault(
    query: str,
    vault_path: str = "",
    k: int = 5,
) -> list[dict]:
    """Search the persistent source vault for previous research on similar topics.

    Returns sources from past runs that match the query (keyword-based).
    """
    try:
        from .vault import Vault
        vault = Vault(vault_path)
        return vault.search(query, k=k)
    except Exception:
        return []
