"""
RAG pipeline — end-to-end ingest and retrieve flows.

Ingest:
  documents → chunk (parent-child when RAG_PARENT_CHILD=1) → embed → upsert

Retrieve:
  query → embed → hybrid_query(VectorStore) → scored chunks (filters applied)

Module-level singletons for store and embedder to avoid creating new
connections on every call during the research loop.
"""

from __future__ import annotations

import logging

import hashlib
import os
import uuid
from typing import Optional

from .chunk import chunk_text, chunk_children_with_parents, Chunk
from .embed import get_embedder, Embedder
from .store import VectorStore, get_vector_store

# Parent-child chunking toggle (Tier-2 #20): retrieve small, feed large.
_PARENT_CHILD = os.getenv("RAG_PARENT_CHILD", "1").lower() not in ("0", "false", "no")

# Module-level singletons — reused across research iterations
_store: Optional[VectorStore] = None
_embedder: Optional[Embedder] = None


def _get_or_create_store() -> VectorStore:
    global _store
    if _store is None:
        _store = get_vector_store()
    return _store


def _get_or_create_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def reset_pipeline() -> None:
    """Reset singletons (useful for testing)."""
    global _store, _embedder
    _store = None
    _embedder = None


def ingest_documents(
    pages: list[dict],
    run_id: str = "",
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
    parent_child: Optional[bool] = None,
) -> int:
    """Ingest extracted pages into the vector store.

    Flow: chunk (parent-child when enabled) → embed → upsert. Uses module-level
    singletons by default for efficient reuse across research iterations.

    Args:
        pages: List of dicts with {url, content, title?, source_type?, acl?}.
        run_id: Unique identifier for this research run.
        store: VectorStore instance (uses singleton if None).
        embedder: Embedder instance (uses singleton if None).
        parent_child: Override parent-child chunking (default: env RAG_PARENT_CHILD).

    Returns:
        Number of chunks ingested.
    """
    if store is None:
        store = _get_or_create_store()
    if embedder is None:
        embedder = _get_or_create_embedder()

    if not pages:
        return 0

    use_parent_child = _PARENT_CHILD if parent_child is None else bool(parent_child)
    all_chunks: list[Chunk] = []

    for page in pages:
        url = page.get("url", "")
        title = page.get("title", "")
        content = page.get("content", "") or page.get("raw_content", "")

        if not content:
            continue

        base_meta = {
            "url": url,
            "title": title[:200] if title else "",
            "source_type": str(page.get("source_type") or "web")[:40],
            "run_id": run_id,
        }
        acl = page.get("acl")
        if acl:
            base_meta["acl"] = str(acl)[:40]

        # Parent-child chunking (retrieve small, feed large) when enabled
        if use_parent_child:
            chunks = chunk_children_with_parents(
                content, chunk_size=600, chunk_overlap=60, metadata=base_meta
            )
        else:
            chunks = chunk_text(
                content, chunk_size=600, chunk_overlap=60, metadata=base_meta
            )

        # Assign unique IDs: url + run_id + chunk_index avoids collisions across runs
        for c in chunks:
            c.id = hashlib.md5(
                f"{url}:{run_id}:{c.metadata.get('chunk_index', 0)}".encode()
            ).hexdigest()[:16]

        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    # Embed in batches (with short timeout — embeddings are best-effort)
    texts = [c.text for c in all_chunks]
    try:
        embeddings = embedder.embed_batch(texts)
        for c, vec in zip(all_chunks, embeddings):
            c.embedding = vec
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # Embeddings are best-effort — FTS5 keyword search handles fallback
        err_msg = str(e)[:80]
        print(f"  [rag] embedding skipped ({err_msg}) — using keyword fallback")

    # Upsert
    store.upsert(all_chunks)

    return len(all_chunks)


def retrieve_chunks(
    query: str,
    k: int = 10,
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
    run_id: str = "",
    filters: Optional[dict] = None,
) -> list[dict]:
    """Retrieve relevant chunks for a query. Uses module-level singletons.

    When run_id is set, only returns chunks from that research run
    (prevents cross-run contamination).

    filters: Optional metadata filter dict, e.g. {source_type, run_id, url} —
             applied via backend where-clauses (Tier-2 #19 ACL/metadata control).
    """
    if store is None:
        store = _get_or_create_store()
    if embedder is None:
        embedder = _get_or_create_embedder()

    embedding = None
    try:
        embedding = embedder.embed(query)
    except RuntimeError:
        pass  # Will use FTS fallback

    # Over-fetch heavily when isolating so current-run chunks survive filtering
    fetch_k = k * 20 if run_id else k
    # Push run_id into backend filters so Qdrant/LanceDB scope server-side —
    # previously only the Python post-filter isolated runs, and backends
    # without filter support (Qdrant) returned other runs' chunks.
    effective_filters = dict(filters or {})
    if run_id and "run_id" not in effective_filters:
        effective_filters["run_id"] = run_id
    results = store.query(text=query, embedding=embedding, k=fetch_k, filters=effective_filters or None)

    if run_id:
        filtered = [r for r in results if r.get("run_id") == run_id]
        if filtered:
            results = filtered
        else:
            # Direct FTS-by-run fallback (dense top-k can be dominated by old runs)
            # Filters must carry through — otherwise ACL/source_type isolation
            # silently leaks in the fallback path (review finding).
            try:
                if getattr(store, "_fts", None):
                    fts_hits = store._fts.query(query, k=k * 10, filters=filters)
                    filtered = [r for r in fts_hits if r.get("run_id") == run_id]
                    if filtered:
                        results = filtered
                    else:
                        tagged = [r for r in results if r.get("run_id")]
                        results = [] if tagged else results  # refuse cross-run
                else:
                    tagged = [r for r in results if r.get("run_id")]
                    results = [] if tagged else results
            except Exception:
                tagged = [r for r in results if r.get("run_id")]
                results = [] if tagged else results

    return results[:k]


def begin_run(run_id: str) -> None:
    """Mark start of a research run; optionally purge prior data for this run_id."""
    store = _get_or_create_store()
    try:
        store.delete_by_run(run_id)
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)
