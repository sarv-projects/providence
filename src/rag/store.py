"""
Vector store abstraction — selects the appropriate backend.

Backends:
    lancedb   — Default embedded vector store (zero Docker)
    fts       — SQLite FTS5 keyword search (always-on fallback)
    qdrant    — Production multi-user (future)

Env: VECTOR_BACKEND=lancedb|fts|qdrant  (default: lancedb)
"""

from __future__ import annotations

import os
from typing import Optional


# Default embedding dimension (OpenAI text-embedding-3-small = 1536)
DEFAULT_VECTOR_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def get_vector_dim() -> int:
    """Get the current embedding dimension from environment or default."""
    return DEFAULT_VECTOR_DIM


class VectorStore:
    """Unified interface across all vector store backends.

    Uses DEFAULT_VECTOR_DIM to ensure the store and embedder agree on dimensions.
    Set EMBEDDING_DIM env var to override (1536 for OpenAI, 768 for others).
    """

    def __init__(self, backend: str = "auto", vector_dim: Optional[int] = None) -> None:
        if backend == "auto":
            backend = os.getenv("VECTOR_BACKEND", "lancedb")
            # Auto-detect Qdrant if URL is set
            if backend == "lancedb" and os.getenv("QDRANT_URL"):
                try:
                    from .backends.qdrant_backend import qdrant_is_available
                    if qdrant_is_available():
                        backend = "qdrant"
                except ImportError:
                    pass

        self.backend_name = backend
        self.vector_dim = vector_dim or get_vector_dim()
        self._fts = None
        self._lancedb = None
        self._qdrant = None

        # Always attach FTS for hybrid sparse search (except pure-fts backend which is only FTS)
        from .backends.fts import FTSStore
        self._fts = FTSStore()

        if backend == "fts":
            pass  # FTS-only
        elif backend == "qdrant":
            from .backends.qdrant_backend import QdrantStore
            self._qdrant = QdrantStore(vector_dim=self.vector_dim)
        else:
            # Default: LanceDB dense + FTS sparse (hybrid)
            from .backends.lancedb_backend import LanceDBStore
            self._lancedb = LanceDBStore(vector_dim=self.vector_dim)

    def upsert(self, chunks: list) -> None:
        """Store chunks in the vector database."""
        if self._qdrant:
            self._qdrant.upsert(chunks)
        if self._lancedb:
            self._lancedb.upsert(chunks)
        if self._fts:
            self._fts.upsert(chunks)

    def query(
        self,
        text: str = "",
        embedding: Optional[list[float]] = None,
        k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Hybrid query: vector similarity (if embedding available) + keyword fallback.

        Args:
            text: Query string for the sparse (FTS) stream.
            embedding: Optional dense vector.
            k: Max results.
            filters: Optional metadata filter dict, e.g. {source_type, run_id, url}.
                     Applied via backend where-clauses (Tier-2 #19).

        Returns the best results from the primary backend.
        """
        results = []

        # Primary vector backend
        if self._qdrant and embedding:
            if len(embedding) < self.vector_dim:
                embedding = embedding + [0.0] * (self.vector_dim - len(embedding))
            embedding = embedding[:self.vector_dim]
            # Qdrant backend filters server-side on run_id — previously the
            # filters were dropped here and other runs' chunks leaked in.
            results = self._qdrant.query(
                embedding, k=k, run_id=str((filters or {}).get("run_id") or "")
            )
        elif self._lancedb and embedding:
            # Pad or truncate embedding to match store dimension
            if len(embedding) < self.vector_dim:
                embedding = embedding + [0.0] * (self.vector_dim - len(embedding))
            embedding = embedding[:self.vector_dim]
            results = self._lancedb.query(embedding, k=k, filters=filters)

        # Always blend FTS keyword results (hybrid) when text is available
        if self._fts and text:
            fts_results = self._fts.query(text, k=k, filters=filters)
            seen = {r["id"] for r in results}
            for r in fts_results:
                if r["id"] not in seen:
                    results.append(r)
                    seen.add(r["id"])

        return results[:k]

    def delete_by_run(self, run_id: str) -> None:
        """Remove all chunks for a specific research run."""
        if self._qdrant:
            self._qdrant.delete_by_run(run_id)
        if self._lancedb:
            self._lancedb.delete_by_run(run_id)
        if self._fts:
            self._fts.delete_by_run(run_id)

    def count(self) -> int:
        """Return total number of stored chunks."""
        if self._lancedb:
            return self._lancedb.count()
        if self._fts:
            return self._fts.count()
        return 0


def get_vector_store(backend: str = "auto") -> VectorStore:
    """Get a configured VectorStore instance with correct vector dimension."""
    return VectorStore(backend=backend)
