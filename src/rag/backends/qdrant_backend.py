"""
Qdrant vector store backend.

Qdrant is a production-grade vector database. It requires a running Qdrant
instance (local or cloud). Set QDRANT_URL and QDRANT_API_KEY to enable.

Auto-detection: if qdrant_client is importable and QDRANT_URL is set,
this backend registers. Otherwise it defers to LanceDB (default).

Env:
    QDRANT_URL=http://localhost:6333      # local instance
    QDRANT_API_KEY=...                     # cloud instance
    QDRANT_COLLECTION=research_chunks      # override collection name
"""

from __future__ import annotations

import os
from typing import Optional

# Graceful import — only load if qdrant_client is installed
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def qdrant_is_available() -> bool:
    """Check if Qdrant is configured and reachable."""
    if not QDRANT_AVAILABLE:
        return False
    url = os.getenv("QDRANT_URL", "")
    return bool(url)


class QdrantStore:
    """Qdrant-backed vector store for RAG chunks.

    Uses a single collection with dense vectors (default 1536-dim)
    and payload fields matching the chunk schema.
    """

    def __init__(
        self,
        vector_dim: int = 1536,
        url: str = "",
        api_key: str = "",
        collection_name: str = "",
    ) -> None:
        if not QDRANT_AVAILABLE:
            raise RuntimeError(
                "qdrant-client not installed. Run: uv add qdrant-client"
            )

        self.vector_dim = vector_dim
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", "")
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", "research_chunks"
        )

        if self.api_key:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self._client = QdrantClient(url=self.url)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = [
            c.name for c in self._client.get_collections().collections
        ]
        if self.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, chunks: list) -> None:
        """Insert or update chunks in Qdrant."""
        if not chunks:
            return

        points = []
        for c in chunks:
            meta = getattr(c, "metadata", {}) or {}
            vec = getattr(c, "embedding", None) or []
            vec_float = [float(v) for v in vec]

            # Pad or truncate
            if len(vec_float) < self.vector_dim:
                vec_float.extend([0.0] * (self.vector_dim - len(vec_float)))
            vec_float = vec_float[:self.vector_dim]

            # Qdrant point IDs must be ints or UUIDs — derive a stable UUID
            # from the chunk id (raw md5-hex / scoped strings are rejected).
            import uuid as _uuid
            try:
                _uuid.UUID(str(c.id))
                point_id = str(c.id)
            except (ValueError, AttributeError, TypeError):
                point_id = str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"providence-chunk:{c.id}"))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec_float,
                    payload={
                        "chunk_id": str(c.id),
                        "text": str(c.text),
                        "url": str(meta.get("url", "")),
                        "title": str(meta.get("title", "")),
                        "source_type": str(meta.get("source_type", "")),
                        "chunk_index": int(meta.get("chunk_index", 0)),
                        "run_id": str(meta.get("run_id", "")),
                    },
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def query(
        self,
        embedding: list[float],
        k: int = 10,
        run_id: str = "",
    ) -> list[dict]:
        """Query by vector similarity. Optionally filter by run_id."""
        vec = [float(v) for v in embedding]
        if len(vec) < self.vector_dim:
            vec.extend([0.0] * (self.vector_dim - len(vec)))
        vec = vec[:self.vector_dim]

        query_filter = None
        if run_id:
            query_filter = Filter(
                must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))]
            )

        try:
            results = self._client.query_points(
                collection_name=self.collection_name,
                query=vec,
                limit=k,
                query_filter=query_filter,
            ).points
        except Exception:
            return []

        return [
            {
                "id": str(r.payload.get("chunk_id") or r.id),
                "text": r.payload.get("text", ""),
                "url": r.payload.get("url", ""),
                "title": r.payload.get("title", ""),
                "source_type": r.payload.get("source_type", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "run_id": r.payload.get("run_id", ""),
                "score": float(r.score),
            }
            for r in results
        ]

    def delete_by_run(self, run_id: str) -> None:
        """Delete all chunks for a given run_id."""
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))]
            ),
        )

    def count(self) -> int:
        """Return total number of stored chunks."""
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count
        except Exception:
            return 0
