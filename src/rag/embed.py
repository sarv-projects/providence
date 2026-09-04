"""
Embedding system for the RAG pipeline.

Uses OpenAI embeddings (text-embedding-3-small by default) with:
- In-memory caching to avoid re-embedding identical text
- Batch processing for efficiency
- Rate limit handling

Falls back gracefully when no API key is available.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Optional

import urllib.error
import urllib.request


class Embedder:
    """Protocol: embed text into a vector."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-small (or custom model) embedder."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        cache_size: int = 5000,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self._cache: dict[str, list[float]] = {}
        self._cache_order: list[str] = []
        self._cache_size = cache_size
        self._lock = threading.RLock()
        self._request_count = 0

    def embed(self, text: str) -> list[float]:
        """Embed a single text, using cache if available."""
        if not self.api_key:
            raise RuntimeError("No OpenAI API key configured for embeddings")

        cache_key = hashlib.sha256(text.encode()).hexdigest()

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        vec = self._call_api(text)

        with self._lock:
            self._cache[cache_key] = vec
            self._cache_order.append(cache_key)
            while len(self._cache_order) > self._cache_size:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)

        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed multiple texts."""
        if not self.api_key:
            raise RuntimeError("No OpenAI API key configured for embeddings")
        if not texts:
            return []

        # Check cache first
        results = []
        uncached: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            with self._lock:
                if cache_key in self._cache:
                    results.append((i, self._cache[cache_key]))
                    continue
            uncached.append((i, text))

        if uncached:
            # Batch API call
            remaining = [t for _, t in uncached]
            vecs = self._call_batch_api(remaining)
            # Defensive: provider must return one vector per input in order.
            # Fill gaps individually rather than misaligning embeddings.
            pending: list[tuple[int, list[float]]] = []
            for idx, (orig_idx, _) in enumerate(uncached):
                if idx < len(vecs):
                    pending.append((orig_idx, vecs[idx]))
            with self._lock:
                for orig_idx, vec in pending:
                    results.append((orig_idx, vec))
                    cache_key = hashlib.sha256(texts[orig_idx].encode()).hexdigest()
                    if cache_key not in self._cache:
                        self._cache[cache_key] = vec
                        self._cache_order.append(cache_key)
                while len(self._cache_order) > self._cache_size:
                    old = self._cache_order.pop(0)
                    self._cache.pop(old, None)
            missing = [t for i, t in uncached if i >= len(vecs)]
            for orig_idx, text in missing:
                results.append((orig_idx, self.embed(text)))

        results.sort(key=lambda x: x[0])
        return [v for _, v in results]

    def _call_api(self, text: str) -> list[float]:
        data = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        url = f"{self.base_url}/embeddings"
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "AutonomousResearchAgent/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = json.loads(resp.read().decode())
                vec = body["data"][0]["embedding"]
                self._request_count += 1
                return [float(v) for v in vec]
        except Exception as e:
            raise RuntimeError(f"Embedding API error: {e}")

    def _call_batch_api(self, texts: list[str]) -> list[list[float]]:
        data = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        url = f"{self.base_url}/embeddings"
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "AutonomousResearchAgent/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())
                items = body["data"]
                # Map by the API's `index` field when present — never assume
                # response order matches request order.
                by_index: dict[int, list[float]] = {}
                ordered: list[list[float]] = []
                for item in items:
                    if isinstance(item, dict) and "index" in item:
                        try:
                            by_index[int(item["index"])] = [float(v) for v in item["embedding"]]
                        except (ValueError, TypeError):
                            continue
                    else:
                        ordered.append([float(v) for v in item["embedding"]])
                if by_index and len(by_index) == len(texts):
                    vecs = [by_index[i] for i in range(len(texts))]
                else:
                    vecs = ordered if len(ordered) == len(texts) else [
                        [float(v) for v in item["embedding"]] for item in items
                    ]
                self._request_count += len(texts)
                return vecs
        except Exception as e:
            raise RuntimeError(f"Embedding batch API error: {e}")


class DummyEmbedder(Embedder):
    """Fallback embedder that uses hash bytes as pseudo-vectors (for testing/no-API).

    Produces deterministic non-NaN vectors from text hash — safe for LanceDB/FTS.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-embedding from text hash.

        Uses hash bytes normalized to [0, 1] then shifted to [-1, 1].
        This avoids NaN/Inf issues with struct.unpack on arbitrary byte sequences.
        """
        h = hashlib.sha256(text.encode()).digest()
        vec: list[float] = []
        for b in h:
            vec.append((b / 255.0) * 2.0 - 1.0)  # range [-1, 1]
        if len(vec) < self.dim:
            vec = (vec * (self.dim // len(vec) + 1))[:self.dim]
        return vec[:self.dim]


class BagOfWordsEmbedder(Embedder):
    """Lightweight local embedder: hashed bag-of-words (better than single SHA vector).

    Produces dim-dimensional vectors without any API. Useful hybrid with FTS.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        # L2 normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def get_embedder(dim: int = 1536) -> Embedder:
    """Get the appropriate embedder based on environment.

    Priority:
      1. EMBEDDING_API_KEY / OPENAI_EMBEDDING_KEY → OpenAI embeddings
      2. USE_CHAT_KEY_FOR_EMBEDDINGS=1 + OPENAI_API_KEY → OpenAI embeddings
      3. BagOfWordsEmbedder (local hashed BoW, dim=384 padded to store dim)
      4. DummyEmbedder last resort
    """
    api_key = (
        os.getenv("EMBEDDING_API_KEY", "")
        or os.getenv("OPENAI_EMBEDDING_KEY", "")
    )
    if not api_key and os.getenv("USE_CHAT_KEY_FOR_EMBEDDINGS", "").lower() in (
        "1", "true", "yes",
    ):
        api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        return OpenAIEmbedder(api_key=api_key)

    # Prefer BoW over pure Dummy for slightly better dense ranking with FTS hybrid
    use_bow = os.getenv("EMBEDDING_LOCAL", "bow").lower()
    if use_bow in ("bow", "bag", "1", "true", ""):
        # Store often expects 1536; BoW uses smaller dim then pad in store
        return BagOfWordsEmbedder(dim=min(dim, 384) if dim > 384 else dim)
    return DummyEmbedder(dim=dim)
