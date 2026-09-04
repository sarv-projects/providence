"""
SQLite FTS5 full-text search backend.

A lightweight, always-available keyword search fallback when vector embeddings
are unavailable (no API key, no GPU). Uses Python's stdlib sqlite3 module.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from typing import Optional


class FTSStore:
    """SQLite FTS5-based keyword search store."""

    def __init__(self, db_path: str = "") -> None:
        if not db_path:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "fts.db"
            )
        self.db_path = os.path.abspath(db_path)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    url TEXT DEFAULT '',
                    title TEXT DEFAULT '',
                    source_type TEXT DEFAULT '',
                    acl TEXT DEFAULT '',
                    chunk_index INTEGER DEFAULT 0,
                    run_id TEXT DEFAULT '',
                    parent_id TEXT DEFAULT '',
                    parent_text TEXT DEFAULT ''
                )
            """)
            # Migrate old stores: add parent/ACL columns if missing
            cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            for col in ("parent_id", "parent_text", "acl"):
                if col not in cols:
                    conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} TEXT DEFAULT ''")
            # FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(id, text, url, title, content='chunks', content_rowid='rowid')
            """)
            # Triggers to keep FTS in sync
            conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, id, text, url, title)
                    VALUES (new.rowid, new.id, new.text, new.url, new.title);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, id, text, url, title)
                    VALUES ('delete', old.rowid, old.id, old.text, old.url, old.title);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, id, text, url, title)
                    VALUES ('delete', old.rowid, old.id, old.text, old.url, old.title);
                    INSERT INTO chunks_fts(rowid, id, text, url, title)
                    VALUES (new.rowid, new.id, new.text, new.url, new.title);
                END;
            """)
            conn.commit()
            conn.close()

    def upsert(self, chunks: list) -> None:
        """Insert or replace chunks."""
        if not chunks:
            return
        with self._lock:
            conn = self._conn()
            for c in chunks:
                meta = getattr(c, "metadata", {}) or {}
                conn.execute(
                    """INSERT OR REPLACE INTO chunks
                       (id, text, url, title, source_type, acl, chunk_index, run_id, parent_id, parent_text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(c.id),
                        str(c.text)[:10000],
                        str(meta.get("url", "")),
                        str(meta.get("title", "")),
                        str(meta.get("source_type", "")),
                        str(meta.get("acl", ""))[:40],
                        int(meta.get("chunk_index", 0)),
                        str(meta.get("run_id", "")),
                        str(meta.get("parent_id", "")),
                        str(meta.get("parent_text", ""))[:20000],
                    ),
                )
            conn.commit()
            conn.close()

    def query(self, text: str, k: int = 10, filters: Optional[dict] = None) -> list[dict]:
        """Full-text keyword search.

        Args:
            text: Query string.
            k: Max results.
            filters: Optional metadata filter dict, e.g. {source_type, run_id, url}.
        """
        with self._lock:
            conn = self._conn()
            # Clean the query for FTS5. FTS5 treats bare multi-term strings as
            # PHRASES — AND-join quoted terms so non-adjacent keywords match.
            clean = " ".join(text.split())[:200]
            # Keep 2-char terms ("AI", "RL", "ML") — unicode61 yields only
            # alnum runs so short tokens are real terms, not noise.
            tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", clean) if len(t) >= 2][:8]
            if not tokens:
                # Empty / punctuation-only / CJK-only query: FTS MATCH would
                # error and the LIKE fallback ('%%') would match EVERYTHING.
                # Return [] instead of leaking arbitrary rows.
                conn.close()
                return []
            match_clause = (
                " AND ".join(f'"{t}"' for t in tokens) if tokens else f'"{clean}"'
            )
            meta_where = ""
            meta_params: list = []
            for field in ("url", "source_type", "run_id", "acl"):
                val = (filters or {}).get(field)
                if val is not None:
                    meta_where += f" AND c.{field} = ?"
                    meta_params.append(str(val))

            rows = None
            try:
                rows = conn.execute(
                    f"""SELECT c.id, c.text, c.url, c.title, c.source_type, c.acl,
                              c.chunk_index, c.run_id, c.parent_id, c.parent_text, rank
                       FROM chunks_fts f
                       JOIN chunks c ON c.rowid = f.rowid
                       WHERE chunks_fts MATCH ?{meta_where}
                       ORDER BY rank LIMIT ?""",
                    (match_clause, *meta_params, k),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = None

            if rows is None:
                # FTS query parse error (special chars) — fall back to LIKE.
                # Escape LIKE wildcards so "100%" doesn't match "100X".
                like_term = f"%{clean[:50].replace(chr(92), chr(92)*2).replace('%', chr(92)+'%').replace('_', chr(92)+'_')}%"
                rows = conn.execute(
                    f"""SELECT id, text, url, title, source_type, acl, chunk_index, run_id,
                              parent_id, parent_text, 1.0
                        FROM chunks WHERE text LIKE ? ESCAPE '\\'{meta_where.replace('c.', '')} LIMIT ?""",
                    (like_term, *meta_params, k),
                ).fetchall()
            conn.close()

            return [
                {
                    "id": r[0], "text": r[1], "url": r[2], "title": r[3],
                    "source_type": r[4], "acl": r[5] if len(r) > 5 else "",
                    "chunk_index": r[6] if len(r) > 6 else 0, "run_id": r[7] if len(r) > 7 else "",
                    "parent_id": r[8] if len(r) > 8 else "",
                    "parent_text": r[9] if len(r) > 9 else "",
                    "score": float(r[10]) if len(r) > 10 else 1.0,
                }
                for r in rows
            ]

    def delete_by_run(self, run_id: str) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute("DELETE FROM chunks WHERE run_id = ?", (run_id,))
            conn.commit()
            conn.close()

    def count(self) -> int:
        with self._lock:
            conn = self._conn()
            cnt = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            conn.close()
            return cnt
