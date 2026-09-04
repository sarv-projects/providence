"""
Vault — persistent cross-run source storage (full-text global corpus).

Saves all search results, extracted pages, and quality scores to disk
for reuse across research runs. Before making paid API calls, the vault
is checked for recent, high-quality sources on similar queries.

Tier-2 (#13) upgrade — "research once, search web":
  - full_text column: full document text, not just snippets
  - FTS5 virtual table over full_text + title + snippet (not LIKE)
  - metadata columns: domain, source_type, acl, version, updated_at
  - store_pages() for extracted/long documents; store_results() keeps snippets

Storage: SQLite database at data/vault.db (auto-created).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from typing import Optional


class Vault:
    """Persistent, searchable archive of research sources across runs."""

    def __init__(self, db_path: str = "") -> None:
        if not db_path:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "data", "vault.db",
            )
        self.db_path = os.path.abspath(db_path)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT DEFAULT '',
                    snippet TEXT DEFAULT '',
                    full_text TEXT DEFAULT '',
                    domain TEXT DEFAULT '',
                    source_type TEXT DEFAULT 'web',
                    acl TEXT DEFAULT 'public',
                    version TEXT DEFAULT '1',
                    quality_score REAL DEFAULT 5.0,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    seen_count INTEGER DEFAULT 1,
                    topics TEXT DEFAULT '[]',
                    search_queries TEXT DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_sources_domain ON sources(domain);
                CREATE INDEX IF NOT EXISTS idx_sources_quality ON sources(quality_score);
                CREATE INDEX IF NOT EXISTS idx_sources_last_seen ON sources(last_seen);
                CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);
            """)
            # Migrate old vaults (add columns that may not exist yet)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(sources)").fetchall()}
            for col, decl in (
                ("full_text", "TEXT DEFAULT ''"),
                ("acl", "TEXT DEFAULT 'public'"),
                ("version", "TEXT DEFAULT '1'"),
            ):
                if col not in cols:
                    conn.execute(f"ALTER TABLE sources ADD COLUMN {col} {decl}")
            # FTS5 index over full_text + title + snippet (real full-text, not LIKE)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS sources_fts
                USING fts5(url, title, snippet, full_text,
                           content='sources', content_rowid='id')
            """)
            conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS sources_ai AFTER INSERT ON sources BEGIN
                    INSERT INTO sources_fts(rowid, url, title, snippet, full_text)
                    VALUES (new.id, new.url, new.title, new.snippet, new.full_text);
                END;
                CREATE TRIGGER IF NOT EXISTS sources_ad AFTER DELETE ON sources BEGIN
                    INSERT INTO sources_fts(sources_fts, rowid, url, title, snippet, full_text)
                    VALUES ('delete', old.id, old.url, old.title, old.snippet, old.full_text);
                END;
                CREATE TRIGGER IF NOT EXISTS sources_au AFTER UPDATE ON sources BEGIN
                    INSERT INTO sources_fts(sources_fts, rowid, url, title, snippet, full_text)
                    VALUES ('delete', old.id, old.url, old.title, old.snippet, old.full_text);
                    INSERT INTO sources_fts(rowid, url, title, snippet, full_text)
                    VALUES (new.id, new.url, new.title, new.snippet, new.full_text);
                END;
            """)
            conn.commit()
            conn.close()

    def store_results(
        self,
        results: list[dict],
        queries: Optional[list[str]] = None,
    ) -> int:
        """Store search results in the vault.

        Stores the snippet plus any long content already present (Exa returns
        full text in raw_content) — a search hit can become a full-text entry.

        Args:
            results: List of search result dicts with {url, title, content/snippet}.
            queries: Search queries that produced these results (for topic tracking).

        Returns:
            Number of new sources stored.
        """
        if not results:
            return 0

        now = time.time()
        queries_json = json.dumps(queries or [])

        with self._lock:
            conn = self._conn()
            new_count = 0
            for r in results:
                url = r.get("url", "")
                if not url:
                    continue

                title = r.get("title", "")
                snippet = (r.get("content", "") or r.get("snippet", "") or "")[:1000]
                # Full text when available (Exa raw_content, extracted pages)
                raw = r.get("raw_content") or r.get("full_text") or ""
                full_text = raw if len(raw) > 800 else ""
                domain = self._extract_domain(url)
                quality = float(r.get("guard_score", r.get("score", 5.0)))
                source_type = str(r.get("source_type") or r.get("source") or "web")[:40]
                acl = str(r.get("acl") or "public")[:40]
                version = str(r.get("version") or "1")[:20]

                # Generate topic tags from snippet
                topics = json.dumps(self._extract_topics(snippet, title))

                # Upsert: insert or update
                existing = conn.execute(
                    "SELECT id, seen_count, search_queries FROM sources WHERE url = ?",
                    (url,),
                ).fetchone()

                if existing:
                    existing_queries = json.loads(existing[2] or "[]")
                    merged_queries = list(set(existing_queries + (queries or [])))
                    # Preserve good data: only overwrite snippet/title/topics
                    # when the new row actually carries content, and never
                    # demote an established quality score with a lower one
                    # (re-storing a URL with a bare score must not clobber a
                    # 9.0 peer-reviewed entry down to 1.5).
                    conn.execute(
                        """UPDATE sources SET
                            title = CASE WHEN ? = '' THEN title ELSE ? END,
                            snippet = CASE WHEN ? = '' THEN snippet ELSE ? END,
                            full_text = CASE WHEN ? = '' THEN full_text ELSE ? END,
                            source_type = ?, acl = ?, version = ?,
                            quality_score = CASE WHEN ? > quality_score THEN ? ELSE quality_score END,
                            last_seen = ?, seen_count = seen_count + 1,
                            topics = CASE WHEN ? = '[]' THEN topics ELSE ? END,
                            search_queries = ?
                            WHERE url = ?""",
                        (title, title, snippet, snippet, full_text, full_text,
                         source_type, acl, version,
                         quality, quality, now, topics, topics,
                         json.dumps(merged_queries), url),
                    )
                else:
                    conn.execute(
                        """INSERT INTO sources
                           (url, title, snippet, full_text, domain, source_type, acl,
                            version, quality_score, first_seen, last_seen, topics,
                            search_queries)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (url, title, snippet, full_text, domain, source_type, acl,
                         version, quality, now, now, topics, queries_json),
                    )
                    new_count += 1

            conn.commit()
            conn.close()
            return new_count

    def store_pages(
        self,
        pages: list[dict],
        queries: Optional[list[str]] = None,
    ) -> int:
        """Store extracted/long documents as full-text vault entries.

        This is the "research once, search web" primitive: every page actually
        fetched during a run becomes a searchable full-text corpus entry.

        Args:
            pages: List of {url, title, content/raw_content, source_type?, acl?}.
            queries: Optional queries for topic tracking.

        Returns:
            Number of new sources stored.
        """
        if not pages:
            return 0
        entries = []
        for p in pages:
            content = (p.get("content") or p.get("raw_content") or "")[:100_000]
            if not content.strip():
                continue
            entries.append({
                "url": p.get("url", ""),
                "title": p.get("title", ""),
                "raw_content": content,
                "content": content[:1000],
                "source_type": p.get("source_type") or p.get("source") or "web",
                "acl": p.get("acl") or "public",
                "guard_score": p.get("guard_score", p.get("score", 5.0)),
            })
        return self.store_results(entries, queries=queries)

    def search(
        self,
        query: str,
        k: int = 10,
        min_quality: float = 3.0,
        max_age_days: float = 90.0,
        filters: Optional[dict] = None,
        include_full_text: bool = False,
    ) -> list[dict]:
        """Search the vault for sources matching a query.

        Uses the FTS5 index over full_text + title + snippet (falling back to
        LIKE only for unparseable queries), ranked by quality and recency.

        Args:
            query: Search query.
            k: Max results.
            min_quality: Minimum quality_score (0-10).
            max_age_days: Only include sources seen within this many days.
            filters: Optional metadata filter dict, e.g. {domain, source_type, acl}.
            include_full_text: Also return the full_text field (truncated).

        Returns:
            List of {url, title, snippet, domain, source_type, acl, version,
            quality_score, last_seen, seen_count, full_text?}.
        """
        with self._lock:
            conn = self._conn()
            cutoff = time.time() - max_age_days * 86400

            # FTS5 MATCH first; fall back to LIKE on parse errors.
            # FTS5 treats bare multi-term strings as PHRASES — AND-join quoted
            # terms so non-adjacent keywords still match (real full-text search).
            fts_q = " ".join(query.lower().split())[:200]
            fts_terms = [
                t for t in re.split(r"[^a-zA-Z0-9]+", fts_q) if len(t) > 2
            ][:8]
            if not fts_terms:
                # Empty / punctuation-only query: the LIKE fallback ('%%')
                # would dump arbitrary vault rows. Return [] instead.
                conn.close()
                return []
            match_clause = (
                " AND ".join(f'"{t}"' for t in fts_terms) if fts_terms else f'"{fts_q}"'
            )
            meta_conditions = []
            meta_params: list = []
            for field in ("domain", "source_type", "acl"):
                val = (filters or {}).get(field)
                if val:
                    meta_conditions.append(f"{field} = ?")
                    meta_params.append(str(val))
            meta_where = (" AND " + " AND ".join(meta_conditions)) if meta_conditions else ""

            # FTS virtual table shares column names with sources → qualify.
            # (LIKE fallback selects from `sources` alone, so it uses the plain list.)
            select_cols_qual = (
                "s.url, s.title, s.snippet, s.domain, s.source_type, s.acl, "
                "s.version, s.quality_score, s.last_seen, s.seen_count"
            )
            select_cols_plain = (
                "url, title, snippet, domain, source_type, acl, version, "
                "quality_score, last_seen, seen_count"
            )
            if include_full_text:
                select_cols_qual += ", substr(s.full_text, 1, 6000)"
                select_cols_plain += ", substr(full_text, 1, 6000)"

            rows = None
            try:
                rows = conn.execute(
                    f"""
                    SELECT {select_cols_qual}
                    FROM sources_fts f
                    JOIN sources s ON s.id = f.rowid
                    WHERE sources_fts MATCH ?
                      AND s.quality_score >= ?
                      AND s.last_seen >= ?
                      {meta_where}
                    ORDER BY s.quality_score DESC, s.last_seen DESC
                    LIMIT ?
                    """,
                    (match_clause, min_quality, cutoff, *meta_params, k),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = None

            if rows is None:
                # LIKE fallback (FTS parse error on exotic queries)
                like_term = f"%{fts_q[:60]}%"
                rows = conn.execute(
                    f"""
                    SELECT {select_cols_plain}
                    FROM sources
                    WHERE (full_text LIKE ? OR snippet LIKE ? OR title LIKE ?)
                      AND quality_score >= ? AND last_seen >= ?
                      {meta_where}
                    ORDER BY quality_score DESC, last_seen DESC
                    LIMIT ?
                    """,
                    (like_term, like_term, like_term, min_quality, cutoff, *meta_params, k),
                ).fetchall()

            conn.close()
            out = []
            for r in rows:
                entry = {
                    "url": r[0], "title": r[1], "snippet": r[2],
                    "domain": r[3], "source_type": r[4], "acl": r[5],
                    "version": r[6], "quality_score": r[7],
                    "last_seen": r[8], "seen_count": r[9],
                }
                if include_full_text and len(r) > 10:
                    entry["full_text"] = r[10]
                out.append(entry)
            return out

    def stats(self) -> dict:
        """Return vault statistics."""
        with self._lock:
            conn = self._conn()
            total = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
            domains = conn.execute(
                "SELECT COUNT(DISTINCT domain) FROM sources"
            ).fetchone()[0]
            avg_quality = conn.execute(
                "SELECT AVG(quality_score) FROM sources"
            ).fetchone()[0] or 0
            full_text_count = conn.execute(
                "SELECT COUNT(*) FROM sources WHERE length(full_text) > 800"
            ).fetchone()[0]
            conn.close()
            return {
                "total_sources": total,
                "unique_domains": domains,
                "avg_quality": round(avg_quality, 1),
                "full_text_entries": full_text_count,
            }

    @staticmethod
    def _extract_domain(url: str) -> str:
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""

    @staticmethod
    def _extract_topics(snippet: str, title: str) -> list[str]:
        """Extract simple topic keywords from snippet+title."""
        text = f"{title} {snippet}".lower()
        # Common topic words (simple heuristic)
        topic_words = {
            "ai", "ml", "quantum", "research", "study", "paper",
            "algorithm", "model", "data", "science", "engineering",
            "medicine", "climate", "energy", "space", "biology",
            "physics", "chemistry", "math", "computer", "network",
            "security", "privacy", "blockchain", "crypto", "web",
            "mobile", "cloud", "server", "database", "linux",
        }
        return [w for w in topic_words if w in text][:10]
