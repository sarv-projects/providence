"""Persistent memory for past searches using a JSON file.

Concurrency-safe: all reads/writes go through a module lock, writes are
atomic (temp file + os.replace), and a corrupt file is backed up instead of
being silently reset (which previously destroyed the entire history).
"""

import json
import os
import threading
import time

MEMORY_FILE = os.path.expanduser("~/.providence_memory.json")
_LOCK = threading.RLock()


def _load() -> list[dict]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        # Back up the corrupt file so it can be recovered manually instead
        # of silently destroying history on the next save.
        try:
            backup = MEMORY_FILE + f".corrupt.{int(time.time())}"
            os.replace(MEMORY_FILE, backup)
            print(f"[memory] corrupt memory file backed up to {backup}")
        except OSError:
            pass
        return []


def _save(memory: list[dict]) -> None:
    # Atomic write: write to a temp file in the same directory, then replace.
    tmp = MEMORY_FILE + f".tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f:
        json.dump(memory[-50:], f, indent=2)  # keep last 50
    os.replace(tmp, MEMORY_FILE)


def save_search(query: str, search_queries: list[str], report_path: str, findings: list[str]) -> None:
    """Save a completed search to memory."""
    with _LOCK:
        memory = _load()
        entry = {
            "query": query,
            "search_queries": search_queries,
            "report_path": report_path,
            "findings_count": len(findings),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        memory.append(entry)
        _save(memory)


def get_history(limit: int = 5) -> list[dict]:
    """Get recent search history."""
    with _LOCK:
        memory = _load()
    return memory[-limit:]


def find_similar(query: str) -> list[dict]:
    """Simple keyword match to find past searches on similar topics."""
    with _LOCK:
        memory = _load()
    keywords = set(query.lower().split())
    matches = []
    for entry in memory:
        entry_words = set(entry["query"].lower().split())
        overlap = keywords & entry_words
        if len(overlap) >= 2:
            matches.append(entry)
    return matches[-3:]  # last 3 matches
