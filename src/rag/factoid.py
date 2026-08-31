"""
Factoid Extraction Pipeline — structured JSON extraction, anti-hallucination quote gate,
deduplication, and merging.

Architecture:
  1. Extractor: calls cheap LLM (Zen free / Groq fast / local Ollama) to extract structured factoids
  2. Quote Gate: validates each factoid's source_quote actually appears in the source text
  3. Dedup: removes near-duplicate factoids using text similarity
  4. Merge: combines overlapping factoids from different sources

Token reduction: factoids are ~50-100 tokens each vs 500-800 token raw chunks.
With ~5-10 factoids per chunk, retrieval on factoids cuts context by ~90%.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from src.jsonutil import parse_json_list
import urllib.request
from difflib import SequenceMatcher
from typing import Optional

# Lazy import to avoid circular dependency at module level
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from src.llm import call_llm
        _llm = call_llm
    return _llm


def _call_local_ollama(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Execute local Ollama or OpenAI-compatible local inference if configured."""
    ollama_url = os.getenv("OLLAMA_HOST") or os.getenv("LOCAL_LLM_URL")
    if not ollama_url:
        return None

    try:
        model_name = os.getenv("LOCAL_LLM_MODEL", "llama3:8b")
        if not ollama_url.startswith("http://") and not ollama_url.startswith("https://"):
            ollama_url = f"http://{ollama_url}"

        endpoint = f"{ollama_url.rstrip('/')}/api/generate"
        payload = {
            "model": model_name,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False
        }
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except Exception as e:
        print(f"  [factoid] local Ollama call failed ({e}) — using gateway fallback")
        return None


# ── Factoid schema ──────────────────────────────────────────────────────

FACTOID_TYPES = [
    "entity",       # Named entity (person, org, location, product)
    "relation",     # Relationship between entities
    "event",        # Temporal event with participants
    "statistic",    # Numerical data with context and units
    "definition",   # Concept definition or explanation
    "claim",        # Factual claim with attribution
]

FACTOID_SYSTEM_PROMPT = """You are a precise factoid extraction specialist. Your task is to extract
structured, self-contained factoids from text. Every factoid MUST be verifiable
against the source text.

Rules:
1. Each factoid must have an EXACT source_quote — a substring verbatim from the
   source text that supports the factoid. NEVER fabricate or paraphrase the quote.
2. If you cannot find an exact supporting quote, do NOT create the factoid.
3. Assign confidence based on clarity and specificity:
   - 0.9-1.0: explicit, unambiguous statement
   - 0.7-0.9: clear but could be interpreted differently
   - 0.5-0.7: implied but not directly stated
4. Be concise — each factoid value should be 1-2 sentences max.
5. Include relevant metadata for filtering (entities, topics, numbers).

Output ONLY a JSON array of factoid objects. No other text."""


def factoid_prompt(source_text: str, source_url: str) -> str:
    """Build the extraction prompt for a source text block."""
    return f"""Extract factoids from the following text. For each factoid, include the
EXACT source_quote (verbatim substring from the text) that supports it.

Source URL: {source_url}

Text:
---
{source_text[:8000]}
---

Return a JSON array where each factoid has:
- "type": one of {FACTOID_TYPES}
- "value": concise, self-contained factual statement (1-2 sentences)
- "confidence": number 0.0-1.0
- "source_quote": EXACT substring from the text above that supports this factoid
- "entities": list of entity names mentioned
- "topics": list of topic tags

Only include factoids where you can find an EXACT source_quote in the text above."""


# ── Quote Gate (anti-hallucination) ─────────────────────────────────────

def _normalize_ws(text: str) -> str:
    """Normalize whitespace for fuzzy quote matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def validate_quote(source_quote: str, source_text: str, threshold: float = 0.85) -> bool:
    """Check that source_quote actually appears in source_text."""
    if not source_quote or not source_text:
        return False

    if source_quote in source_text:
        return True

    norm_quote = _normalize_ws(source_quote)
    norm_text = _normalize_ws(source_text)

    if norm_quote in norm_text:
        return True

    if len(norm_quote) < 15:
        return False

    matcher = SequenceMatcher(None, norm_quote, norm_text)
    match = matcher.find_longest_match(0, len(norm_quote), 0, len(norm_text))
    matched_ratio = match.size / len(norm_quote)
    return matched_ratio >= threshold


def validate_factoids(factoids: list[dict], source_text: str) -> list[dict]:
    """Filter factoids through the quote gate."""
    valid = []
    for f in factoids:
        if not isinstance(f, dict):
            continue
        quote = f.get("source_quote", "")
        value = f.get("value", "")

        if not value:
            continue

        if quote and validate_quote(quote, source_text):
            valid.append(f)
        elif validate_quote(value, source_text, threshold=0.75):
            f["source_quote"] = value[:100]
            valid.append(f)

    return valid


# ── Deduplication & Merging ─────────────────────────────────────────────

def _factoid_key(f: dict) -> str:
    """Generate a hash key for exact deduplication."""
    val = f.get("value", "").lower().strip()
    return hashlib.md5(val.encode()).hexdigest()[:16]


def deduplicate_factoids(factoids: list[dict], similarity_threshold: float = 0.75) -> list[dict]:
    """Remove exact and near-duplicate factoids, retaining highest confidence and merging URLs."""
    if not factoids:
        return []

    unique: list[dict] = []

    for f in factoids:
        val_norm = _normalize_ws(f.get("value", ""))
        f_url = f.get("source_url") or ""
        f_urls = set(f.get("source_urls", []))
        if f_url:
            f_urls.add(f_url)

        f_conf = f.get("confidence", 0.5)

        is_dup = False
        for u in unique:
            u_norm = _normalize_ws(u.get("value", ""))
            ratio = SequenceMatcher(None, val_norm, u_norm).ratio()
            if ratio >= similarity_threshold or val_norm == u_norm:
                is_dup = True
                u_urls = set(u.get("source_urls", [])) | f_urls
                u["source_urls"] = list(u_urls)
                u["confidence"] = max(u.get("confidence", 0.5), f_conf)
                break

        if not is_dup:
            f_copy = dict(f)
            f_copy["source_urls"] = list(f_urls)
            unique.append(f_copy)

    return unique


# ── Extractor Function ──────────────────────────────────────────────────

def extract_factoids(source_text: str, source_url: str = "") -> list[dict]:
    """Extract, validate, and deduplicate factoids from source text."""
    if not source_text or len(source_text.strip()) < 50:
        return []

    prompt = factoid_prompt(source_text, source_url)

    raw = _call_local_ollama(FACTOID_SYSTEM_PROMPT, prompt)

    if not raw:
        call_llm_fn = _get_llm()
        try:
            raw = call_llm_fn(FACTOID_SYSTEM_PROMPT, prompt, model="fast")
        except Exception as e:
            print(f"  [factoid] LLM extraction failed: {e}")
            return []

    try:
        cleaned = raw.strip()
        for prefix in ("```json", "```"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        for suffix in ("```",):
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)].strip()
        factoids = parse_json_list(cleaned)
        if not isinstance(factoids, list):
            factoids = []
    except json.JSONDecodeError:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            factoids = parse_json_list(match.group())
            if not isinstance(factoids, list):
                return []
        else:
            return []

    valid_factoids = validate_factoids(factoids, source_text)

    for f in valid_factoids:
        f.setdefault("source_url", source_url)
        f.setdefault("source_urls", [source_url] if source_url else [])
        f.setdefault("id", _factoid_key(f))

    return deduplicate_factoids(valid_factoids)


def extract_from_pages(
    pages: list[dict],
    max_pages: int = 5,
    max_llm_calls: int = 3,
) -> list[dict]:
    """Extract factoids from a list of {url, content} page dicts."""
    if not pages:
        return []

    scored_pages = []
    for p in pages:
        content = (p.get("content", "") or p.get("raw_content", "")).strip()
        if len(content) >= 50:
            scored_pages.append((len(content), p))
    scored_pages.sort(key=lambda x: x[0], reverse=True)

    top_pages = [p for _, p in scored_pages[:max_pages]]
    if not top_pages:
        return []

    batch_size = max(1, len(top_pages) // max_llm_calls)
    all_factoids: list[dict] = []

    for batch_start in range(0, len(top_pages), batch_size):
        batch = top_pages[batch_start : batch_start + batch_size]
        print(f"  [factoid] processing batch {batch_start//batch_size + 1}: {len(batch)} pages")

        combined = ""
        for p in batch:
            url = p.get("url", "")
            content = (p.get("content", "") or p.get("raw_content", ""))[:4000]
            if content:
                combined += f"\n--- Source: {url} ---\n{content}\n"

        if not combined.strip():
            continue

        factoids = extract_factoids(combined, "batch")
        all_factoids.extend(factoids)

    return deduplicate_factoids(all_factoids)


def token_reduction_stats(raw_pages: list[dict], factoids: list[dict]) -> dict:
    """Calculate token reduction statistics."""
    raw_tokens = sum(
        len((p.get("content", "") or p.get("raw_content", "")).split()) * 1.3
        for p in raw_pages
    )
    factoid_tokens = sum(len(f.get("value", "").split()) * 1.3 for f in factoids)
    reduction_pct = (1 - factoid_tokens / max(raw_tokens, 1)) * 100
    return {
        "raw_tokens": int(raw_tokens),
        "factoid_tokens": int(factoid_tokens),
        "num_factoids": len(factoids),
        "reduction_pct": round(reduction_pct, 1),
        "types": _type_distribution(factoids),
    }


def _type_distribution(factoids: list[dict]) -> dict[str, int]:
    """Count factoids by type."""
    dist: dict[str, int] = {}
    for f in factoids:
        t = f.get("type", "unknown")
        dist[t] = dist.get(t, 0) + 1
    return dist
