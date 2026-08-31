"""Robust JSON parsing shared by all agents (stolen from GPT Researcher's
``json_repair`` + shape-fallback parsing patterns).

Layered recovery used instead of bare ``json.loads`` after every LLM prompt:
  1. exact ``json.loads`` on the fence-stripped text
  2. ``json_repair`` (tolerant of unterminated strings, trailing commas, …)
  3. regex extraction of the first ``{...}`` / ``[...]`` block
  4. caller-provided default

This removes the try/except+fallback duplication at every call site and makes
the repair tier (not just the fallback) explicit.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # json-repair is optional — parse still degrades to regex + default
    from json_repair import loads as _json_repair_loads

    _HAS_JSON_REPAIR = True
except Exception:  # pragma: no cover - environment without json-repair
    _HAS_JSON_REPAIR = False

_FENCE_RE = re.compile(r"```(?:json)?\s*", re.I)
_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def strip_code_fences(raw: str) -> str:
    """Remove ```json ... ``` / ``` ... ``` wrappers and trimming whitespace."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text, count=1).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _first_block(text: str, pattern: re.Pattern) -> Optional[str]:
    m = pattern.search(text)
    return m.group(0) if m else None


def parse_json_lenient(raw: str, default: Any = None) -> Any:
    """Parse LLM text as JSON with escalating recovery. Returns ``default``
    when every attempt fails.

    Order of attempts:
      1. regex-extracted first ``{...}`` / ``[...]`` block (handles trailing
         prose / leading CoT the fastest),
      2. the whole fence-stripped text via ``json.loads``,
      3. the same candidates through ``json_repair`` (tolerant parse),
    so a junk-prefixed response never short-circuits a valid embedded block.
    """
    text = strip_code_fences(raw)
    if not text:
        return default

    candidates: list[str] = []
    if not text.startswith(("{", "[")):
        block = _first_block(text, _OBJECT_RE) or _first_block(text, _ARRAY_RE)
        if block and block != text:
            candidates.append(block)
    candidates.append(text)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass
        if _HAS_JSON_REPAIR:
            try:
                value = _json_repair_loads(candidate)
            except Exception:
                continue
            if value or candidate is candidates[-1]:
                return value
    return default


def parse_json_dict(raw: str, default: Optional[dict] = None) -> dict:
    """Parse a JSON object; falls back to ``default`` (or ``{}``) on failure or
    when the parsed value is not a dict."""
    if default is None:
        default = {}
    data = parse_json_lenient(raw, default=default)
    return data if isinstance(data, dict) else default


def parse_json_list(raw: str, default: Optional[list] = None) -> list:
    """Parse a JSON array; falls back to ``default`` (or ``[]``) on failure or
    when the parsed value is not a list. When the lenient parser returns a dict
    but the raw text contains an array (e.g. prose-wrapped ``[ {...} ]``), the
    array block is parsed explicitly so outline-style calls never lose data."""
    if default is None:
        default = []
    data = parse_json_lenient(raw, default=default)
    if isinstance(data, list):
        return data
    text = strip_code_fences(raw)
    block = _first_block(text, _ARRAY_RE) if text else None
    if block and block != text:
        arr = parse_json_lenient(block, default=default)
        if isinstance(arr, list):
            return arr
    return data if isinstance(data, list) else default