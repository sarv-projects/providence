"""Clarifying-questions prelude for ambiguous research queries (ChatGPT-style probing)."""

from __future__ import annotations

import json
import re
from typing import Any

from src.jsonutil import parse_json_dict


AMBIGUITY_HINTS = (
    "best", "better", "should i", "which", "recommend", "vs", "versus",
    "help me", "anything about", "tell me about", "overview of", "what about",
)


def is_ambiguous(query: str) -> bool:
    """Heuristic: short, vague, or multi-interpretation research intents."""
    q = (query or "").strip()
    if not q:
        return True
    words = re.findall(r"[a-zA-Z0-9]+", q.lower())
    if len(words) <= 4:
        return True
    if len(words) <= 10 and any(h in q.lower() for h in AMBIGUITY_HINTS):
        # "compare X vs Y" is specific enough if both nouns present
        if " vs " in q.lower() or " versus " in q.lower():
            return len(words) < 6
        return True
    # Pronoun-heavy without subject
    if re.search(r"\b(it|this|that|they)\b", q.lower()) and len(words) < 12:
        return True
    return False


def generate_clarifying_questions(query: str, max_questions: int = 4) -> dict[str, Any]:
    """Return {ambiguous, questions, assumptions, refined_query_hint}.

    Uses a fast LLM when available; falls back to template questions.
    """
    ambiguous = is_ambiguous(query)
    fallback = {
        "ambiguous": ambiguous,
        "questions": [
            "What is the primary goal of this research (overview, decision, literature survey, implementation guide)?",
            "What time range or currency matters (e.g. only 2024–2026, or historical)?",
            "Any preferred source types (academic papers, industry blogs, official docs)?",
            "Any systems, vendors, or domains to include or exclude?",
        ][:max_questions],
        "assumptions": [
            "Produce a technical research report with citations.",
            "Prefer recent high-quality sources when available.",
        ],
        "refined_query_hint": query,
    }
    if not ambiguous:
        return {
            "ambiguous": False,
            "questions": [],
            "assumptions": fallback["assumptions"],
            "refined_query_hint": query,
        }

    try:
        from src.llm import call_llm
        prompt = f"""The user research query may be ambiguous:
"{query}"

Return JSON only:
  - "ambiguous": true/false
  - "questions": 2-{max_questions} short clarifying questions (list of strings)
  - "assumptions": 1-3 assumptions you would make if no answers
  - "refined_query_hint": one improved query string incorporating likely defaults
"""
        raw = call_llm(
            "You clarify research scope. Return valid JSON only.",
            prompt,
            model="fast",
        )
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = parse_json_dict(cleaned)
        if not isinstance(data, dict):
            return fallback
        qs = data.get("questions") or []
        if not isinstance(qs, list):
            qs = fallback["questions"]
        return {
            "ambiguous": bool(data.get("ambiguous", True)),
            "questions": [str(q) for q in qs[:max_questions]],
            "assumptions": list(data.get("assumptions") or fallback["assumptions"])[:5],
            "refined_query_hint": str(data.get("refined_query_hint") or query),
        }
    except Exception:
        return fallback


def apply_clarifications(
    query: str,
    clarifications: dict[str, str] | None = None,
    assumptions: list[str] | None = None,
) -> str:
    """Fold Q&A + assumptions into an enriched research query string."""
    parts = [query.strip()]
    if clarifications:
        answered = [f"Q: {k} A: {v}" for k, v in clarifications.items() if v]
        if answered:
            parts.append("User clarifications:\n" + "\n".join(answered))
    if assumptions:
        parts.append("Assumptions:\n" + "\n".join(f"- {a}" for a in assumptions if a))
    return "\n\n".join(parts)
