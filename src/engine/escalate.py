"""Shared chat→research escalation heuristic.

Single source of truth used by both the CLI chat loop (main.py) and the web
chat endpoint (src/web) — previously two divergent copies of the same logic.
"""

TRIGGERS = (
    "research", "deep dive", "comprehensive", "compare ", " vs ",
    "versus", "literature review", "survey of", "write a report",
    "investigate", "analyze in depth", "pros and cons",
)


def should_escalate_to_research(text: str) -> bool:
    """Heuristic: long / multi-part / explicit research intent → deep research."""
    t = (text or "").lower().strip()
    if len(t.split()) < 8:
        return False
    return any(x in t for x in TRIGGERS)
