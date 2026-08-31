"""
Triangulator agent — adversarial bias mitigation via Pro/Con/Neutral agents.

Architecture:
  1. Detection: heuristic check if query is subjective/controversial
  2. Three sub-agents run in parallel on different provider routes:
     - Pro: argues FOR the proposition
     - Con: argues AGAINST the proposition
     - Neutral: presents a balanced view
  3. Synthesis Arbiter: compares outputs, identifies bias, generates neutral synthesis
  4. Results enrich the state's findings with bias-cancelled perspectives

Integration: runs after Critic approves synthesis, before Synthesizer outline.
Only triggers on accurate/comprehensive quality dials or subjective queries.
"""

import json
import re
import threading
import time

from src.jsonutil import parse_json_dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.llm import call_llm as _call_llm
from src.state import ResearchState
from .registry import register

# ── Sub-agent system prompts ───────────────────────────────────────────
PRO_PROMPT = (
    "You are a passionate advocate. Your task is to argue STRONGLY IN FAVOR of "
    "the following proposition. Present the best evidence, strongest arguments, "
    "and most compelling case. Be persuasive. Acknowledge counterpoints only to "
    "refute them. Output as a structured argument with key points."
)

CON_PROMPT = (
    "You are a skeptical critic. Your task is to argue STRONGLY AGAINST the "
    "following proposition. Present the strongest counter-arguments, identify "
    "weaknesses, risks, and downsides. Be thorough in your critique. "
    "Output as a structured argument with key points."
)

NEUTRAL_PROMPT = (
    "You are a balanced analyst. Your task is to present a FAIR, NEUTRAL "
    "assessment of the following proposition. Weigh both sides equally. "
    "Identify areas of agreement and genuine disagreement. Do not take sides. "
    "Output as a structured balanced analysis with key points from all perspectives."
)

ARBITER_PROMPT = (
    "You are a bias detection specialist. Your task is to compare three analyses "
    "(pro, con, neutral) of the same proposition and:\n"
    "1. Identify biased framing or one-sided arguments in each\n"
    "2. Find common ground across all three perspectives\n"
    "3. Generate a balanced, bias-mitigated synthesis\n"
    "4. Assign a bias score (0-10, where 0=no bias, 10=extreme bias)\n"
    "5. Recommend which perspective(s) are most credible\n"
    "Output as JSON with keys: bias_assessment, common_ground, synthesis, bias_score, credibility"
)

# ── Subjective query detection ──────────────────────────────────────────
SUBJECTIVE_PATTERNS = [
    r"\b(best|better|worse|worst)\b",
    r"\b(vs\.?|versus)\b",
    r"\b(should|must|ought)\b",
    r"\b(pros?|cons?|advantages?|disadvantages?|benefits?|drawbacks?)\b",
    r"\b(debate|controvers|dispute|disagree)\b",
    r"\b(ethical|moral|right|wrong)\b",
    r"\b(opinion|believe|argue|claim)\b",
    r"\b(which [\w ]+ is (better|best|prefer))\b",
    r"\b(compare|comparison|contrast)\b",
    r"\b(against|opposing|supporter|critic)\b",
]

# Rate limiting
_last_triangulation = [0.0]
_tri_lock = threading.RLock()
MIN_TRI_INTERVAL = 5.0
MAX_TRI_PER_RUN = 3
_tri_count = [0]


def _is_subjective(query: str) -> bool:
    """Heuristic: does this query look subjective/controversial?"""
    query_lower = query.lower()
    for pattern in SUBJECTIVE_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    # Also trigger if query is explicitly comparative
    if " vs " in query_lower or " versus " in query_lower:
        return True
    return False


def _should_triangulate(state: ResearchState) -> bool:
    """Check and claim a triangulation slot (thread-safe).

    Returns True if triangulation should proceed, AND atomically
    claims the slot by incrementing the counter + updating timestamp.
    """
    with _tri_lock:
        if _tri_count[0] >= MAX_TRI_PER_RUN:
            return False
        if time.time() - _last_triangulation[0] < MIN_TRI_INTERVAL:
            return False
        # Claim slot atomically — prevents race conditions
        _tri_count[0] += 1
        _last_triangulation[0] = time.time()

    query = state.get("query", "")
    if _is_subjective(query):
        return True

    # Also trigger if findings are numerous and diverse
    claims = state.get("claims", [])
    if len(claims) >= 8:
        return True

    # Release claimed slot if we decided not to triangulate
    with _tri_lock:
        _tri_count[0] = max(0, _tri_count[0] - 1)
        _last_triangulation[0] = 0.0  # reset timestamp so next call isn't blocked
    return False


@register("triangulator")
def triangulator(state: ResearchState) -> ResearchState:
    """Run adversarial triangulation for bias mitigation.

    Skips if query is not subjective or rate limits are hit.
    Enriches state findings with balanced perspectives.
    """
    if not _should_triangulate(state):
        return state

    query = state["query"]
    state["status"] = "Triangulating for bias mitigation..."
    print(f"\n⚖️  [Triangulator] Running adversarial triangulation for: {query[:80]}")

    # Build proposition from query + findings
    findings_summary = "\n".join(state.get("findings", [])[:10])
    proposition = f"Proposition: {query}\n\nSupporting context:\n{findings_summary[:2000]}"

    # Run all three agents in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_call_llm, PRO_PROMPT, f"Argue FOR: {proposition}", model="fast"): "pro",
            executor.submit(_call_llm, CON_PROMPT, f"Argue AGAINST: {proposition}", model="fast"): "con",
            executor.submit(_call_llm, NEUTRAL_PROMPT, f"Present balanced: {proposition}", model="fast"): "neutral",
        }
        for future in as_completed(futures):
            role = futures[future]
            try:
                results[role] = future.result(timeout=120)
                print(f"  [{role}] {len(results[role])} chars")
            except Exception as e:
                print(f"  [{role}] failed: {e}")
                results[role] = f"[{role} perspective unavailable: {e}]"
                future.cancel()

    # Run Synthesis Arbiter
    arbiter_input = (
        f"PROPOSITION: {query}\n\n"
        f"=== PRO ARGUMENT ===\n{results.get('pro', 'N/A')[:1500]}\n\n"
        f"=== CON ARGUMENT ===\n{results.get('con', 'N/A')[:1500]}\n\n"
        f"=== NEUTRAL ANALYSIS ===\n{results.get('neutral', 'N/A')[:1500]}"
    )
    try:
        arbiter_raw = _call_llm(ARBITER_PROMPT, arbiter_input, model="strong")
        # Robust JSON extraction: strip code fences, try raw, fallback to regex
        arbiter = parse_json_dict(arbiter_raw, default={})
        if not arbiter:
            arbiter = {"bias_score": 5, "synthesis": "Arbiter unavailable", "error": "parse_failed"}
    except Exception as e:
        arbiter = {"bias_score": 5, "synthesis": "Arbiter unavailable", "error": str(e)}

    # Enrich state with triangulation results
    bias_score = arbiter.get("bias_score", 5)
    synthesis = arbiter.get("synthesis", "")
    common_ground = arbiter.get("common_ground", "")

    # Add balanced perspectives to findings
    if synthesis:
        state["findings"].append(f"[Bias-Mitigated Synthesis] {synthesis[:500]}")
    if common_ground:
        state["findings"].append(f"[Common Ground] {common_ground[:300]}")

    # Add pro/con summaries
    for role, label in [("pro", "Pro"), ("con", "Con"), ("neutral", "Neutral")]:
        text = results.get(role, "")
        if text:
            state["findings"].append(f"[{label} Perspective] {text[:300]}")

    # Store triangulation metadata
    state["findings"].append(f"[Bias Assessment] Score: {bias_score}/10")
    for cred in arbiter.get("credibility", []):
        if isinstance(cred, dict):
            state["findings"].append(f"[Credibility] {cred.get('perspective','')}: {cred.get('assessment','')[:150]}")

    print(f"  Bias score: {bias_score}/10")
    print(f"  Synthesis: {len(synthesis)} chars")
    state["status"] = f"Triangulation complete (bias={bias_score}/10)"
    return state


def reset_triangulator() -> None:
    """Reset rate limiter (for testing)."""
    global _last_triangulation, _tri_count
    with _tri_lock:
        _last_triangulation[0] = 0.0
        _tri_count[0] = 0


def disable_triangulator() -> None:
    """Disable Triangulator for the current run (sets counter at max)."""
    global _tri_count
    with _tri_lock:
        _tri_count[0] = MAX_TRI_PER_RUN
