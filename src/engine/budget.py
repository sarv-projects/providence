"""Runtime budget enforcement for research runs."""


from __future__ import annotations

import logging

import time
from typing import Any


def check_budgets(state: dict) -> tuple[bool, str]:
    """Return (ok, reason). If not ok, research should stop / force complete."""
    budgets = state.get("budgets") or {}
    if not budgets:
        return True, ""

    # Iterations — enforced here too so non-graph callers (temporal
    # activities, evals) cannot loop unbounded on stale flags.
    iteration = int(state.get("iteration") or 0)
    _max_iter = state.get("max_iterations")
    _max_iter = int(_max_iter) if _max_iter is not None else 0
    if _max_iter > 0 and iteration >= _max_iter:
        return False, f"Iteration budget exceeded ({iteration} >= {_max_iter})"

    # Time
    started = float(budgets.get("started_at") or 0)
    max_time = int(budgets.get("max_time_s") or 0)
    if started and max_time > 0:
        elapsed = time.time() - started
        if elapsed > max_time:
            return False, f"Time budget exceeded ({elapsed:.0f}s > {max_time}s)"

    # Cost
    spent = float(budgets.get("spent_usd") or 0)
    max_cost = float(budgets.get("max_cost_usd") or 0)
    if max_cost > 0 and spent >= max_cost:
        return False, f"Cost budget exceeded (${spent:.4f} >= ${max_cost:.2f})"

    # Tool calls
    tool_calls = int(budgets.get("tool_calls") or 0)
    max_tools = int(budgets.get("max_tool_calls") or 0)
    if max_tools > 0 and tool_calls >= max_tools:
        return False, f"Tool-call budget exceeded ({tool_calls} >= {max_tools})"

    # Tokens — previously configured but never enforced anywhere.
    tokens_used = int(budgets.get("tokens_used") or 0)
    max_tokens = int(budgets.get("max_tokens") or 0)
    if max_tokens > 0 and tokens_used >= max_tokens:
        return False, f"Token budget exceeded ({tokens_used} >= {max_tokens})"

    return True, ""


def record_tool_calls(state: dict, n: int = 1, kind: str = "") -> None:
    budgets = state.setdefault("budgets", {})
    budgets["tool_calls"] = int(budgets.get("tool_calls") or 0) + n
    if kind:
        by_kind = budgets.setdefault("tool_calls_by_kind", {})
        by_kind[kind] = int(by_kind.get(kind) or 0) + n


def budget_status_line(state: dict) -> str:
    """One-line live budget visibility for agent prompts (BATS pattern).

    Agents condition their planning on remaining resources instead of blindly
    exhausting them — matching accuracy with ~40% fewer search calls.
    """
    budgets = state.get("budgets") or {}
    if not budgets:
        return ""
    parts: list[str] = []
    calls = int(budgets.get("tool_calls") or 0)
    max_calls = int(budgets.get("max_tool_calls") or 0)
    if max_calls:
        parts.append(f"tool calls {calls}/{max_calls}")
    by_kind = budgets.get("tool_calls_by_kind") or {}
    search_n = int(by_kind.get("search") or 0)
    extract_n = int(by_kind.get("extract") or 0)
    if search_n or extract_n:
        parts.append(f"(search {search_n}, extract {extract_n})")
    started = float(budgets.get("started_at") or 0)
    max_time = int(budgets.get("max_time_s") or 0)
    if started and max_time:
        elapsed = int(time.time() - started)
        parts.append(f"time {elapsed}/{max_time}s")
    spent = float(budgets.get("spent_usd") or 0)
    max_cost = float(budgets.get("max_cost_usd") or 0)
    if max_cost:
        parts.append(f"cost ${spent:.3f}/${max_cost:.2f}")
    if not parts:
        return ""
    return "LIVE BUDGET: " + " · ".join(parts)


def sync_cost_from_metrics(state: dict) -> None:
    """Best-effort: pull estimated spend from gateway metrics into state budgets.

    Fallback path only. When a per-run cost sink is wired (graph runs — see
    ``src.llm.set_run_cost_sink``), ``budgets["spent_usd"]`` is already exact
    per-run accounting and the global-metrics baseline method must NOT
    overwrite it; the metric-derived estimate is kept separately instead.
    """
    try:
        from src.gateway.metrics import DEFAULT_METRICS
        snap = DEFAULT_METRICS.snapshot()
        total = 0.0
        for _prov, models in (snap.get("per_provider_model") or {}).items():
            for _model, stats in (models or {}).items():
                total += float(stats.get("cost_usd") or 0)
        budgets = state.setdefault("budgets", {})
        base = float(budgets.get("_cost_baseline") or 0)
        if "_cost_baseline" not in budgets:
            budgets["_cost_baseline"] = total
            if "tokens_used" not in budgets:
                budgets["spent_usd"] = 0.0
        elif "tokens_used" not in budgets:
            # No per-run sink → legacy global-metrics attribution
            budgets["spent_usd"] = max(0.0, total - base)
        else:
            # Per-run sink is the source of truth; keep the estimate aside
            budgets["spent_usd_metrics_est"] = max(0.0, total - base)
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)


def force_complete(state: dict, reason: str) -> dict:
    state["needs_more_research"] = False
    # Clear loop-driving flags or the graph takes another gather/adversary
    # round after the budget is already exhausted.
    state["socratic_reopen"] = False
    state["replan"] = False
    state["status"] = f"Budget stop: {reason}"
    state["gaps"] = list(state.get("gaps") or []) + [f"Budget: {reason}"]
    return state
