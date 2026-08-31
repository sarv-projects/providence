"""
Multi-provider LLM wrapper routed through the resilient gateway.

The interface is unchanged (``call_llm`` / ``call_llm_strong``) so the existing
LangGraph nodes keep working. Under the hood every call goes through the
BYOK LLM gateway with:

- Multi-provider failover (Zen free default → paid keys last)
- Circuit breakers per model endpoint
- Retries with exponential backoff + full jitter
- Per-(tenant, model) rate limiting and concurrency caps
- Cost/token accounting + metrics for the dashboard

Provider priority:
  1. Workhorse (fast/strong): OpenCode Zen free (no key)
  2. Thinker: Gemini Flash only (GEMINI_API_KEY)
  3. Paid Groq/OpenAI only after Zen workhorse routes fail
"""

import logging
import os
import threading

from dotenv import load_dotenv

from src.gateway import build_gateway_from_env
from src.gateway.router import AllRoutesFailed, QuotaExceeded

load_dotenv()

# Fast model for most tasks, strong model for synthesis.
# Tiering (Tier-2 #18, u14app thinking/task split):
#   thinker  — reasoning nodes (scout, plan refine, contradiction, strategy, adjudication)
#   task     — extraction/labeling nodes (cheap, high-throughput)
#   strong   — synthesis
#   fast     — default fallback
DEFAULT_MODEL = "fast"
STRONG_MODEL = "strong"
TASK_MODEL = "task"
THINKER_MODEL = "thinker"

# task tier is an alias for the fast tier at the gateway level (no dedicated
# routes needed) — it exists so node call-sites express intent explicitly.
_TIER_ALIASES = {"task": "fast"}

_gateway = None

# Per-run cost/token accounting (#14): a thread-local "sink" receives
# (prompt_tokens, completion_tokens, estimated_cost_usd) for every call made
# on the current thread. run_research installs one that accumulates directly
# into that run's state["budgets"], giving true per-run attribution instead of
# the race-prone global-metrics baseline approach.
_run_ctx = threading.local()


def set_run_cost_sink(sink) -> None:
    """Install a per-run cost sink for the current thread.

    ``sink(prompt_tokens, completion_tokens, estimated_cost_usd)`` is called
    after every successful LLM call made from this thread.
    """
    _run_ctx.cost_sink = sink


def clear_run_cost_sink() -> None:
    """Remove the per-run cost sink for the current thread."""
    _run_ctx.cost_sink = None


def set_run_request_context(model: str | None = None, max_tokens: int | None = None) -> None:
    """Apply per-run model/token settings to graph LLM calls."""
    _run_ctx.model_override = model or None
    _run_ctx.max_tokens_override = max_tokens if max_tokens and max_tokens > 0 else None


def clear_run_request_context() -> None:
    _run_ctx.model_override = None
    _run_ctx.max_tokens_override = None


def _notify_run_cost_sink(res) -> None:
    """Best-effort notification of the thread's cost sink (never raises)."""
    sink = getattr(_run_ctx, "cost_sink", None)
    if sink is None:
        return
    try:
        from .gateway.providers import PRICING
        price = PRICING.get(getattr(res, "model", "") or "", PRICING.get("*default"))
        cost = (
            (res.prompt_tokens / 1_000_000) * price[0]
            + (res.completion_tokens / 1_000_000) * price[1]
        )
        sink(res.prompt_tokens, res.completion_tokens, cost)
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)


def _get_gateway():
    global _gateway
    if _gateway is None:
        _gateway = build_gateway_from_env()
    return _gateway


def reset_gateway() -> None:
    """Reset the gateway singleton (useful for testing)."""
    global _gateway
    _gateway = None


def gateway_info() -> dict:
    """Return info about currently available providers and models."""
    gw = _get_gateway()
    info = {
        "fast_routes": len(gw.get_routes("fast")),
        "strong_routes": len(gw.get_routes("strong")),
        "thinker_routes": len(gw.get_routes("thinker")),
        "routes": [],
    }
    for tier in ("fast", "strong", "thinker"):
        for route in gw.get_routes(tier):
            info["routes"].append({
                "tier": tier,
                "provider": route.provider.name,
                "model": route.model,
                "has_key": bool(getattr(route.provider, "api_keys", [])),
            })
    return info


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,  # kept for API compatibility; gateway does its own retries
    max_tokens: int | None = None,
) -> str:
    gw = _get_gateway()
    override = getattr(_run_ctx, "model_override", None)
    requested_model = override if override and model in ("fast", "strong", "thinker", "task") else model
    tier = requested_model if requested_model in ("fast", "strong", "thinker", "task") else requested_model
    tier = _TIER_ALIASES.get(tier, tier)
    if not gw.get_routes(tier):
        # If the tier has no routes, fall back to "fast".
        if tier != "fast" and gw.get_routes("fast"):
            tier = "fast"
    effective_max_tokens = max_tokens or getattr(_run_ctx, "max_tokens_override", None)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        result = gw.complete(messages, model=tier, max_tokens=effective_max_tokens)
        _notify_run_cost_sink(result)
    except QuotaExceeded as e:
        raise RuntimeError(f"Quota / rate limit exceeded: {e}")
    except AllRoutesFailed as e:
        raise RuntimeError(f"All LLM providers failed: {e}")
    return result.text


def call_llm_strong(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
) -> str:
    """Use the stronger model tier for synthesis."""
    return call_llm(system_prompt, user_prompt, model=STRONG_MODEL, max_tokens=max_tokens)


def call_llm_stream(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int | None = None,
):
    """Streaming LLM call — yields text chunks via generator.

    Uses the gateway's streaming transport. Falls back to non-streaming
    if the provider doesn't support streaming.
    """
    gw = _get_gateway()
    override = getattr(_run_ctx, "model_override", None)
    requested_model = override if override and model in ("fast", "strong", "thinker", "task") else model
    tier = requested_model
    # Apply the same tier alias as call_llm (previously streaming silently
    # dropped the task→fast alias).
    tier = _TIER_ALIASES.get(tier, tier)
    if not gw.get_routes(tier) and tier != "fast" and gw.get_routes("fast"):
        tier = "fast"
    effective_max_tokens = max_tokens or getattr(_run_ctx, "max_tokens_override", None)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Per-run accounting: streaming previously bypassed the run cost sink.
    # Accumulate the stream and notify the sink with estimated tokens/cost
    # once the stream completes (tokens are unknown until then).
    prompt_tokens = sum(len(m.get("content") or "") for m in messages) // 4
    completion_chars = 0
    try:
        for chunk in gw.complete_stream(messages, model=tier, max_tokens=effective_max_tokens):
            completion_chars += len(chunk)
            yield chunk
        # Notify sink with an estimate (exact per-call cost is recorded in
        # gateway metrics; sink keeps the run's budget view consistent).
        sink = getattr(_run_ctx, "cost_sink", None)
        if sink is not None:
            completion_tokens = max(1, completion_chars // 4)
            try:
                from .gateway.providers import PRICING
                price = PRICING.get("*default")
                est_cost = (
                    (prompt_tokens / 1_000_000) * price[0]
                    + (completion_tokens / 1_000_000) * price[1]
                )
                sink(prompt_tokens, completion_tokens, est_cost)
            except Exception:
                logging.getLogger(__name__).debug("cost sink notify failed", exc_info=True)
    except (QuotaExceeded, AllRoutesFailed) as e:
        raise RuntimeError(f"LLM streaming failed: {e}")
