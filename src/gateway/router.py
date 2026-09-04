"""
The LLM Gateway — orchestrator that ties together every resiliency layer.

Flow for each request:
  BYOK auth -> budget check -> rate limit -> concurrency slot -> circuit check
  -> retry-with-jitter loop over an ordered route chain (provider/model/key
  failover) -> budget charge -> metrics.

This is the in-process form (the LangGraph agent calls it directly). The exact
same config can be lifted into a standalone HTTP/sidecar gateway later.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .circuit import CircuitRegistry
from .keys import KeyManager, Reservation, Tenant
from .metrics import DEFAULT_METRICS, MetricsRegistry
from .providers import (
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderResult,
    ProviderTimeoutError,
)
from .ratelimit import RateLimiter


class QuotaExceeded(Exception):
    """Budget exhausted or rate-limited for this tenant/request."""


class AllRoutesFailed(Exception):
    """Every route in the chain failed after retries/failover."""


@dataclass
class Route:
    provider: object          # anything with .complete(messages, model, ...) and .name
    model: str
    tier: str = "default"
    priority: int = 0
    name: str = ""


@dataclass
class GatewayStats:
    attempts: int = 0
    route_order: List[str] = field(default_factory=list)


class Gateway:
    """Resilient, multi-provider, BYOK-aware LLM gateway."""

    def __init__(
        self,
        key_manager: Optional[KeyManager] = None,
        metrics: Optional[MetricsRegistry] = None,
        ratelimiter: Optional[RateLimiter] = None,
        circuits: Optional[CircuitRegistry] = None,
        max_attempts: int = 3,
        retry_base_s: float = 0.5,
        retry_cap_s: float = 8.0,
        default_tenant: str = "default",
    ) -> None:
        self.km = key_manager or KeyManager()
        self.metrics = metrics or DEFAULT_METRICS
        self.rl = ratelimiter or RateLimiter()
        self.circuits = circuits or CircuitRegistry(
            on_state_change=lambda route, state: self.metrics.record_circuit_state(route, state)
        )
        self.max_attempts = max(1, max_attempts)
        self.retry_base_s = retry_base_s
        self.retry_cap_s = retry_cap_s
        self.default_tenant = default_tenant
        self._routes: Dict[str, List[Route]] = {}
        self._lock = threading.RLock()

    # ---- registration ---------------------------------------------------
    def register(self, route: Route) -> None:
        with self._lock:
            self._routes.setdefault(route.tier, []).append(route)
            self._routes[route.tier].sort(key=lambda r: r.priority)

    def get_routes(self, tier: str) -> List[Route]:
        with self._lock:
            routes = self._routes.get(tier)
            if routes is not None:
                return list(routes)
            # The settings UI stores provider/model route IDs (for example
            # ``opencode_free/nemotron-...``), while the graph usually asks
            # for a tier.  Resolve an exact route ID to a single route so a
            # selected model is an actual request constraint, not decoration.
            return [
                route
                for tier_routes in self._routes.values()
                for route in tier_routes
                if route.name == tier
            ]

    def complete_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        tenant: Optional[str] = None,
        virtual_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
    ):
        """Streaming completion — yields text chunks via generator.

        Uses the first available route in the tier. Falls back through
        routes on failure, but only the first route is streamed.
        """
        routes = self.get_routes(model)
        if not routes:
            raise ValueError(f"No routes registered for model/tier: {model}")

        tenant_obj = self._resolve_tenant(virtual_key, tenant)
        last_error: Optional[Exception] = None

        # Atomic quota reservation (same as complete())
        prompt_est = sum(len(m.get("content") or "") for m in messages) // 4
        est_tokens = prompt_est + (max_tokens or 1024)
        est_cost, _ = self._estimate_request(est_tokens)
        reservation = self.km.reserve_for(tenant_obj, estimated_cost_usd=est_cost, estimated_tokens=est_tokens)
        if reservation is None:
            raise QuotaExceeded("Budget / daily token quota exhausted")

        try:
            for route in routes:
                if not self.circuits.get(route.name).allow_request():
                    last_error = ProviderConnectionError(f"circuit open for {route.name}")
                    continue

                keys = []
                if hasattr(route.provider, "api_keys") and route.provider.api_keys:
                    keys = route.provider.api_keys
                api_key = self._pick_key(route.provider, keys)

                # Retry with backoff on retriable errors — but ONLY while
                # nothing has been yielded yet (retrying mid-stream would
                # duplicate output to the caller).
                for attempt in range(1, self.max_attempts + 1):
                    yielded_any = False
                    try:
                        if not self.rl.enter_parallel():
                            raise QuotaExceeded("No concurrency slot")
                        started = time.time()
                        try:
                            if not self.rl.acquire(tenant_obj.tenant_id, route.model):
                                raise QuotaExceeded(f"Rate limit: {route.model}")

                            # Yield chunks from streaming provider
                            full_text = ""
                            for chunk in route.provider.complete_stream(
                                messages, model=route.model,
                                temperature=temperature, max_tokens=max_tokens, api_key=api_key,
                            ):
                                yielded_any = True
                                full_text += chunk
                                yield chunk

                            # Record success after full stream — same accounting
                            # path as complete(): circuit, cost/token charge,
                            # metrics, reservation commit.
                            self.circuits.get(route.name).on_success()
                            latency_s = time.time() - started
                            # Tokens are unknown until the provider reports
                            # usage; estimate ~4 chars per token.
                            prompt_tokens = prompt_est
                            completion_tokens = max(1, len(full_text) // 4)
                            cost = self._estimate_cost(
                                route.provider.name, route.model,
                                ProviderResult(
                                    text=full_text,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    latency_s=latency_s,
                                ),
                            )
                            self.km.charge(
                                tenant_obj, cost, prompt_tokens + completion_tokens,
                                reservation=reservation,
                            )
                            self.metrics.record_success(
                                route.provider.name, route.model, latency_s,
                                prompt_tokens, completion_tokens, cost, tenant_obj.tenant_id,
                            )
                            self.metrics.log_event(
                                "success", route=route.name, streamed=True,
                                chars=len(full_text), estimated_tokens=completion_tokens,
                            )
                            return
                        finally:
                            self.rl.exit_parallel()

                    except (ProviderHTTPError, ProviderTimeoutError, ProviderConnectionError) as e:
                        last_error = e
                        # Streaming previously bypassed circuit/metrics —
                        # record failures so breakers trip and dashboards
                        # stay accurate.
                        self.metrics.record_error(
                            route.provider.name, route.model, type(e).__name__,
                            str(e)[:120], tenant_obj.tenant_id,
                        )
                        retriable = not (isinstance(e, ProviderHTTPError) and not e.retriable)
                        self.circuits.get(route.name).on_failure(retriable)
                        if not retriable or yielded_any or attempt >= self.max_attempts:
                            break  # next route (or final failure)
                        self._backoff(attempt)
                    except QuotaExceeded:
                        raise  # concurrency/rate-limit denials are not retryable here

            raise AllRoutesFailed(
                f"All {len(routes)} route(s) failed; last error: {last_error}"
            ) from last_error
        except GeneratorExit:
            # Caller abandoned the stream (break/GC): GeneratorExit is NOT an
            # Exception subclass, so it bypasses `except Exception` — release
            # the quota hold explicitly before propagating.
            self.km.release_reservation(reservation)
            raise
        except Exception:
            # Failure path: no charge happened — return the quota hold.
            self.km.release_reservation(reservation)
            raise

    # ---- public entry point ---------------------------------------------
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        tenant: Optional[str] = None,
        virtual_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
    ) -> ProviderResult:
        """
        Route a completion through the resilient chain.

        ``model`` selects a *tier* (fallback chain) of routes. When
        ``virtual_key`` is given it is used for BYOK auth + budget enforcement;
        otherwise ``tenant`` (or the default tenant) is used in in-process mode.
        """
        routes = self.get_routes(model)
        if not routes:
            raise ValueError(f"No routes registered for model/tier: {model}")

        tenant = self._resolve_tenant(virtual_key, tenant)
        stats = GatewayStats()
        last_error: Optional[Exception] = None

        # Atomic quota reservation: check + hold in ONE lock so N concurrent
        # requests cannot each pass the check before any of them charges.
        prompt_est = sum(len(m.get("content") or "") for m in messages) // 4
        est_tokens = prompt_est + (max_tokens or 1024)
        est_cost, _ = self._estimate_request(est_tokens)
        reservation = self.km.reserve_for(tenant, estimated_cost_usd=est_cost, estimated_tokens=est_tokens)
        if reservation is None:
            raise QuotaExceeded("Budget / daily token quota exhausted")

        try:
            for route in routes:
                if not self.circuits.get(route.name).allow_request():
                    self.metrics.log_event("skip", reason="circuit_open", route=route.name)
                    last_error = ProviderConnectionError(f"circuit open for {route.name}")
                    continue

                try:
                    return self._call_route(
                        route, messages, tenant, max_tokens, temperature, stats,
                        reservation=reservation,
                    )
                except (ProviderHTTPError, ProviderTimeoutError, ProviderConnectionError) as e:
                    last_error = e
                    # A non-retriable 4xx means this route (or its key/model) is
                    # unusable for this request — try the next route once.
                    if isinstance(e, ProviderHTTPError) and not e.retriable:
                        continue

            self.metrics.log_event("all_failed", routes=[r.name for r in routes])
            raise AllRoutesFailed(
                f"All {len(routes)} route(s) failed; last error: {last_error}"
            ) from last_error
        except Exception:
            # Failure path: no charge happened — return the quota hold.
            self.km.release_reservation(reservation)
            raise

    def _resolve_tenant(self, virtual_key: Optional[str], tenant: Optional[str]) -> Tenant:
        if virtual_key:
            t = self.km.authorize(virtual_key)
            if t is None:
                raise QuotaExceeded("Invalid or over-budget virtual key")
            return t
        t = self.km.tenant(tenant or self.default_tenant) or self.km.create_tenant(
            tenant or self.default_tenant
        )
        return t

    @staticmethod
    def _estimate_request(tokens_est: int) -> tuple[float, int]:
        """Conservative (cost_usd, tokens) estimate used to reserve quota."""
        from .providers import PRICING
        price = PRICING.get("*default")
        return (tokens_est / 1_000_000) * price[1], tokens_est

    def _call_route(
        self,
        route: Route,
        messages: List[Dict[str, str]],
        tenant: Tenant,
        max_tokens: Optional[int],
        temperature: float,
        stats: GatewayStats,
        reservation: Optional["Reservation"] = None,
    ) -> ProviderResult:
        model = route.model
        # concurrency + rate limit
        if not self.rl.enter_parallel():
            self.metrics.record_ratelimit(False, tenant.tenant_id, model)
            raise QuotaExceeded("No concurrency slot available")
        admitted = self.rl.acquire(tenant.tenant_id, model)
        if not admitted:
            self.metrics.record_ratelimit(False, tenant.tenant_id, model)
            self.rl.exit_parallel()
            raise QuotaExceeded(f"Rate limit reached for {model}")
        self.metrics.record_ratelimit(True, tenant.tenant_id, model)
        try:
            return self._do_retry(route, messages, tenant, max_tokens, temperature, stats,
                                  reservation=reservation)
        finally:
            self.rl.exit_parallel()

    def _do_retry(
        self,
        route: Route,
        messages: List[Dict[str, str]],
        tenant: Tenant,
        max_tokens: Optional[int],
        temperature: float,
        stats: GatewayStats,
        reservation: Optional["Reservation"] = None,
    ) -> ProviderResult:
        attempt = 0
        while attempt < self.max_attempts:
            attempt += 1
            stats.attempts += 1
            try:
                keys = []
                if hasattr(route.provider, "api_keys") and route.provider.api_keys:
                    keys = route.provider.api_keys
                api_key = self._pick_key(route.provider, keys)
                res: ProviderResult = route.provider.complete(
                    messages,
                    model=route.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                )
                self.circuits.get(route.name).on_success()
                cost = self._estimate_cost(route.provider.name, route.model, res)
                self.km.charge(tenant, cost, res.prompt_tokens + res.completion_tokens, reservation=reservation)
                self.metrics.record_success(
                    route.provider.name, route.model, res.latency_s,
                    res.prompt_tokens, res.completion_tokens, cost, tenant.tenant_id,
                )
                self.metrics.log_event(
                    "success", route=route.name, attempt=attempt,
                    latency_s=round(res.latency_s, 3),
                )
                return res
            except ProviderHTTPError as e:
                self.metrics.record_error(
                    route.provider.name, route.model, f"http_{e.status}",
                    e.args[0] if e.args else "", tenant.tenant_id,
                )
                self.circuits.get(route.name).on_failure(e.retriable)
                if not e.retriable or attempt >= self.max_attempts:
                    raise
                self._backoff(attempt)
            except (ProviderTimeoutError, ProviderConnectionError) as e:
                self.metrics.record_error(
                    route.provider.name, route.model, type(e).__name__,
                    str(e)[:120], tenant.tenant_id,
                )
                self.circuits.get(route.name).on_failure(True)
                if attempt >= self.max_attempts:
                    raise
                self._backoff(attempt)
        raise ProviderConnectionError(f"unreachable: {route.name}")

    # Per-provider key rotation counters (thread-safe enough for load balancing)
    _key_rr_index: Dict[str, int] = {}
    _key_rr_lock = threading.RLock()

    def _pick_key(self, provider, keys: List[str]) -> Optional[str]:
        """Round-robin across the provider's key pool to spread rate limits."""
        if not keys:
            return None
        if len(keys) == 1:
            return keys[0]
        name = getattr(provider, "name", "default")
        with self._key_rr_lock:
            idx = self._key_rr_index.get(name, 0) % len(keys)
            self._key_rr_index[name] = idx + 1
            return keys[idx]

    @staticmethod
    def _estimate_cost(provider: str, model: str, res: ProviderResult) -> float:
        from .providers import PRICING
        price = PRICING.get(model, PRICING.get("*default"))
        in_per_m, out_per_m = price
        return (res.prompt_tokens / 1_000_000) * in_per_m + (
            res.completion_tokens / 1_000_000
        ) * out_per_m

    def _backoff(self, attempt: int) -> None:
        """Exponential backoff with full jitter: sleep = rand(0, min(cap, base*2^n))."""
        cap = min(self.retry_cap_s, self.retry_base_s * (2 ** (attempt - 1)))
        time.sleep(random.uniform(0, cap))
