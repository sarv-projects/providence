"""
Rate limiting for the LLM gateway.

Implements a token-bucket limiter keyed by ``(tenant, model)`` so each
customer/tenant and each model has its own RPM (requests/min) and TPM
(tokens/min) budget. A lightweight global concurrency semaphore caps parallel
in-flight requests (protects against overwhelming a provider, mirroring
LiteLLM's ``max_parallel_requests``).

Clients of the gateway should call :meth:`RateLimiter.acquire` before a call;
if it returns ``False`` the call is denied (429 / QuotaExceeded) and the caller
can retry after a backoff as advised.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple


class TokenBucket:
    """A token bucket with refill. Requests consume 1 token; tokens/min model."""

    def __init__(self, rate_per_min: float, burst: int) -> None:
        # tokens available accumulate at rate_per_min/60 per second, capped at burst
        self.rate_per_sec = rate_per_min / 60.0
        self.capacity = max(1, burst)
        self._tokens = float(self.capacity)
        self._ts = time.monotonic()
        self._lock = threading.Lock()

    def try_consume(self, n: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._ts
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
            self._ts = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    @property
    def tokens(self) -> float:
        return self._tokens


class RateLimiter:
    def __init__(
        self,
        default_rpm: int = 60,
        default_tpm: int = 120_000,
        max_parallel: int = 20,
    ) -> None:
        self.default_rpm = default_rpm
        self.default_tpm = default_tpm
        self._rpm: Dict[Tuple[str, str], TokenBucket] = {}
        self._tpm: Dict[Tuple[str, str], TokenBucket] = {}
        self._lock = threading.RLock()
        self._sem = threading.BoundedSemaphore(max_parallel)
        self._parallel = 0

    def _bucket_for(self, buckets: Dict, key: Tuple[str, str], rate: float, burst: int) -> TokenBucket:
        with self._lock:
            b = buckets.get(key)
            if b is None:
                b = TokenBucket(rate, burst)
                buckets[key] = b
            return b

    def rpm_bucket(self, tenant: str, model: str) -> TokenBucket:
        return self._bucket_for(self._rpm, (tenant, model), self.default_rpm, self.default_rpm)

    def tpm_bucket(self, tenant: str, model: str) -> TokenBucket:
        return self._bucket_for(self._tpm, (tenant, model), self.default_tpm, self.default_tpm)

    def acquire(self, tenant: str, model: str, estimated_tokens: int = 0) -> bool:
        """Check request- and token-rate limits. Returns True if admitted."""
        # Check TPM first so a token-deny doesn't burn an RPM token.
        if estimated_tokens:
            if not self.tpm_bucket(tenant, model).try_consume(estimated_tokens):
                return False
        if not self.rpm_bucket(tenant, model).try_consume(1):
            return False
        return True

    def enter_parallel(self) -> bool:
        """Block until a parallel slot is free. Returns True when acquired."""
        if self._sem.acquire(timeout=60.0):
            with self._lock:
                self._parallel += 1
            return True
        return False

    def exit_parallel(self) -> None:
        with self._lock:
            self._parallel = max(0, self._parallel - 1)
        self._sem.release()

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._parallel

    def status(self) -> Dict[str, int]:
        with self._lock:
            return {"in_flight": self._parallel, "max_parallel": self._sem._initial_value}
