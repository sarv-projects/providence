"""
BYOK (Bring Your Own Key) key management for the LLM gateway.

Two distinct kinds of keys, like LiteLLM/Portkey:

1. **Virtual Keys** (`xa_...`) — issued to end users/tenants. They never see a
   raw provider key. We store only a SHA-256 *hash* of the virtual key (so a DB
   leak does not expose usable keys). Each virtual key maps to a tenant and
   carries a budget (USD) and daily token limit. These are what the app/agent
   authenticates with and what cost is attributed to for billing.

2. **Provider Keys** — the real upstream secrets (Groq, OpenAI, ...). They live
   in a pool per provider and are *rotated* on a TTL. Rotation follows a
   **grace period** pattern (like LiteLLM's ``_KEY_ROTATION_GRACE_PERIOD``): the
   old key stays "rotating" for a grace window so in-flight requests finish,
   while a new key is fetched from a ``rotate_callback`` (e.g. a Vault / AWS SM /
   KMS adapter in production). Provider secrets can optionally be encrypted at
   rest with ``GATEWAY_MASTER_KEY`` (AES-GCM) via the helper functions.

Everything here is stdlib-only and does not touch the network unless you supply
a ``rotate_callback``.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


def hash_virtual_key(virtual_key: str) -> str:
    """SHA-256 hash used to store a virtual key without keeping plaintext."""
    return hashlib.sha256(virtual_key.encode("utf-8")).hexdigest()


def generate_virtual_key(prefix: str = "xa_") -> str:
    return prefix + secrets.token_urlsafe(24)


def generate_provider_key(prefix: str = "pk_") -> str:
    return prefix + secrets.token_urlsafe(24)


# ---- Optional AES-GCM encryption of provider secrets at rest ----------
def _master_key() -> Optional[bytes]:
    raw = os.getenv("GATEWAY_MASTER_KEY")
    if not raw:
        return None
    # derive 32-byte key from an arbitrary-length env secret
    return hashlib.sha256(raw.encode("utf-8")).digest()


class KeyManager:
    """In-process BYOK store. Swap internals for a real DB/secrets backend in prod."""

    def __init__(self, rotating_grace_s: float = 3600.0) -> None:
        self._lock = threading.RLock()
        self._rotating_grace_s = rotating_grace_s
        self._virtual: Dict[str, str] = {}          # hash -> tenant_id
        self._tenants: Dict[str, Tenant] = {}
        self._provider_keys: Dict[str, List[ProviderKey]] = {}
        self._rotate_callbacks: Dict[str, Callable[[str, str], str]] = {}

    # ---- virtual keys ---------------------------------------------------
    def create_tenant(self, tenant_id: str, budget_usd: float = 100.0, tokens_per_day: int = 1_000_000) -> Tenant:
        with self._lock:
            t = Tenant(tenant_id=tenant_id, budget_usd=budget_usd, tokens_per_day=tokens_per_day)
            self._tenants[tenant_id] = t
            return t

    def mint_virtual_key(self, tenant_id: str, budget_usd: Optional[float] = None) -> str:
        """Create a virtual key for a tenant and return the raw key (shown once)."""
        with self._lock:
            if tenant_id not in self._tenants:
                self.create_tenant(tenant_id)
            if budget_usd is not None:
                self._tenants[tenant_id].budget_usd = budget_usd
            vk = generate_virtual_key()
            self._virtual[hash_virtual_key(vk)] = tenant_id
            return vk

    def resolve_virtual_key(self, virtual_key: str) -> Optional[Tenant]:
        """Validate a presented virtual key; return the tenant or None."""
        with self._lock:
            tenant_id = self._virtual.get(hash_virtual_key(virtual_key))
            if tenant_id is None:
                return None
            t = self._tenants.get(tenant_id)
            if t is None or not t.enabled:
                return None
            return t

    def authorize(
        self,
        virtual_key: str,
        estimated_cost_usd: float = 0.0,
        estimated_tokens: int = 0,
    ) -> Optional[Tenant]:
        """BYOK auth + budget pre-check. Returns tenant if allowed, else None.

        Enforces BOTH the USD budget and the daily token limit (with automatic
        day rollover — ``tokens_today`` resets when the UTC day changes).
        Pending reservations from in-flight requests are counted too.
        """
        t = self.resolve_virtual_key(virtual_key)
        if t is None:
            return None
        with self._lock:
            # Daily reset: UTC day rollover
            today = time.strftime("%Y-%m-%d", time.gmtime())
            if t.last_reset_day != today:
                t.last_reset_day = today
                t.tokens_today = 0
            if t.spent_usd + t.pending_cost_usd + max(0.0, estimated_cost_usd) > t.budget_usd:
                return None
            if t.tokens_today + t.pending_tokens + max(0, estimated_tokens) > t.tokens_per_day:
                return None  # daily token quota exhausted
        return t

    # ---- atomic quota reservations --------------------------------------
    def reserve_for(
        self,
        tenant: Tenant,
        estimated_cost_usd: float = 0.0,
        estimated_tokens: int = 0,
    ) -> Optional["Reservation"]:
        """Atomically reserve quota BEFORE a call (check + hold in one lock).

        Prevents the TOCTOU race where N concurrent requests each pass
        ``authorize()`` before any of them charges. On call success, pass the
        reservation to ``charge()`` to swap the hold for actual usage; on
        failure, ``release_reservation()`` returns the hold.
        """
        with self._lock:
            today = time.strftime("%Y-%m-%d", time.gmtime())
            if tenant.last_reset_day != today:
                tenant.last_reset_day = today
                tenant.tokens_today = 0
            if tenant.spent_usd + tenant.pending_cost_usd + max(0.0, estimated_cost_usd) > tenant.budget_usd:
                return None
            if tenant.tokens_today + tenant.pending_tokens + max(0, estimated_tokens) > tenant.tokens_per_day:
                return None
            tenant.pending_cost_usd += max(0.0, estimated_cost_usd)
            tenant.pending_tokens += max(0, estimated_tokens)
            return Reservation(
                tenant_id=tenant.tenant_id,
                est_cost_usd=max(0.0, estimated_cost_usd),
                est_tokens=max(0, estimated_tokens),
            )

    def release_reservation(self, reservation: "Reservation") -> None:
        """Return a reservation's hold (call failed / was not charged)."""
        with self._lock:
            t = self._tenants.get(reservation.tenant_id)
            if t is None:
                return
            t.pending_cost_usd = max(0.0, t.pending_cost_usd - reservation.est_cost_usd)
            t.pending_tokens = max(0, t.pending_tokens - reservation.est_tokens)

    def charge(
        self,
        tenant: Tenant,
        cost_usd: float,
        tokens: int,
        reservation: Optional["Reservation"] = None,
    ) -> None:
        with self._lock:
            tenant.spent_usd += cost_usd
            tenant.tokens_today += tokens
            if reservation is not None:
                # Swap the hold for actual usage
                tenant.pending_cost_usd = max(0.0, tenant.pending_cost_usd - reservation.est_cost_usd)
                tenant.pending_tokens = max(0, tenant.pending_tokens - reservation.est_tokens)

    def tenant(self, tenant_id: str) -> Optional[Tenant]:
        with self._lock:
            return self._tenants.get(tenant_id)

    # ---- provider keys & rotation --------------------------------------
    def register_provider_key(
        self,
        provider: str,
        key: str = "",
        ttl_s: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._provider_keys.setdefault(provider, []).append(
                ProviderKey(provider=provider, key=key, ttl_s=ttl_s)
            )

    def set_rotate_callback(self, provider: str, cb: Callable[[str, str], str]) -> None:
        """``cb(provider, old_key_hint) -> new_key`` — hook to Vault/SM/KMS in prod."""
        with self._lock:
            self._rotate_callbacks[provider] = cb

    def _rotate(self, k: ProviderKey) -> None:
        cb = self._rotate_callbacks.get(k.provider)
        new_key = cb(k.provider, k.key) if cb else ""
        if new_key:
            with self._lock:
                k.key = new_key
            k.created_at = time.time()
            k.failures = 0
            k.status = "active"
            k.grace_until = None
        else:
            k.status = "disabled"   # needs operator attention / external rotation
            k.grace_until = None

    def usable_provider_keys(self, provider: str, now: Optional[float] = None) -> List[ProviderKey]:
        """Return currently usable keys, rotating any that have passed TTL+grace.

        Concurrency: expiry/status classification happens under the lock; the
        external ``rotate_callback`` is invoked OUTSIDE the lock (it may do
        network I/O or take a different lock — calling it under our lock risks
        deadlock). New key state is then applied atomically under a fresh lock
        acquisition, guarded by the key's ``created_at`` so a concurrent
        rotation cannot be clobbered.
        """
        now = now or time.time()
        to_rotate: List[ProviderKey] = []
        with self._lock:
            for k in list(self._provider_keys.get(provider, [])):
                if k.status in ("disabled",):  # expired & no replacement
                    continue
                if k.ttl_s:
                    deadline = k.created_at + k.ttl_s
                    if now > deadline:
                        if now > deadline + self._rotating_grace_s:
                            k.status = "disabled"
                            continue
                        k.status = "rotating"
                    else:
                        k.status = "active"
            out = [k for k in self._provider_keys.get(provider, []) if k.status != "disabled"]
            # Capture callbacks under the lock; invoke them outside it.
            to_rotate = [
                k for k in out
                if k.status == "rotating" and k.provider in self._rotate_callbacks
            ]

        # Perform external rotation without holding the lock (deadlock-safe).
        results: List[tuple[ProviderKey, str, float]] = []
        for k in to_rotate:
            old_created = k.created_at
            try:
                new_key = self._rotate_callbacks[k.provider](k.provider, k.key)
            except Exception:
                new_key = ""
            results.append((k, new_key or "", old_created))

        # Apply results atomically; skip if the key rotated concurrently.
        with self._lock:
            for k, new_key, old_created in results:
                if k.created_at != old_created:
                    continue  # another caller already rotated it
                if new_key:
                    k.key = new_key
                    k.created_at = time.time()
                    k.failures = 0
                    k.status = "active"
                    k.grace_until = None
                elif now > k.created_at + (k.ttl_s or 0) + self._rotating_grace_s:
                    k.status = "disabled"   # needs operator attention / external rotation
                    k.grace_until = None
                # else: keep "rotating" — grace window still open, retry later

            out = [
                k for k in self._provider_keys.get(provider, [])
                if k.status in ("active", "rotating")
            ]
        return out


def encrypt_secret(plaintext: str) -> Optional[str]:
    """Return ``enc:v1:<b64(iv|ciphertext|tag)>`` or None if no master key set."""
    key = _master_key()
    if key is None or not plaintext:
        return None
    iv = os.urandom(12)
    cipher = __import__("cryptography.hazmat.primitives.ciphers.aead", fromlist=["AESGCM"]).AESGCM(key)
    ct = cipher.encrypt(iv, plaintext.encode("utf-8"), None)
    blob = b"enc:v1:" + base64.b64encode(iv + ct)
    return blob.decode("utf-8")


def decrypt_secret(value: str) -> str:
    if value.startswith("enc:v1:"):
        key = _master_key()
        if key is None:
            raise RuntimeError("GATEWAY_MASTER_KEY not set; cannot decrypt provider secret")
        b64 = value[len("enc:v1:"):]
        raw = base64.b64decode(b64)
        iv, ct = raw[:12], raw[12:]
        cipher = __import__("cryptography.hazmat.primitives.ciphers.aead", fromlist=["AESGCM"]).AESGCM(key)
        return cipher.decrypt(iv, ct, None).decode("utf-8")
    return value


@dataclass
class Tenant:
    tenant_id: str
    budget_usd: float = 100.0
    tokens_per_day: int = 1_000_000
    spent_usd: float = 0.0
    tokens_today: int = 0
    enabled: bool = True
    last_reset_day: str = ""  # UTC "YYYY-MM-DD" of last tokens_today reset
    pending_cost_usd: float = 0.0  # held by in-flight reservations
    pending_tokens: int = 0


@dataclass
class Reservation:
    """A quota hold created by ``KeyManager.reserve_for``."""
    tenant_id: str
    est_cost_usd: float = 0.0
    est_tokens: int = 0


@dataclass
class ProviderKey:
    provider: str
    key: str = ""
    status: str = "active"          # active | rotating | disabled
    ttl_s: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    grace_until: Optional[float] = None
    failures: int = 0
    last_good: Optional[float] = None