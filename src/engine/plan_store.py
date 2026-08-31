"""Editable research plan store (Google Deep Research–style plan review).

L1: plan is generated and research continues immediately (optional edit via
     plan_first=True).
L2: plan approval is required before gather/analyze.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional


class ResearchPlan:
    def __init__(
        self,
        query: str,
        mode: str = "standard",
        autonomy: str = "L1",
    ) -> None:
        self.plan_id = f"plan_{uuid.uuid4().hex[:12]}"
        self.query = query
        self.mode = mode
        self.autonomy = autonomy
        self.status = "draft"  # draft|awaiting_approval|approved|rejected|running|complete
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.plan: dict = {}
        self.outline: list[dict] = []
        self.search_queries: list[str] = []
        self.clarifying_questions: list[str] = []
        self.clarifications: dict[str, str] = {}  # question -> answer
        self.needs_clarification: bool = False
        self.job_id: str = ""
        self.error: str = ""

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "mode": self.mode,
            "autonomy": self.autonomy,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "plan": self.plan,
            "outline": self.outline,
            "search_queries": self.search_queries,
            "clarifying_questions": self.clarifying_questions,
            "clarifications": self.clarifications,
            "needs_clarification": self.needs_clarification,
            "job_id": self.job_id,
            "error": self.error,
        }


class PlanStore:
    # Bounded store: plans accumulate forever otherwise (L1 runs create one
    # plan per research call).
    MAX_PLANS = 500
    MAX_AGE_S = 7 * 24 * 3600  # drop terminal plans older than 7 days

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._plans: dict[str, ResearchPlan] = {}

    def _evict_locked(self) -> None:
        now = time.time()
        terminal = ("complete", "rejected", "error")
        expired = [
            pid for pid, p in self._plans.items()
            if p.status in terminal and (now - p.updated_at) > self.MAX_AGE_S
        ]
        for pid in expired:
            del self._plans[pid]
        if len(self._plans) > self.MAX_PLANS:
            # Evict oldest terminal plans first, then oldest overall
            def _sort_key(pair):
                p = pair[1]
                terminal_rank = 0 if p.status in terminal else 1
                return (terminal_rank, p.updated_at)
            ordered = sorted(self._plans.items(), key=_sort_key)
            for pid, _ in ordered[: len(self._plans) - self.MAX_PLANS]:
                del self._plans[pid]

    def create(self, query: str, mode: str = "standard", autonomy: str = "L1") -> ResearchPlan:
        p = ResearchPlan(query, mode=mode, autonomy=autonomy)
        with self._lock:
            self._plans[p.plan_id] = p
            self._evict_locked()
        return p

    def get(self, plan_id: str) -> Optional[ResearchPlan]:
        with self._lock:
            return self._plans.get(plan_id)

    def update(self, plan_id: str, **fields: Any) -> Optional[ResearchPlan]:
        with self._lock:
            p = self._plans.get(plan_id)
            if not p:
                return None
            for k, v in fields.items():
                if hasattr(p, k) and v is not None:
                    setattr(p, k, v)
            p.updated_at = time.time()
            return p

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            items = sorted(self._plans.values(), key=lambda x: x.created_at, reverse=True)
            return [p.to_dict() for p in items[:limit]]


_PLANS = PlanStore()


def get_plans() -> PlanStore:
    return _PLANS
