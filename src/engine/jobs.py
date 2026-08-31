"""In-process async research job registry (shared state for long runs)."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional


class ResearchJob:
    def __init__(self, query: str, mode: str = "standard", autonomy: str = "L1") -> None:
        self.job_id = f"job_{uuid.uuid4().hex[:12]}"
        self.query = query
        self.mode = mode
        self.autonomy = autonomy
        self.plan_id = ""
        self.cancel_requested = False
        self.status = "queued"  # queued|running|complete|error|aborted
        self.created_at = time.time()
        self.started_at: float = 0.0
        self.finished_at: float = 0.0
        self.error = ""
        self.run_id = ""
        self.plan: dict = {}
        self.learned: list[str] = []
        self.gaps: list[str] = []
        self.next_action = ""
        self.thoughts: list[dict] = []
        self.report = ""
        self.markdown_path = ""
        self.findings_count = 0
        self.sources_count = 0
        self.pages_scanned = 0
        self.iterations = 0
        self.stage = "queued"
        self.perspectives: list[str] = []

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "query": self.query,
            "mode": self.mode,
            "autonomy": self.autonomy,
            "plan_id": self.plan_id,
            "status": self.status,
            "stage": self.stage,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round(
                (self.finished_at or time.time()) - (self.started_at or self.created_at), 1
            ),
            "error": self.error,
            "plan": self.plan,
            "learned": self.learned[-20:],
            "gaps": self.gaps[-20:],
            "next_action": self.next_action,
            "thoughts": self.thoughts[-30:],
            "findings_count": self.findings_count,
            "sources_count": self.sources_count,
            "pages_scanned": self.pages_scanned,
            "iterations": self.iterations,
            "report": (self.report or "")[:50000],
            "markdown_path": self.markdown_path,
            "perspectives": list(self.perspectives[-12:]),
            "finished": self.status in ("complete", "error", "aborted"),
            "cancel_requested": self.cancel_requested,
        }


class JobRegistry:
    # Completed jobs kept in memory (bounded to avoid unbounded growth in a
    # long-running server — each job can carry ~50k chars of report text).
    MAX_COMPLETED_JOBS = 100
    COMPLETED_TTL_S = 6 * 3600  # evict completed jobs after 6 hours

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, ResearchJob] = {}

    def _evict_locked(self, now: float) -> None:
        """Drop old completed jobs (caller must hold the lock)."""
        completed = [
            (jid, j) for jid, j in self._jobs.items()
            if j.status in ("complete", "error", "aborted")
        ]
        expired = [
            jid for jid, j in completed
            if j.finished_at and (now - j.finished_at) > self.COMPLETED_TTL_S
        ]
        for jid in expired:
            del self._jobs[jid]
        # Still over the cap → evict oldest completed first
        completed = [
            (jid, j) for jid, j in self._jobs.items()
            if j.status in ("complete", "error", "aborted")
        ]
        overflow = len(completed) - self.MAX_COMPLETED_JOBS
        if overflow > 0:
            completed.sort(key=lambda pair: pair[1].finished_at or 0)
            for jid, _ in completed[:overflow]:
                del self._jobs[jid]

    def create(self, query: str, mode: str = "standard", autonomy: str = "L1") -> ResearchJob:
        job = ResearchJob(query, mode=mode, autonomy=autonomy)
        with self._lock:
            self._jobs[job.job_id] = job
            self._evict_locked(time.time())
        return job

    def get(self, job_id: str) -> Optional[ResearchJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in jobs[:limit]]

    def update(self, job_id: str, clear: tuple = (), **fields: Any) -> None:
        """Update job fields.

        ``clear`` names fields to reset to their zero value — ``None`` values
        in ``fields`` are still ignored (previously there was NO way to clear
        a field, e.g. ``error`` after a retry).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in fields.items():
                if job.cancel_requested and k == "status" and v != "aborted":
                    continue
                if hasattr(job, k) and v is not None:
                    setattr(job, k, v)
            for k in clear:
                if hasattr(job, k):
                    default = "" if isinstance(getattr(job, k), str) else (
                        [] if isinstance(getattr(job, k), list) else
                        {} if isinstance(getattr(job, k), dict) else 0.0
                    )
                    setattr(job, k, default)

    def add_thought(self, job_id: str, kind: str, text: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.thoughts.append({"ts": time.time(), "kind": kind, "text": text[:500]})
            if kind == "learned":
                job.learned.append(text[:300])
            elif kind == "gap":
                job.gaps.append(text[:300])
            elif kind == "next":
                job.next_action = text[:300]

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation of a queued/running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in ("complete", "error", "aborted"):
                return False
            job.cancel_requested = True
            job.status = "aborted"
            job.stage = "cancelled"
            job.error = "Research cancelled by user"
            job.finished_at = time.time()
            return True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.cancel_requested)


JOBS = JobRegistry()


def get_jobs() -> JobRegistry:
    return JOBS
