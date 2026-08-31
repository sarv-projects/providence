"""
Progress Tracker — shared, thread-safe state for real-time research progress.

Deep-research style thinking panel:
  learned[]  — what we know so far
  gaps[]     — missing info
  next_action — what the agent will do next
  thoughts[] — chronological thought stream
  job_id     — async job linkage
"""

from __future__ import annotations

import logging

import threading
import time
from typing import Any, Optional


class ResearchProgress:
    """Thread-safe tracker for a research run's progress."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Archived snapshots of previous runs, keyed by job_id. Concurrent /
        # overlapping runs previously clobbered each other's entire progress
        # state because start() reset the single global object.
        self._by_job: dict[str, dict] = {}
        self._reset()

    def _reset(self) -> None:
        with self._lock:
            self.run_id: str = ""
            self.job_id: str = ""
            self.query: str = ""
            self.mode: str = ""
            self.stage: str = "idle"
            self.iteration: int = 0
            self.max_iterations: int = 6
            self.findings_count: int = 0
            self.factoids_count: int = 0
            self.sources_count: int = 0
            self.sections: list[dict] = []
            self.current_section: str = ""
            self.section_index: int = 0
            self.total_sections: int = 0
            self.status: str = ""
            self.elapsed_s: float = 0.0
            self.started_at: float = 0.0
            self.finished: bool = False
            self.error: str = ""
            self.report: str = ""
            self.markdown_path: str = ""
            # Thinking panel (Google Deep Research style)
            self.learned: list[str] = []
            self.gaps: list[str] = []
            self.next_action: str = ""
            self.thoughts: list[dict] = []
            self.plan: dict = {}
            self.off_topic: bool = False
            self.pages_scanned: int = 0

    def start(
        self,
        query: str,
        run_id: str = "",
        max_iterations: int = 6,
        job_id: str = "",
        mode: str = "",
    ) -> None:
        with self._lock:
            # Archive the previous run before resetting so its snapshot stays
            # queryable by job_id (polling clients of run A no longer see run
            # B wipe A's progress mid-flight).
            if self.job_id and self.started_at:
                self._by_job[self.job_id] = self.snapshot()
                if len(self._by_job) > 50:  # bound the archive
                    self._by_job.pop(next(iter(self._by_job)), None)
            self._reset()
            self.run_id = run_id
            self.job_id = job_id
            self.query = query
            self.mode = mode
            self.max_iterations = max_iterations
            self.stage = "starting"
            self.status = "Starting research..."
            self.started_at = time.time()

    def snapshot_for(self, job_id: str) -> dict:
        """Snapshot for a specific job without falling back to another run."""
        with self._lock:
            if not job_id or job_id == self.job_id:
                return self.snapshot()
            archived = self._by_job.get(job_id)
            if archived is not None:
                return dict(archived)
            return {
                "job_id": job_id,
                "status": "unknown",
                "stage": "unknown",
                "finished": True,
                "error": "Progress job not found",
            }

    def think(self, kind: str, text: str) -> None:
        """Append a thinking-panel event."""
        with self._lock:
            entry = {"ts": time.time(), "kind": kind, "text": (text or "")[:500]}
            self.thoughts.append(entry)
            if len(self.thoughts) > 100:
                self.thoughts = self.thoughts[-100:]
            if kind == "learned" and text:
                self.learned.append(text[:300])
                self.learned = self.learned[-30:]
            elif kind == "gap" and text:
                self.gaps.append(text[:300])
                self.gaps = self.gaps[-30:]
            elif kind == "next" and text:
                self.next_action = text[:400]
            # mirror to job registry
            if self.job_id:
                try:
                    from src.engine.jobs import get_jobs
                    get_jobs().add_thought(self.job_id, kind, text)
                except Exception:
                    logging.getLogger(__name__).debug("ignored error", exc_info=True)

    def update(
        self,
        stage: str = "",
        iteration: int = -1,
        findings_count: int = -1,
        factoids_count: int = -1,
        sources_count: int = -1,
        pages_scanned: int = -1,
        sections: list[dict] | None = None,
        current_section: str = "",
        section_index: int = -1,
        total_sections: int = -1,
        status: str = "",
        error: str = "",
        finished: bool | None = None,
        report: str = "",
        markdown_path: str = "",
        plan: dict | None = None,
        next_action: str = "",
        off_topic: bool | None = None,
        learned: list[str] | None = None,
        gaps: list[str] | None = None,
    ) -> None:
        with self._lock:
            if stage:
                self.stage = stage
            if iteration >= 0:
                self.iteration = iteration
            if findings_count >= 0:
                self.findings_count = findings_count
            if factoids_count >= 0:
                self.factoids_count = factoids_count
            if sources_count >= 0:
                self.sources_count = sources_count
            if pages_scanned >= 0:
                self.pages_scanned = pages_scanned
            if sections is not None:
                self.sections = sections
            if current_section:
                self.current_section = current_section
            if section_index >= 0:
                self.section_index = section_index
            if total_sections >= 0:
                self.total_sections = total_sections
            if status:
                self.status = status
            if error:
                self.error = error
            if finished is not None:
                self.finished = finished
            if report:
                self.report = report
            if markdown_path:
                self.markdown_path = markdown_path
            if plan is not None:
                self.plan = plan
            if next_action:
                self.next_action = next_action
            if off_topic is not None:
                self.off_topic = off_topic
            if learned is not None:
                self.learned = learned[-30:]
            if gaps is not None:
                self.gaps = gaps[-30:]
            self.elapsed_s = time.time() - self.started_at if self.started_at else 0
            # sync job
            if self.job_id:
                try:
                    from src.engine.jobs import get_jobs
                    jobs = get_jobs()
                    if not jobs.is_cancelled(self.job_id):
                        jobs.update(
                            self.job_id,
                            stage=self.stage,
                            status="complete" if self.finished and not self.error else (
                                "error" if self.error and self.finished else "running"
                            ),
                            findings_count=self.findings_count,
                            sources_count=self.sources_count,
                            pages_scanned=self.pages_scanned,
                            iterations=self.iteration,
                            report=self.report,
                            markdown_path=self.markdown_path,
                            next_action=self.next_action,
                            plan=self.plan,
                            error=self.error,
                        )
                except Exception:
                    logging.getLogger(__name__).debug("ignored error", exc_info=True)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "run_id": self.run_id,
                "job_id": self.job_id,
                "query": self.query,
                "mode": self.mode,
                "stage": self.stage,
                "iteration": self.iteration,
                "max_iterations": self.max_iterations,
                "findings_count": self.findings_count,
                "factoids_count": self.factoids_count,
                "sources_count": self.sources_count,
                "pages_scanned": self.pages_scanned,
                "sections": [
                    {"title": s.get("title", ""), "chars": len(s.get("content", ""))}
                    for s in self.sections
                ],
                "current_section": self.current_section,
                "section_progress": f"{self.section_index}/{self.total_sections}"
                if self.total_sections else "",
                "status": self.status,
                "elapsed_s": round(self.elapsed_s, 1),
                "finished": self.finished,
                "error": self.error,
                "report": self.report[:50000] if self.report else "",
                "markdown_path": self.markdown_path,
                # thinking panel
                "learned": list(self.learned[-15:]),
                "gaps": list(self.gaps[-15:]),
                "next_action": self.next_action,
                "thoughts": list(self.thoughts[-25:]),
                "plan": self.plan,
                "off_topic": self.off_topic,
            }


CURRENT_PROGRESS = ResearchProgress()


def get_progress() -> ResearchProgress:
    """Progress for the current context.

    Returns the thread's run-scoped instance when the thread is executing a
    research run (see ``start_run_progress``), so concurrent runs each mutate
    their own object; falls back to the shared default for CLI/tests/web
    threads that don't belong to a run.
    """
    p = getattr(_tls, "progress", None)
    return p if p is not None else CURRENT_PROGRESS


# ── Per-run isolation (registry) ─────────────────────────────────────────
_tls = threading.local()
_REGISTRY: Dict[str, ResearchProgress] = {}
_registry_lock = threading.RLock()
_REGISTRY_MAX = 50


def start_run_progress(job_id: str = "") -> ResearchProgress:
    """Bind a fresh ResearchProgress to the calling thread and register it.

    Agent nodes call ``get_progress()`` on the run's thread and get THIS
    instance — concurrent runs no longer share (and clobber) one live object.
    Polling endpoints fetch it by job_id via ``get_progress_by_job``.
    """
    p = ResearchProgress()
    _tls.progress = p
    if job_id:
        with _registry_lock:
            if len(_REGISTRY) >= _REGISTRY_MAX:
                _REGISTRY.pop(next(iter(_REGISTRY)), None)
            _REGISTRY[job_id] = p
    return p


def end_run_progress() -> None:
    """Unbind the thread's run-scoped progress (call in a finally block)."""
    _tls.progress = None


def get_progress_by_job(job_id: str) -> Optional[ResearchProgress]:
    """Look up a run's isolated progress object by job_id."""
    with _registry_lock:
        return _REGISTRY.get(job_id)
