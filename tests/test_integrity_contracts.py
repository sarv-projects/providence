"""Offline integration contracts for the evidence/job/API boundaries."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from src.evidence import verify_claims
from src.engine.jobs import JobRegistry


class EvidenceContracts(unittest.TestCase):
    def _state(self, claim: dict) -> dict:
        return {
            "run_corpus": [{
                "url": "https://source.test/article",
                "text": "The study enrolled 42 participants in 2025.",
            }],
            "fetched_sources": {
                "https://source.test/article": {"status": "fetched"},
            },
            "search_results": [{
                "url": "https://not-fetched.test/article",
                "raw_content": "The study enrolled 42 participants in 2025.",
            }],
            "claims": [claim],
        }

    def test_verifies_verbatim_span_and_offsets(self) -> None:
        result = verify_claims(self._state({
            "text": "The study enrolled 42 participants in 2025.",
            "atoms": ["The study enrolled 42 participants in 2025."],
            "evidence": [{
                "url": "https://source.test/article",
                "quote": "The study enrolled 42 participants in 2025.",
            }],
        }))
        row = result["claims"][0]
        self.assertEqual(row["status"], "supported")
        self.assertEqual(result["spans"][0]["start"], 0)
        self.assertEqual(result["spans"][0]["end"], 43)

    def test_lexical_overlap_without_quote_is_not_support(self) -> None:
        result = verify_claims(self._state({
            "text": "The study enrolled 42 participants in 2025.",
            "evidence_ids": ["https://source.test/article"],
        }))
        self.assertEqual(result["claims"][0]["status"], "uncertain")
        self.assertEqual(result["spans"], [])

    def test_search_result_is_not_an_evidence_document(self) -> None:
        state = self._state({
            "text": "The study enrolled 42 participants in 2025.",
            "evidence": [{
                "url": "https://not-fetched.test/article",
                "quote": "The study enrolled 42 participants in 2025.",
            }],
        })
        self.assertEqual(verify_claims(state)["claims"][0]["status"], "uncertain")


class JobContracts(unittest.TestCase):
    def test_cancel_is_terminal_and_idempotently_rejected(self) -> None:
        jobs = JobRegistry()
        job = jobs.create("test")
        self.assertTrue(jobs.cancel(job.job_id))
        self.assertFalse(jobs.cancel(job.job_id))
        self.assertTrue(jobs.is_cancelled(job.job_id))
        self.assertEqual(jobs.get(job.job_id).to_dict()["status"], "aborted")


class ApiContracts(unittest.TestCase):
    def test_missing_job_is_http_404_and_model_budget_fields_are_typed(self) -> None:
        from src.web import ResearchRequest, app

        request = ResearchRequest(
            query="offline contract",
            model="opencode_free/test",
            max_cost_usd=1.25,
            max_iterations=2,
            max_tokens=512,
        )
        self.assertEqual(request.max_iterations, 2)

        async def call() -> int:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/jobs/missing-job/cancel")
                return response.status_code

        self.assertEqual(asyncio.run(call()), 404)

    def test_research_request_forwards_model_and_budgets(self) -> None:
        from src.web import ResearchRequest, research

        async def call() -> dict:
            with patch("src.web.run_research", return_value={"job_id": "job_contract"}) as mocked:
                await research(ResearchRequest(
                    query="offline contract", background=True,
                    model="provider/model", max_cost_usd=0.25,
                    max_iterations=1, max_tokens=256,
                ))
                return mocked.call_args.kwargs

        forwarded = asyncio.run(call())
        self.assertEqual(forwarded["model"], "provider/model")
        self.assertEqual(forwarded["max_cost_usd"], 0.25)
        self.assertEqual(forwarded["max_iterations_override"], 1)
        self.assertEqual(forwarded["max_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
