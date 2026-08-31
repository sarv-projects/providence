"""
Evaluation framework (EvalOps) for Providence — deep research engine with verified evidence.

Provides component-level and system-level evaluators running against real tool, RAG,
planner, and compilation pipelines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Any

from src.tools.registry import get_registry
from src.rag.chat_memory import ChatMemory
from src.rag.chunk import chunk_text
from src.rag.store import VectorStore
from src.engine.agents.planner import planner
from src.engine.agents.compiler import compiler
from src.state import ResearchState, initial_state


@dataclass
class EvalResult:
    """Result of a single evaluation test."""
    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class EvalSuite:
    """Collection of evaluation results."""
    name: str
    results: List[EvalResult] = field(default_factory=list)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def total_count(self) -> int:
        return len(self.results)
    
    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)


class ComponentEvaluator:
    """Evaluates individual system components against operational benchmarks."""
    
    def __init__(self):
        self.suites: List[EvalSuite] = []
    
    def run_all_component_suites(self) -> List[EvalSuite]:
        """Run all component evaluation suites."""
        self.suites = [
            EvalSuite(name="Tool Selection", results=eval_tool_selection()),
            EvalSuite(name="Plan Coherence", results=eval_plan_coherence()),
            EvalSuite(name="Memory Recall", results=eval_memory_recall()),
            EvalSuite(name="RAG IR (smoke-only)", results=eval_rag_ir()),
            EvalSuite(name="Citation Grounding", results=eval_citation_grounding()),
            # Tier-2 #17: publishable, deterministic suites
            EvalSuite(name="DR3 Sandbox (deterministic)", results=eval_dr3_sandbox()),
            EvalSuite(name="DRNOISE (adversarial)", results=eval_drnoise_adversarial()),
            EvalSuite(name="FACT Citations", results=eval_fact_citations()),
        ]
        return self.suites


class SystemEvaluator:
    """Evaluates full system end-to-end performance."""
    
    def __init__(self):
        self.suites: List[EvalSuite] = []
    
    def run_system_suites(self) -> List[EvalSuite]:
        """Run all system evaluation suites."""
        self.suites = [
            EvalSuite(name="Task Completion", results=eval_task_completion()),
            EvalSuite(name="Trajectory Analysis", results=eval_trajectory()),
            EvalSuite(name="Resource Efficiency", results=eval_efficiency()),
            EvalSuite(name="Research Quality", results=eval_research_quality()),
        ]
        return self.suites


# ── Component Evaluator Implementations ───────────────────────────

def eval_tool_selection() -> List[EvalResult]:
    """Evaluate tool discovery and capability matching accuracy."""
    start = time.time()
    registry = get_registry()
    tools = registry.list_all()
    
    web_tools = registry.list_by_capability("web_search")
    extract_tools = registry.list_by_capability("extract")

    # Actual capability ratio — previously floored at 0.5, inflating the score.
    score = (len(web_tools) + len(extract_tools)) / max(1, len(tools) * 2)
    score = min(1.0, score)
    passed = len(tools) >= 2 and len(web_tools) >= 1
    
    return [
        EvalResult(
            name="tool_selection_accuracy",
            passed=passed,
            score=score,
            details={
                "total_tools": len(tools),
                "web_search_tools": [t.name for t in web_tools],
                "extract_tools": [t.name for t in extract_tools],
            },
            duration_seconds=round(time.time() - start, 3)
        )
    ]


def eval_plan_coherence() -> List[EvalResult]:
    """Evaluate plan decomposition quality and structural coherence."""
    start = time.time()
    test_state = initial_state("Impact of Quantum Computing on Cryptography")
    planned = planner(test_state)
    
    plan = planned.get("plan", {})
    outline = planned.get("outline", [])
    queries = planned.get("search_queries", [])
    
    has_subtopics = len(plan.get("subtopics", [])) >= 2
    has_outline = len(outline) >= 2
    has_queries = len(queries) >= 1
    
    checks = [has_subtopics, has_outline, has_queries]
    score = sum(1 for c in checks if c) / len(checks)
    passed = score >= 0.66
    
    return [
        EvalResult(
            name="plan_coherence",
            passed=passed,
            score=round(score, 2),
            details={
                "subtopics_count": len(plan.get("subtopics", [])),
                "outline_sections": len(outline),
                "planned_queries": len(queries),
            },
            duration_seconds=round(time.time() - start, 3)
        )
    ]


def eval_memory_recall() -> List[EvalResult]:
    """Evaluate multi-turn conversation memory recall precision."""
    start = time.time()
    memory = ChatMemory(session_id="eval_test_session")
    memory.clear()
    
    memory.add("user", "My favorite programming language is Python.")
    memory.add("assistant", "Python is great for AI and research.")
    memory.add("user", "What is my favorite language?")
    
    context = memory.build_context("System Prompt")
    recall_found = any("Python" in m.get("content", "") for m in context)
    score = 1.0 if recall_found else 0.0
    
    return [
        EvalResult(
            name="memory_recall",
            passed=recall_found,
            score=score,
            details={"messages_stored": len(memory), "recall_target_found": recall_found},
            duration_seconds=round(time.time() - start, 3)
        )
    ]


def eval_rag_ir() -> List[EvalResult]:
    """SMOKE TEST ONLY — not a retrieval-quality measure.

    This merely checks that chunking + store + query do not crash and return
    something; ``found_relevant`` is trivially true (``len(chunks) > 0``).
    Real retrieval quality (Recall@k / MRR against expected doc IDs) lives in
    the DR3 sandbox suite below. Kept as a startup sanity check only.
    """
    start = time.time()
    sample_text = (
        "Transformer models use self-attention mechanisms to process sequential data in parallel. "
        "Attention Is All You Need was published in 2017 by Vaswani et al. "
        "LanceDB is an open-source vector database for AI applications."
    )
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=10)

    store = VectorStore(backend="fts")
    store.upsert(chunks)

    query = "Transformer self attention"
    retrieved = store.query(text=query, k=3)

    found_relevant = len(retrieved) > 0 or len(chunks) > 0
    score = 1.0 if found_relevant else 0.0

    return [
        EvalResult(
            name="rag_ir_smoke",
            passed=found_relevant,
            score=score,
            details={
                "smoke_only": True,
                "note": "not a quality measure — see DR3 sandbox suite for Recall@k/MRR",
                "chunks_indexed": len(chunks),
                "chunks_retrieved": len(retrieved),
            },
            duration_seconds=round(time.time() - start, 3)
        )
    ]


def eval_citation_grounding() -> List[EvalResult]:
    """Evaluate citation grounding: claim text must appear in source quote/evidence."""
    start = time.time()
    from src.rag.factoid import validate_quote

    source_page = (
        "Quantum computers use qubits as their fundamental unit of information. "
        "Superposition allows parallel states to be represented simultaneously."
    )
    claims = [
        {
            "text": "Quantum computers use qubits",
            "evidence_ids": ["https://example.com/quantum"],
            "source_quote": "Quantum computers use qubits as their fundamental unit",
            "confidence": "high",
        },
        {
            "text": "Superposition allows parallel states",
            "evidence_ids": ["https://example.com/physics"],
            "source_quote": "Superposition allows parallel states to be represented",
            "confidence": "high",
        },
        {
            "text": "Hallucinated claim about unicorn CPUs",
            "evidence_ids": [],
            "source_quote": "unicorns power the chip",
            "confidence": "low",
        },
    ]

    grounded = 0
    for c in claims:
        has_evidence = bool(c.get("evidence_ids"))
        quote_ok = validate_quote(c.get("source_quote", ""), source_page, threshold=0.7)
        if has_evidence and quote_ok:
            grounded += 1

    score = grounded / len(claims)
    # Expect 2/3 grounded (hallucination correctly fails)
    passed = grounded == 2 and score >= 0.6

    return [
        EvalResult(
            name="citation_grounding",
            passed=passed,
            score=round(score, 2),
            details={
                "total_claims": len(claims),
                "grounded_claims": grounded,
                "expected_grounded": 2,
            },
            duration_seconds=round(time.time() - start, 3),
        )
    ]


# ── System Evaluator Implementations ───────────────────────────────

def eval_task_completion() -> List[EvalResult]:
    """Evaluate end-to-end task completion pipeline.

    A run counts as complete ONLY if the compiler actually shipped the report.
    Previously any >100-char report string passed — including the
    "# Research Report (BLOCKED)" text the compiler emits when the ship gate
    FAILS, which let gate-failing runs score 100%.
    """
    start = time.time()
    state = initial_state("Overview of Artificial Intelligence in Healthcare")
    state["findings"] = ["AI improves diagnostic accuracy.", "Machine learning accelerates drug discovery."]
    # Realistic fixture: pages the run ACTUALLY fetched (fake example.com URLs
    # are banned by the ship gate — a gate-failing fixture must fail this eval).
    fetched_url = "https://arxiv.org/abs/2401.00001"
    state["extracted_pages"] = [{
        "url": fetched_url,
        "title": "AI in Healthcare",
        "content": (
            "AI is transforming healthcare diagnostics and discovery. "
            "Machine learning models improve diagnostic accuracy across clinical settings."
        ),
    }]
    state["retrieved_chunks"] = [{
        "url": fetched_url,
        "title": "AI in Healthcare",
        "text": "AI is transforming healthcare diagnostics and discovery.",
        "id": "c1",
        "score": 0.95,
    }]
    state["claims"] = [{
        "text": "AI is transforming healthcare diagnostics and discovery",
        "evidence_ids": [fetched_url],
    }]
    state["sections"] = [
        {"title": "Overview", "content": "AI is transforming healthcare diagnostics and discovery.", "sources": [fetched_url]},
        {"title": "Sources", "content": "[1] [AI in Healthcare](%s)" % fetched_url, "sources": [fetched_url]},
    ]
    compiled = compiler(state)
    report = compiled.get("report", "")

    # 1. Ship gate must not block the compiled state
    from src.engine.agents.compiler import _validate_ship_gate
    _, gate_issues = _validate_ship_gate(compiled)
    gate_passed = not gate_issues
    # 2. Report must exist and must not be the BLOCKED stub
    has_report = len(report) > 100
    not_blocked = not report.lstrip().lower().startswith("# research report (blocked")

    passed = has_report and not_blocked and gate_passed
    score = 1.0 if passed else (0.3 if has_report and not_blocked else 0.0)

    return [
        EvalResult(
            name="task_completion",
            passed=passed,
            score=score,
            details={
                "report_chars": len(report),
                "ship_gate_passed": gate_passed,
                "ship_gate_issues": gate_issues[:5],
                "blocked_report": not not_blocked,
            },
            duration_seconds=round(time.time() - start, 3)
        )
    ]


def eval_trajectory() -> List[EvalResult]:
    """Evaluate the graph's actual routing decisions on sample states.

    Previously compared two constants (2 <= 3) — trivially true. Now exercises
    the real conditional-edge functions from the live graph.
    """
    start = time.time()
    from src.graph import should_continue_research, after_adjudicator

    # needs_more_research → loop again
    s_loop = initial_state("t")
    s_loop["needs_more_research"] = True
    s_loop["abort_synthesis"] = False
    # sufficient evidence → adversary
    s_done = initial_state("t")
    s_done["needs_more_research"] = False
    s_done["abort_synthesis"] = False
    # abort → compile_abort
    s_abort = initial_state("t")
    s_abort["abort_synthesis"] = True
    # socratic reopen → re-gather
    s_soc = initial_state("t")
    s_soc["socratic_reopen"] = True
    s_soc["abort_synthesis"] = False
    # no reopen → triangulate
    s_tri = initial_state("t")
    s_tri["socratic_reopen"] = False
    s_tri["abort_synthesis"] = False

    checks = {
        "loop_again": should_continue_research(s_loop) == "research_again",
        "proceed_to_adversary": should_continue_research(s_done) == "adversary",
        "abort_routes_to_compile": should_continue_research(s_abort) == "compile_abort",
        "socratic_hop_reopens": after_adjudicator(s_soc) == "socratic_again",
        "clean_claim_route_triangulates": after_adjudicator(s_tri) == "triangulate",
    }
    passed_count = sum(1 for ok in checks.values() if ok)
    passed = passed_count == len(checks)

    return [
        EvalResult(
            name="trajectory",
            passed=passed,
            score=round(passed_count / len(checks), 2),
            details={"route_checks": checks},
            duration_seconds=round(time.time() - start, 3)
        )
    ]


def eval_efficiency() -> List[EvalResult]:
    """Measure real work: wall-clock of an index+query cycle and per-chunk
    token cost. Previously ``passed=True`` was hardcoded."""
    start = time.time()
    sample_corpus = [
        "Chunk 1 content for testing retrieval efficiency.",
        "Chunk 2 content for testing retrieval efficiency.",
        "Chunk 3 content for testing retrieval efficiency.",
    ]
    total_tokens_est = sum(len(c.split()) * 1.3 for c in sample_corpus)

    # Do the actual work being measured
    store = VectorStore(backend="fts")
    chunks = chunk_text(" ".join(sample_corpus), chunk_size=100, chunk_overlap=10)
    store.upsert(chunks)
    hits = store.query(text="retrieval efficiency", k=2)
    elapsed = time.time() - start

    # Pass criteria: retrieval actually returned something within a sane time
    retrieved_ok = len(hits) > 0
    time_ok = elapsed < 10.0
    tokens_ok = total_tokens_est < 10000
    passed = retrieved_ok and time_ok and tokens_ok
    score = sum(1.0 for ok in (retrieved_ok, time_ok, tokens_ok) if ok) / 3.0

    return [
        EvalResult(
            name="efficiency",
            passed=passed,
            score=round(score, 2),
            details={"estimated_tokens": int(total_tokens_est)},
            duration_seconds=round(time.time() - start, 3)
        )
    ]


def eval_research_quality() -> List[EvalResult]:
    """Evaluate compiler ship-gate on a minimal assembled state (live path)."""
    start = time.time()
    state = initial_state("Artificial Intelligence quality eval")
    state["findings"] = ["AI expands rapidly.", "ML is widely adopted."]
    state["claims"] = [
        {
            "text": "AI expands rapidly.",
            "evidence_ids": ["https://example.com/ai"],
            "confidence": "medium",
        }
    ]
    state["evidence_map"] = {"https://example.com/ai": ["AI expands rapidly."]}
    state["sections"] = [
        {
            "title": "Overview",
            "content": "Artificial Intelligence is expanding rapidly across industries.",
            "sources": ["https://example.com/ai"],
        },
        {
            "title": "Findings",
            "content": "Machine learning is widely adopted in production systems.",
            "sources": ["https://example.com/ai"],
        },
        {
            "title": "Sources",
            "content": "[1] [AI Research](https://example.com/ai)",
            "sources": ["https://example.com/ai"],
        },
    ]
    compiled = compiler(state)
    report = compiled.get("report", "")
    has_title = report.startswith("# ")
    has_sections = "## " in report
    has_sources = "Sources" in report
    long_enough = len(report) >= 100

    checks = [has_title, has_sections, has_sources, long_enough]
    score = sum(1 for c in checks if c) / len(checks)

    return [
        EvalResult(
            name="research_quality",
            passed=score >= 0.75,
            score=round(score, 2),
            details={
                "has_title": has_title,
                "has_sections": has_sections,
                "has_sources": has_sources,
                "report_chars": len(report),
            },
            duration_seconds=round(time.time() - start, 3),
        )
    ]


# ── Tier-2 #17: publishable eval suites (DR³-Eval / DRNOISE / FACT) ────────


def eval_dr3_sandbox() -> List[EvalResult]:
    """DR³-Eval style: deterministic static sandbox — no network, no LLM.

    A localized corpus of sandbox pages (supportive + one distractor) is
    ingested into a throwaway temp-dir FTS store (never the shared data/fts.db)
    and queried. Measures Information Recall (gold facts recovered) and
    Citation Coverage (every retrieved URL belongs to the sandbox — no
    out-of-corpus leakage).
    """
    import tempfile

    start = time.time()
    sandbox = [
        {
            "url": "https://sandbox.dev/transformer",
            "title": "Transformer Architecture",
            "content": (
                "Transformer models use self-attention mechanisms to process sequential data in parallel. "
                "Attention Is All You Need was published in 2017 by Vaswani et al. "
                "Self-attention computes weighted representations of all positions simultaneously."
            ),
        },
        {
            "url": "https://sandbox.dev/attention",
            "title": "Attention Mechanisms",
            "content": (
                "Self-attention allows each token to attend to every other token in the sequence. "
                "The attention head computes queries, keys, and values to weight relevance. "
                "This removes the sequential bottleneck of recurrent networks."
            ),
        },
        {
            "url": "https://sandbox.dev/distractor",
            "title": "Unrelated Cooking",
            "content": (
                "Sourdough bread requires a live starter culture and careful hydration. "
                "Baking times vary by oven temperature and ambient humidity."
            ),
        },
    ]
    # Gold facts are exact (lowercased) substrings of the sandbox corpus, so
    # recall measures genuine retrieval, not string-luck.
    gold_facts = [
        "self-attention mechanisms to process sequential data in parallel",
        "attention is all you need was published in 2017",
        "queries, keys, and values to weight relevance",
    ]

    # Deterministic sandbox: throwaway FTS-only store in a temp dir. The shared
    # data/fts.db is polluted by real runs and must never feed an eval.
    from src.rag.backends.fts import FTSStore
    tmp_dir = tempfile.mkdtemp(prefix="dr3_sandbox_")
    store = FTSStore(db_path=os.path.join(tmp_dir, "sandbox.db"))
    chunks = []
    for p in sandbox:
        chunks.extend(chunk_text(
            p["content"], chunk_size=200, chunk_overlap=20,
            metadata={"url": p["url"], "title": p["title"], "source_type": "sandbox"},
        ))
    store.upsert(chunks)

    retrieved = store.query(text="transformer self-attention parallel", k=5)
    retrieved_text = " ".join(r.get("text", "") for r in retrieved).lower()
    retrieved_urls = {r.get("url") for r in retrieved}
    sandbox_urls = {p["url"] for p in sandbox}

    recall = sum(1 for f in gold_facts if f in retrieved_text) / len(gold_facts)
    coverage = (
        len(retrieved_urls & sandbox_urls) / max(1, len(retrieved_urls))
        if retrieved_urls else 0.0
    )
    passed = recall >= 0.5 and coverage >= 0.9

    return [
        EvalResult(
            name="dr3_information_recall",
            passed=passed,
            score=round((recall + coverage) / 2, 2),
            details={
                "recall": round(recall, 2),
                "citation_coverage": round(coverage, 2),
                "retrieved": len(retrieved),
                "chunks_indexed": len(chunks),
            },
            duration_seconds=round(time.time() - start, 3),
        )
    ]


def eval_drnoise_adversarial() -> List[EvalResult]:
    """DRNOISE-style: verification-inertia test (no network, no LLM).

    A gold answer is supported by TWO indirect record-chain documents; a
    plausible-but-wrong document states a conflicting value directly. The
    adjudicator must flag the conflict (reconcile the chain) instead of
    accepting the answer-like document. Also asserts the evidence-graph
    ship-gate still blocks a zero-support graph.
    """
    start = time.time()
    from src.engine.agents.adversary import claim_adjudicator
    from src.engine.agents.compiler import _validate_ship_gate

    state = initial_state("reactor output 2025")
    state["mode"] = "quick"  # skips the live-web atomic verify inside adjudicator
    state["claims"] = [
        {
            "text": "The reactor reached 100 MW output on June 1 2025",
            "evidence_ids": ["https://drnoise.example/report-a", "https://drnoise.example/report-b"],
        },
        {
            "text": "The reactor reached 300 MW output on June 1 2025",
            "evidence_ids": ["https://drnoise.example/plausible-wrong"],
        },
    ]
    # Indirect record chain (two corroborating docs) + the noisy doc
    state["run_corpus"] = [
        {"url": "https://drnoise.example/report-a", "title": "Report A",
         "text": "The official June 1 record states reactor output at 100 MW during commissioning."},
        {"url": "https://drnoise.example/report-b", "title": "Report B",
         "text": "According to the June 1 commissioning record, the reactor delivered 100 MW."},
        {"url": "https://drnoise.example/plausible-wrong", "title": "Claim Sheet",
         "text": "The reactor reached 300 MW output on June 1 2025 per the summary sheet."},
    ]
    state["evidence_map"] = {}
    state["retrieved_chunks"] = []
    state["extracted_pages"] = []
    state["search_results"] = []
    state["devil_advocate_done"] = True
    state["socratic_done"] = False
    state["socratic_hops"] = 0
    state["gaps"] = []
    state["research_debt"] = []

    out = claim_adjudicator(state)
    debt = " ".join(out.get("research_debt") or []).lower()
    contested = out.get("contested_claims") or []
    conflict_flagged = "conflict" in debt
    # The single-source wrong claim should be contested (or at least the run
    # must not be able to ship treating both values as equally supported)
    wrong_contested = any(
        "300 mw" in (c.get("text") or "").lower()
        for c in contested
    )

    # Zero-support graph must still fail the ship gate
    gate_state = initial_state("gate")
    gate_state["sections"] = [{"title": "Body", "content": "x" * 200, "sources": []}]
    gate_state["claims"] = [{"text": "c"}]
    gate_state["evidence_graph"] = [{"relation": "unsupported"} for _ in range(6)]
    gate_state["abort_synthesis"] = False
    _, gate_issues = _validate_ship_gate(gate_state)
    gate_blocks = any("Evidence graph" in i for i in gate_issues)

    score = (0.6 if conflict_flagged else 0.0) + (0.4 if gate_blocks else 0.0)
    return [
        EvalResult(
            name="drnoise_conflict_reconciliation",
            passed=conflict_flagged and gate_blocks,
            score=round(score, 2),
            details={
                "conflict_flagged": conflict_flagged,
                "wrong_claim_contested": wrong_contested,
                "ship_gate_blocks_zero_support": gate_blocks,
                "debt_notes": len(out.get("research_debt") or []),
            },
            duration_seconds=round(time.time() - start, 3),
        )
    ]


def eval_fact_citations() -> List[EvalResult]:
    """FACT-style citation metric: existence + support, no network, no LLM.

    A fabricated report cites [1] a real this-run URL (whose chunk supports the
    claim phrase) and [2] an invented URL that was never fetched. The metric
    must give credit only where the citation exists in the run's source set and
    the chunk text supports the claim.
    """
    start = time.time()
    import re

    run_sources = {"https://fact.example/real"}
    chunk_text_by_url = {
        "https://fact.example/real": (
            "Retrieval augmented generation reduces hallucination by grounding generation in retrieved documents."
        )
    }
    # Fake report: [1] real, [2] fabricated URL never fetched this run
    report = (
        "RAG reduces hallucination by grounding generation [1]. "
        "Unicorn CPUs triple inference speed [2]."
    )
    sources_list = [
        "https://fact.example/real",
        "https://fact.example/fabricated",
    ]

    # Claim phrases per citation (simplified: first 6 content words of the sentence)
    sentences = re.split(r"(?<=[.!?])\s+", report)
    citation_checks = []
    for sent in sentences:
        nums = re.findall(r"\[(\d+)\]", sent)
        if not nums:
            continue
        words = re.findall(r"[a-zA-Z]{4,}", sent.lower())
        phrase = " ".join(words[:6])
        for n in nums:
            idx = int(n) - 1
            if 0 <= idx < len(sources_list):
                url = sources_list[idx]
                exists = url in run_sources
                support = (
                    exists
                    and sum(1 for w in phrase.split() if w in chunk_text_by_url.get(url, "").lower())
                    >= 2
                )
                citation_checks.append({"url": url, "exists": exists, "support": support})

    total = max(len(citation_checks), 1)
    exists_ok = sum(1 for c in citation_checks if c["exists"])
    support_ok = sum(1 for c in citation_checks if c["support"])
    existence_acc = exists_ok / total
    support_acc = support_ok / total
    # [1] real: exists+support ✓ ; [2] fabricated: exists ✗ → both metrics 0.5
    passed = exists_ok == 1 and support_ok == 1

    return [
        EvalResult(
            name="fact_citation_accuracy",
            passed=passed,
            score=round((existence_acc + support_acc) / 2, 2),
            details={
                "existence_accuracy": round(existence_acc, 2),
                "support_accuracy": round(support_acc, 2),
                "citations_checked": len(citation_checks),
                "fabricated_caught": exists_ok == 1,
            },
            duration_seconds=round(time.time() - start, 3),
        )
    ]


def create_component_evaluator() -> ComponentEvaluator:
    return ComponentEvaluator()


def create_system_evaluator() -> SystemEvaluator:
    return SystemEvaluator()
