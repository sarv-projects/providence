from typing import TypedDict


class SearchResult(TypedDict):
    title: str
    url: str
    content: str
    raw_content: str
    score: float


class ExtractedPage(TypedDict):
    url: str
    content: str


class Section(TypedDict):
    title: str
    content: str
    sources: list[str]   # evidence_ids for this section


class ResearchState(TypedDict):
    # --- Input ---
    query: str

    # --- Planner output ---
    plan: dict            # {topic, subtopics, outline: [{title, queries}], source_types}
    search_queries: list[str]
    plan_approved: bool
    plan_id: str
    clarifications: dict
    clarifying_questions: list[str]
    scout: dict  # pre-plan thinker web + Gemini scout

    # --- Researcher output ---
    search_results: list[SearchResult]
    extracted_pages: list[ExtractedPage]
    clean_content: list[str]
    claims: list[dict]    # [{text, atoms, evidence: [{url, quote}], confidence}]

    # --- RAG ---
    run_id: str
    chunks_ingested: int
    retrieved_chunks: list[dict]
    # Full-text corpus of pages actually fetched this run (researcher_gather
    # accumulates; compiler/adjudicator/evidence read). Declared here so the
    # LangGraph schema carries it between nodes instead of dropping it.
    run_corpus: list[dict]

    # --- Factoid Pipeline ---
    factoids: list[dict]     # structured {type, value, confidence, source_quote, source_url, entities, topics}
    factoid_stats: dict      # {raw_tokens, factoid_tokens, num_factoids, reduction_pct, types}

    # --- Retriever Guard ---
    guard_stats: dict        # {total, passed, blocked, avg_score, domains}

    # --- Critic output ---
    findings: list[str]
    gaps: list[str]
    needs_more_research: bool
    replan: bool
    off_topic: bool
    abort_synthesis: bool

    # --- Adversary / CoVe / Research debt (Ultra steals) ---
    devil_advocate_done: bool
    socratic_hops: int
    socratic_reopen: bool
    socratic_done: bool
    adjudicated_claims: list
    contested_claims: list
    synthetic_claims: list
    research_debt: list
    confidence_note: str

    # --- RE-TRAC structured memory (cross-iteration research state) ---
    research_memory: dict  # {answers: [...], consulted_sources: [url, ...], open_hypotheses: [...]}

    # --- Structured missing-facts (r1-reasoning-rag) ---
    missing_facts: list[dict]  # [{fact, sub_topic, suggested_queries: [...]}]

    # --- Evidence ledger (fetched-source ledger for the compiler ship-gate) ---
    fetched_sources: dict  # {canonical_url: {url,title,status,content_hash,chars,fetched_at}}
    verified_spans: list  # [{claim, atom, quote, url, start, end}]

    # --- Evidence graph (Argus) ---
    evidence_graph: list[dict]  # [{claim_id, claim, evidence_url, relation: support|contradiction|unsupported, score}]

    # --- Atomic verification (DeepVerifier-style, Tier-2 #14) ---
    atomic_verified: list[dict]  # [{claim, atoms, dra_label: SUPPORTED|REFUTED|AMBIGUOUS|UNVERIFIABLE, score, evidence_urls}]

    # --- Task-id ledger (langgraph-deep-research) ---
    task_ledger: list[dict]  # [{finding, task_id, section_title, iteration}]

    # --- Search-mode routing (WebSwarm) ---
    search_modes: dict  # {query: atom|deep|wide|entity_collect|web_structure}

    # --- Newswire pass (GDELT/NewsData once-per-run cache) ---
    news_hits: list  # tier-1 newswire hits already fetched this run (only set on success, so failures retry)

    # --- Query-type classification (Anthropic) ---
    query_type: str  # depth_first | breadth_first | straightforward

    # --- Fruitless-action gate (Jina node-DeepResearch) ---
    fruitless: dict  # {search_disabled, visit_disabled, search_streak, visit_streak}

    # --- Synthesizer output ---
    outline: list[dict]         # [{title, order}]
    sections: list[Section]     # progressively written sections
    evidence_map: dict[str, list[str]]  # evidence_id → [url, title]

    # --- Final output ---
    report: str
    markdown_path: str

    # --- Loop control ---
    iteration: int
    max_iterations: int
    mode: str
    status: str
    error: str
    job_id: str

    # --- Optional runtime controls (modes / autonomy / budgets) ---
    autonomy: str
    quality: dict
    budgets: dict
    mode_flags: dict


def initial_state(query: str, max_iterations: int = 6) -> ResearchState:
    import time
    import uuid
    # Iteration caps are enforced by the live graph / budgets; the legacy
    # src/nodes.py path keeps its own MAX_ITERATIONS and is no longer mutated
    # here (see src/graph.py for the active loop control).
    return {
        "query": query,
        "plan": {},
        "search_queries": [],
        "plan_approved": False,
        "plan_id": "",
        "clarifications": {},
        "clarifying_questions": [],
        "scout": {},
        "search_results": [],
        "extracted_pages": [],
        "clean_content": [],
        "claims": [],
        "run_id": uuid.uuid4().hex[:12],
        "chunks_ingested": 0,
        "retrieved_chunks": [],
        "run_corpus": [],
        "factoids": [],
        "factoid_stats": {},
        "guard_stats": {},
        "findings": [],
        "gaps": [],
        "needs_more_research": False,
        "replan": False,
        "off_topic": False,
        "abort_synthesis": False,
        "devil_advocate_done": False,
        "socratic_hops": 0,
        "socratic_reopen": False,
        "socratic_done": False,
        "adjudicated_claims": [],
        "contested_claims": [],
        "synthetic_claims": [],
        "research_debt": [],
        "confidence_note": "",
        "research_memory": {
            "answers": [],
            "consulted_sources": [],
            "open_hypotheses": [],
        },
        "missing_facts": [],
        "fetched_sources": {},
        "verified_spans": [],
        "evidence_graph": [],
        "atomic_verified": [],
        "task_ledger": [],
        "search_modes": {},
        "news_hits": [],
        "query_type": "",
        "fruitless": {
            "search_disabled": False,
            "visit_disabled": False,
            "search_streak": 0,
            "visit_streak": 0,
        },
        # Marginal-value stop bookkeeping (critic) — initialized for consistency
        "_marginal_prev_claims": 0,
        "_marginal_prev_urls": 0,
        "outline": [],
        "sections": [],
        "evidence_map": {},
        "report": "",
        "markdown_path": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "mode": "standard",
        "status": "Starting research...",
        "error": "",
        "job_id": "",
        "autonomy": "L1",
        "quality": {
            "max_tokens_per_call": 8000,
            "max_search_results": 10,
            "max_extract_pages": 5,
            "thinker_enabled": False,
            "triangulation_enabled": False,
            "factoid_enabled": False,
        },
        "budgets": {
            "max_tokens": 100000,
            "max_cost_usd": 0.50,
            "max_time_s": 600,
            "max_tool_calls": 20,
            "max_iterations": max_iterations,
            "started_at": time.time(),
            "tool_calls": 0,
            "spent_usd": 0.0,
        },
        "mode_flags": {
            "recency_bias": False,
            "academic_bias": False,
            "structured_output": False,
            "vault_rag": True,
            "requires_temporal": False,
        },
    }
