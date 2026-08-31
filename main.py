"""
Providence — Deep Research Engine
==================================
Multi-agent research: Planner · Researcher · Critic · Synthesizer · Compiler
Gateway + RAG + progressive section output + citation ship-gate

Commands:
    uv run python main.py research "topic" [--mode MODE]   Run research
    uv run python main.py chat                              Start chat session
    uv run python main.py doctor                            Check provider/tool status
    uv run python main.py --history                         Show past searches
"""

import sys
import time

from src.graph import run_research
from src.llm import call_llm, gateway_info
from src.memory import get_history, save_search, find_similar
from src.eval import create_component_evaluator, create_system_evaluator
from src.web import app


def _should_escalate_to_research(text: str) -> bool:
    # Shared heuristic (single source of truth — also used by the web API)
    from src.engine.escalate import should_escalate_to_research
    return should_escalate_to_research(text)


def print_header(text: str) -> None:
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str) -> None:
    print()
    print(f"  {text}")
    print("  " + "-" * 40)


def _check_tavily() -> tuple[bool, str]:
    import os
    key = os.getenv("TAVILY_API_KEY", "")
    # Never print a key prefix — even partial key material in logs is a leak.
    return bool(key), "set" if key else "not set"


def doctor() -> None:
    """Display provider/tool readiness."""
    print_header("SYSTEM DOCTOR")

    print_section("LLM Gateway (Zen free is PRIMARY)")
    info = gateway_info()
    print(
        f"  Fast: {info['fast_routes']}  |  Strong: {info['strong_routes']}"
        f"  |  Thinker: {info.get('thinker_routes', 0)}"
    )
    if info["routes"]:
        print()
        for r in info["routes"]:
            key_status = "free" if not r["has_key"] else "key"
            print(f"  [{r['tier']}] {r['provider']}/{r['model']}  [{key_status}]")
    else:
        print("  No routes — check .env")

    print_section("OpenCode Zen free")
    try:
        from src.providers.catalog import load_catalog
        from src.providers.models_catalog import probe_model
        slot = load_catalog().providers.get("opencode_free")
        models = list(slot.models) if slot else []
        print(f"  Configured free models: {', '.join(models[:6])}{'…' if len(models)>6 else ''}")
        print(f"  Primary default: {models[0] if models else 'nemotron-3-ultra-free'}")
        # One quick live check only (full matrix: Settings → Test Zen free models)
        probe_id = models[0] if models else "nemotron-3-ultra-free"
        r = probe_model("opencode_free", probe_id, timeout=20.0)
        if r.get("ok"):
            print(f"  Live check OK  {probe_id}  ({r.get('latency_s')}s)  {(r.get('reply') or '')[:40]}")
        else:
            print(f"  Live check FAIL {probe_id}: {(r.get('error') or '')[:80]}")
        print("  Full free-model matrix: uv run / Settings → Test Zen free models")
    except Exception as e:
        print(f"  Probe skipped: {e}")

    try:
        from src.engine.temporal.client import temporal_configured
        print_section("Temporal")
        print(f"  Configured: {'yes' if temporal_configured() else 'no (in-process fallback)'}")
        print(f"  Worker: uv run python main.py worker")
    except Exception:
        pass

    from src.tools import get_registry
    print_section("Research Tools")
    registry = get_registry()
    for tool in registry.list_all():
        caps = ", ".join(sorted(tool.capabilities))
        free_paid = "free (no config)" if "free" in tool.capabilities else "needs API key"
        print(f"  {tool.name:<20} p={tool.priority:<4} {free_paid}")
        print(f"  {'':20}    [{caps}]")

    tavily_ok, preview = _check_tavily()
    if not tavily_ok:
        print(f"\n  💡 Tip: set TAVILY_API_KEY for comprehensive web search")

    print_section("Environment")
    import os
    for var in ["GROQ_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
                "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
                "NVIDIA_API_KEY", "CO_API_KEY"]:
        print(f"  {var:<25} {'set' if os.getenv(var) else '  -'}")

    print_section("Modes")
    print("  chat | quick | standard | deep | recency | academic | compare | ultra-long")
    print()


def chat(mode: str = "chat") -> None:
    """Start a chat session with conversation memory."""
    from src.engine.modes import load_modes, get_mode
    from src.rag.chat_memory import get_chat_memory, reset_chat_memory

    registry = load_modes()
    chat_mode = get_mode(registry, mode)
    memory = get_chat_memory()

    print_header(f"CHAT ({chat_mode.description})")
    print(f"  Budget: ${chat_mode.budgets.max_cost_usd:.2f} max")
    print(f"  Memory: {len(memory)} messages loaded")
    print("  /exit /doctor /research <topic> /clear")
    print()

    SYSTEM_PROMPT = (
        "You are a helpful, knowledgeable research assistant. "
        "Answer accurately and cite sources when possible. "
        "Be concise but thorough."
    )

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "/q"):
            print("  Goodbye!"); break
        if user_input.lower() == "/doctor":
            doctor(); continue
        if user_input.lower() == "/clear":
            reset_chat_memory()
            memory = get_chat_memory()
            print("  Memory cleared."); continue
        if user_input.lower().startswith("/research"):
            topic = user_input[len("/research"):].strip()
            if topic:
                print(f"  Escalating to research: {topic}")
                _run_research(topic)
            else:
                print("  Usage: /research <topic>")
            continue

        # Auto-escalate to research when mode allows and query looks research-heavy
        if chat_mode.escalate_to_research and _should_escalate_to_research(user_input):
            print(f"  ↗ Escalating to research (detected deep-research intent)...")
            memory.add("user", user_input)
            _run_research(user_input, mode="standard")
            memory.add("assistant", f"[Escalated to research] Topic: {user_input}")
            continue

        # Add user message to memory and build multi-turn context
        memory.add("user", user_input)
        messages = memory.build_context(SYSTEM_PROMPT)

        # Format prior turns into the user prompt (gateway takes system+user strings)
        history_lines = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                continue
            history_lines.append(f"{role.upper()}: {m.get('content', '')}")
        # Keep recent window only
        history_block = "\n".join(history_lines[-12:])
        user_prompt = (
            f"Conversation so far:\n{history_block}\n\n"
            f"Respond to the latest user message."
            if history_lines
            else user_input
        )

        print("  ", end="", flush=True)
        try:
            response = call_llm(SYSTEM_PROMPT, user_prompt)
            print(f"Assistant: {response}\n")
            memory.add("assistant", response)
        except RuntimeError as e:
            print(f"\n  Error: {e}\n")


def _run_research(query: str, mode: str = "standard", autonomy: str = "L1") -> None:
    """Run multi-agent research with progressive output display."""
    from src.engine.modes import load_modes, get_mode
    registry = load_modes()
    mode_config = get_mode(registry, mode)

    print_header(f"RESEARCH: {query}")
    print(f"  Mode: {mode} ({mode_config.description})")
    print(f"  Autonomy: {autonomy}")
    print(f"  Budget: {mode_config.budgets.max_iterations} iters, ${mode_config.budgets.max_cost_usd:.2f} max")
    start = time.time()

    try:
        result = run_research(query, mode=mode, autonomy=autonomy)
        elapsed = time.time() - start

        print_header("RESEARCH COMPLETE")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Iterations: {result.get('iteration', 0)}")
        print(f"  Findings: {len(result.get('findings', []))}")
        print(f"  Claims: {len(result.get('claims', []))}")
        print(f"  Sources tracked: {len(result.get('evidence_map', {}))}")
        print(f"  Report: {result.get('markdown_path', 'N/A')}")
        print()

        # Show section-by-section preview
        sections = result.get("sections", [])
        if sections:
            print("─" * 60)
            for s in sections:
                content = s.get("content", "")
                print(f"  ## {s['title']} ({len(content)} chars)")
            print("─" * 60)
            if sections:
                preview = sections[0].get("content", "")
                print(preview[:400])
                if len(preview) > 400:
                    print("...")
                print("─" * 60)
        else:
            # Fallback to legacy output
            preview = result.get("report", "")[:500]
            if preview:
                print("─" * 60)
                print(preview)
                print("...")
                print("─" * 60)

        # Save to memory
        save_search(
            query=query,
            search_queries=result.get("search_queries", []),
            report_path=result.get("markdown_path", ""),
            findings=result.get("findings", []),
        )

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()


def show_history() -> None:
    """Display past searches."""
    print_header("PAST RESEARCHES")
    history = get_history(20)
    if not history:
        print("  No past searches found.")
    for entry in history:
        print(f"  {entry['timestamp']} — {entry['query']}")
        print(f"    Report: {entry.get('report_path', 'N/A')}")


def run_eval(suite: str = "all") -> None:
    """Run evaluation suites."""
    print_header(f"EVALUATION: {suite.upper()}")
    
    if suite == "component":
        evaluator = create_component_evaluator()
        suites = evaluator.run_all_component_suites()
    elif suite == "system":
        evaluator = create_system_evaluator()
        suites = evaluator.run_system_suites()
    elif suite == "all":
        evaluator = create_component_evaluator()
        component_suites = evaluator.run_all_component_suites()
        
        evaluator = create_system_evaluator()
        system_suites = evaluator.run_system_suites()
        
        suites = component_suites + system_suites
    else:
        print(f"  Unknown suite: {suite}")
        print("  Available: component, system, all")
        return
    
    print(f"  Running {len(suites)} evaluation suites...")
    print()
    
    total_passed = 0
    total_tests = 0
    
    for suite in suites:
        print(f"  Suite: {suite.name}")
        print(f"    Pass rate: {suite.pass_rate:.2%} ({suite.passed_count}/{suite.total_count})")
        
        for result in suite.results:
            status = "✅" if result.passed else "❌"
            print(f"    {status} {result.name}: {result.score:.2f} ({result.duration_seconds:.2f}s)")
        
        total_passed += suite.passed_count
        total_tests += suite.total_count
        print()
    
    overall_rate = total_passed / total_tests if total_tests > 0 else 0.0
    print(f"  Overall: {overall_rate:.2%} ({total_passed}/{total_tests})")
    print()
    
    if overall_rate < 0.8:
        print("  ⚠️  Warning: Overall pass rate below 80%")
    else:
        print("  ✅ All evaluations passed")


def print_usage() -> None:
    print("Providence — Deep Research Engine")
    print()
    print("Commands:")
    print('  uv run python main.py research "topic" [--mode MODE]')
    print("  uv run python main.py chat [--mode MODE]")
    print("  uv run python main.py doctor")
    print("  uv run python main.py eval [suite]")
    print("  uv run python main.py server")
    print("  uv run python main.py worker          # Temporal durable worker")
    print("  uv run python main.py --history")
    print()
    print("Modes: chat | quick | standard | deep | recency | academic | compare | ultra-long")
    print("Autonomy: --autonomy L1|L2|L3")
    print()
    print("Eval suites: component | system | all")


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print_usage()
        return

    if "--history" in args:
        show_history()
        return

    if args[0] == "doctor":
        doctor()
        return

    if args[0] == "eval":
        suite = "all"
        if len(args) > 1:
            suite = args[1]
        run_eval(suite)
        return

    if args[0] == "server":
        import os
        import uvicorn
        port_raw = os.getenv("PORT", "8001")
        try:
            port = int(port_raw)
        except ValueError:
            print(f"  Invalid PORT '{port_raw}' — defaulting to 8001")
            port = 8001
        print("Starting web API server...")
        print(f"API docs: http://localhost:{port}/docs")
        uvicorn.run(app, host="0.0.0.0", port=port)
        return

    if args[0] == "worker":
        # Temporal worker for ultra-long / durable research
        try:
            import asyncio
            from temporalio.client import Client
            from temporalio.worker import Worker
            from src.engine.temporal.workflows import ResearchWorkflow, HumanInLoopWorkflow
            from src.engine.temporal.activities import (
                plan_research_activity,
                research_subtask_activity,
                synthesize_report_activity,
                human_approval_activity,
                gateway_llm_activity,
            )
        except ImportError as e:
            print(f"Temporal worker dependencies missing: {e}")
            print("Install temporalio and start a Temporal server (localhost:7233).")
            sys.exit(1)

        address = __import__("os").getenv("TEMPORAL_SERVER_ADDRESS", "localhost:7233")
        task_queue = __import__("os").getenv("TEMPORAL_TASK_QUEUE", "research-agent")

        async def _run_worker() -> None:
            print(f"Connecting Temporal worker → {address} queue={task_queue}")
            client = await Client.connect(address)
            worker = Worker(
                client,
                task_queue=task_queue,
                workflows=[ResearchWorkflow, HumanInLoopWorkflow],
                activities=[
                    plan_research_activity,
                    research_subtask_activity,
                    synthesize_report_activity,
                    human_approval_activity,
                    gateway_llm_activity,
                ],
            )
            print("Temporal worker running. Ctrl+C to stop.")
            await worker.run()

        asyncio.run(_run_worker())
        return

    if args[0] == "chat":
        mode = "chat"
        if "--mode" in args:
            idx = args.index("--mode")
            if idx + 1 < len(args):
                mode = args[idx + 1]
        chat(mode=mode)
        return

    if args[0] == "research":
        remaining = args[1:]
    else:
        remaining = args

    mode = "standard"
    autonomy = "L1"
    query_parts = []
    i = 0
    while i < len(remaining):
        if remaining[i] == "--mode" and i + 1 < len(remaining):
            mode = remaining[i + 1]
            i += 2
        elif remaining[i] == "--autonomy" and i + 1 < len(remaining):
            autonomy = remaining[i + 1].upper()
            i += 2
        else:
            query_parts.append(remaining[i])
            i += 1

    query = " ".join(query_parts)
    if not query:
        print("Error: Please provide a research topic.")
        print('Usage: uv run python main.py research "your topic" [--mode standard] [--autonomy L1]')
        sys.exit(1)

    similar = find_similar(query)
    if similar:
        print_header("PAST SIMILAR RESEARCH FOUND")
        for s in similar:
            print(f"  {s['timestamp']} — {s['query']}")
            print(f"    Report: {s.get('report_path', 'N/A')}")
        print()

    _run_research(query, mode=mode, autonomy=autonomy)


if __name__ == "__main__":
    main()
