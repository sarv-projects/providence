"""
DeepResearch Bench runner — execute Providence against the official 100-task
benchmark and emit results in the format the official harness expects.

Usage:
    uv run python benchmarks/drb_run.py --range 51-55 --mode standard
    uv run python benchmarks/drb_run.py --mode quick          # all 100 tasks

Input:   research/deep_research_bench/data/prompt_data/query.jsonl (official repo)
Output:  research/deep_research_bench/data/test_data/raw_data/providence.jsonl
         {"id": <task_id>, "prompt": "...", "article": "<markdown report>"}

Resumable: tasks already present in the output file are skipped, so a
partial run can be continued with the same command.
"""

import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Harness lives inside the repo at research/deep_research_bench (gitignored).
# NOT /tmp — a reboot wiped /tmp once and took 34 completed reports with it.
DRB_DIR = os.path.expanduser(
    os.getenv("DRB_DIR", os.path.join(PROJECT_ROOT, "research", "deep_research_bench"))
)
QUERY_FILE = os.path.join(DRB_DIR, "data", "prompt_data", "query.jsonl")
# Official harness format (only id / prompt / article)
OUT_FILE = os.path.join(DRB_DIR, "data", "test_data", "raw_data", "providence.jsonl")
# Extra run stats (not part of the official format)
STATS_FILE = os.path.join(BASE_DIR, "logs", "drb_stats.jsonl")
LOG_FILE = os.path.join(BASE_DIR, "logs", "drb_run.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def slugify(text: str, maxlen: int = 48) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else " " for c in text)
    return "_".join(keep.split())[:maxlen].strip("_")


def load_queries() -> list[dict]:
    with open(QUERY_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_ids() -> set[int]:
    if not os.path.exists(OUT_FILE):
        return set()
    ids = set()
    with open(OUT_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def append_result(entry: dict) -> None:
    """Write the official 3-key record (id/prompt/article) plus a stats line."""
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    official = {"id": entry["id"], "prompt": entry["prompt"], "article": entry["article"]}
    with open(OUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(official, ensure_ascii=False) + "\n")
    stats = {k: v for k, v in entry.items() if k != "article"}
    with open(STATS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(stats, ensure_ascii=False) + "\n")


def run_one(query: dict, mode: str) -> dict:
    from src.graph import run_research

    qid = query["id"]
    prompt = query["prompt"]
    t0 = time.time()
    print(f"\n{'='*70}\n[DRB task {qid}] ({query.get('topic','')}/{query.get('language','')}) mode={mode}\n{'='*70}", flush=True)
    try:
        result = run_research(prompt, mode=mode, autonomy="L1", skip_clarify=True)
        report = result.get("report") or ""
        if not report:
            md_path = result.get("markdown_path", "")
            if md_path and os.path.isfile(md_path):
                with open(md_path, encoding="utf-8", errors="replace") as f:
                    report = f.read()
        article = report.strip()
        log_line = {
            "id": qid,
            "prompt": prompt,
            "language": query.get("language", ""),
            "topic": query.get("topic", ""),
            "duration_s": round(time.time() - t0, 1),
            "chars": len(article),
            "sections": len(result.get("sections") or []),
            "findings": len(result.get("findings") or []),
            "error": None,
            "article": article,
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log_line = {
            "id": qid,
            "prompt": prompt,
            "language": query.get("language", ""),
            "topic": query.get("topic", ""),
            "duration_s": round(time.time() - t0, 1),
            "chars": 0,
            "sections": 0,
            "findings": 0,
            "error": str(exc),
            "article": "",
        }
    return log_line


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepResearch Bench runner")
    ap.add_argument("--range", default=None, help="e.g. 51-55 or 0-49 (ids are 1-based)")
    ap.add_argument("--mode", default="standard")
    args = ap.parse_args()

    queries = load_queries()
    if args.range:
        lo, hi = (int(x) for x in args.range.split("-"))
        queries = [q for q in queries if lo <= q["id"] <= hi]

    done = load_done_ids()
    pending = [q for q in queries if q["id"] not in done]
    print(f"queries: {len(queries)} pending: {len(pending)} already-done: {len(queries)-len(pending)}", flush=True)

    ok = fail = 0
    for q in pending:
        log_line = run_one(q, args.mode)
        # An empty article means the run produced nothing — do NOT record it
        # as done, or resume (load_done_ids) skips it forever in the 100-task
        # run. Retry next invocation instead.
        if not (log_line.get("article") or "").strip():
            fail += 1
            print(f"[task {q['id']}] EMPTY: no article produced — not recorded, will retry", flush=True)
            continue
        append_result(log_line)
        if log_line["error"]:
            fail += 1
            print(f"[task {q['id']}] ERROR: {log_line['error'][:200]}", flush=True)
        else:
            ok += 1
            print(f"[task {q['id']}] ok · {log_line['chars']} chars · {log_line['duration_s']}s", flush=True)
        pass  # stats already written by append_result

    print(f"\nDONE · ok={ok} fail={fail} · output: {OUT_FILE}")


if __name__ == "__main__":
    main()
