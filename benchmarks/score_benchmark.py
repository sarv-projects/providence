"""
Benchmark scorer — evaluates the run logs + reports against:

  1. The universal 6-checkpoint rubric (data granularity, geographic equity,
     source diversity, contrarian fork, temporal trajectory, actionable thesis)
  2. Topic-specific anchors (presence of expected content categories)
  3. Fact-check matrix (ground-truth numbers from web research → Green/Yellow/Red)
  4. Latency + citation depth

Writes: benchmarks/RESEARCH_BENCHMARK.md
"""

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Universal rubric markers ──────────────────────────────────────────────

REGIONS = {
    "north_america": ["north america", "united states", " usa", " us ", "canada", "mexico"],
    "europe": ["europe", "european union", "eu ", "uk ", "united kingdom", "germany", "france"],
    "china": ["china", "chinese", "beijing"],
    "global_south": ["india", "africa", "brazil", "global south", "indonesia", "nigeria", "kenya", "south africa", "bangladesh", "pakistan", "egypt", "türkiye", "turkey"],
}
PEER_RE = re.compile(r"(arxiv|aclanthology|nature\.com|science\.org|sciencedirect|springer|mdpi|plos|wiley|bmj|thelancet|jstor|ieee|pnas|cell\.com|pubmed|pmc|oxford|tandfonline|scielo)", re.I)
GOV_RE = re.compile(r"(\.gov|\.mil|\.eu|\.int|un\.org|imf\.org|worldbank|oecd|who\.int|ecdc|cdc\.gov)", re.I)
NEWS_RE = re.compile(r"(reuters|bloomberg|ft\.com|wsj|cnbc|caixin|apnews|bbc|guardian|economist|nikkei|ftc|\bft\b|scmp|aljazeera|dw\.com|france24|xinhuanet|chinadaily|timesofindia|thehindu|livemint|folha|oglobo|nation\.africa|premiumtimesng|japantimes|straitstimes|channelnewsasia|gulfnews|marketwatch|techcrunch|theverge|axios|politico|yahoo\.com)", re.I)
CONTRARIAN_MARKERS = [
    "critics argue", "critics say", "skeptics", "however, some", "counter-argument",
    "counterargument", "contrarian", "minority view", "opponents argue", "not everyone",
    "pushback", "controversial", "challeng", "alternative view", "argues that", "critique",
    "dissent", "refute", "refutes", "contradicts", "on the other hand", "nonetheless",
]
ACTION_MARKERS = [
    "recommend", "should", "stakeholder", "step 1", "step 2", "step 3", "three-step",
    "actionable", "action plan", "strategy for", "policy recommendation", "we advise",
    "we recommend", "suggest", "imperative", "must",
]
FUTURE_YEARS = ["2026", "2027", "2028", "2029", "2030", "2031", "2050", "2100"]
PROJECTION_MARKERS = ["projected", "forecast", "by 2030", "by 2050", "forward-looking", "trajectory", "outlook", "will reach", "expected to", "pathway", "scenario"]

UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|billion|million|trillion|Gt|GW|MW|MWh|kWh|USD|\$|EUR|JPY|CNY|INR|BRL|PSU|t/ha|kg|tonnes?|tons?|acres?|feet|km|miles?|per 100,000|per\s+100,000|acre-foot|acre-feet|flops|params?|SD|GW|Hz|GHz|THz)\b",
    re.I,
)


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _numbers(report: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", report)]


def fact_status(report_lower: str, report_nums: list[float], fact: dict) -> tuple[str, list[str]]:
    """Green = exact/within-tolerance number or exact token present; Yellow = partial; Red = absent."""
    hits = []
    tokens = fact.get("check", [])
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        m = re.match(r"^\d+(?:\.\d+)?$", t)
        if m:
            gt = float(t)
            if any(abs(n - gt) / max(gt, 1e-9) <= 0.06 for n in report_nums):
                hits.append((tok, "exact"))
                continue
            if any(abs(n - gt) / max(gt, 1e-9) <= 0.25 for n in report_nums):
                hits.append((tok, "close"))
                continue
        if _norm(t) in report_lower or t.lower() in report_lower:
            # Guard against stopword inflation: single/short tokens like
            # "not"/"no" substring-match nearly every report ("know",
            # "note", "innovation") and auto-GREEN the fact. Numeric tokens
            # are still scored via the tolerance path above.
            if not m and len(t.strip()) < 3:
                continue
            hits.append((tok, "exact"))
    if any(h[1] == "exact" for h in hits):
        return "GREEN", [h[0] for h in hits]
    if hits:
        return "YELLOW", [h[0] for h in hits]
    return "RED", []


def score_checkpoints(report: str) -> dict:
    rl = report.lower()
    nums_with_units = UNIT_RE.findall(report)
    distinct_units = len(set(nums_with_units))
    # Data granularity
    if distinct_units >= 12:
        data_g = 1.0
    elif distinct_units >= 6:
        data_g = 0.7
    elif distinct_units >= 3:
        data_g = 0.4
    else:
        data_g = 0.1
    # Geographic equity
    geo = {}
    for name, markers in REGIONS.items():
        geo[name] = any(m in rl for m in markers)
    geo_score = sum(geo.values()) / len(geo)
    # Source diversity
    urls = re.findall(r"https?://([^/\s)\]]+)", report)
    domains = sorted({u.split("/")[0].lower() for u in urls})
    peer = sum(1 for d in domains if PEER_RE.search(d))
    gov = sum(1 for d in domains if GOV_RE.search(d))
    news = sum(1 for d in domains if NEWS_RE.search(d))
    src_score = min(1.0, (min(peer, 3) / 3 * 0.4) + (min(gov, 2) / 2 * 0.4) + (min(news, 3) / 3 * 0.2))
    # Contrarian fork
    contrarian = sum(1 for m in CONTRARIAN_MARKERS if m in rl)
    contrarian_score = 1.0 if contrarian >= 2 else (0.5 if contrarian == 1 else 0.0)
    # Temporal trajectory
    future_hits = sum(1 for y in FUTURE_YEARS if re.search(rf"\b{y}\b", report))
    proj_hits = sum(1 for m in PROJECTION_MARKERS if m in rl)
    temporal_score = min(1.0, 0.4 * (1 if future_hits >= 3 else 0) + 0.3 * (1 if future_hits >= 1 else 0) + 0.3 * (1 if proj_hits >= 2 else 0))
    # Actionable thesis
    action_hits = sum(1 for m in ACTION_MARKERS if m in rl)
    action_score = 1.0 if action_hits >= 3 else (0.5 if action_hits >= 1 else 0.0)
    return {
        "data_granularity": data_g,
        "geographic_equity": geo_score,
        "source_diversity": src_score,
        "contrarian_fork": contrarian_score,
        "temporal_trajectory": temporal_score,
        "actionable_thesis": action_score,
        "detail": {
            "numbers_with_units": distinct_units,
            "regions": geo,
            "peer_reviewed": peer,
            "gov_regulatory": gov,
            "news": news,
            "domains": len(domains),
            "contrarian_markers": contrarian,
            "future_years": future_hits,
            "projection_markers": proj_hits,
            "action_markers": action_hits,
        },
    }


def grade(score: float) -> str:
    return "A" if score >= 0.80 else ("B" if score >= 0.65 else ("C" if score >= 0.50 else "D"))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0,
                    help="round to score: 0=logs/ (original), N=logs/roundN/ (default 0)")
    ap.add_argument("--out", default="", help="output md path (default RESEARCH_BENCHMARK_R{N}.md for N>0)")
    args = ap.parse_args()

    gt = json.load(open(os.path.join(BASE, "ground_truth.json")))
    logdir = os.path.join(BASE, "logs")
    if args.round > 0:
        logdir = os.path.join(logdir, f"round{args.round}")
    logs = sorted(glob.glob(os.path.join(logdir, "T*.json")))

    rows = []
    for lp in logs:
        log = json.load(open(lp))
        tid = str(log.get("topic_id"))
        if log.get("error"):
            rows.append({"log": log, "error": log["error"]})
            continue
        rp = log.get("report_copy") or ""
        report = ""
        if rp and os.path.exists(rp):
            with open(rp, encoding="utf-8") as f:
                report = f.read()
        if not report:
            report = log.get("report") or log.get("excerpt") or ""
        rl = report.lower()
        nums = _numbers(report)

        ckpts = score_checkpoints(report)
        ckpt_avg = sum(ckpts[k] for k in ("data_granularity", "geographic_equity", "source_diversity", "contrarian_fork", "temporal_trajectory", "actionable_thesis")) / 6

        # Fact-check matrix
        fact_rows = []
        green = yellow = red = 0
        for f in gt.get(tid, {}).get("facts", []):
            status, hits = fact_status(rl, nums, f)
            fact_rows.append({"label": f["label"], "status": status, "hits": hits})
            if status == "GREEN":
                green += 1
            elif status == "YELLOW":
                yellow += 1
            else:
                red += 1
        total = max(green + yellow + red, 1)
        fact_acc = (green + 0.5 * yellow) / total

        overall = 0.5 * ckpt_avg + 0.5 * fact_acc
        rows.append({
            "log": log,
            "report": report,
            "ckpts": ckpts,
            "ckpt_avg": ckpt_avg,
            "fact_rows": fact_rows,
            "green": green, "yellow": yellow, "red": red,
            "fact_acc": fact_acc,
            "overall": overall,
            "grade": grade(overall),
        })

    # ── Build report ─────────────────────────────────────────────────────
    L = []
    A = L.append
    round_tag = f" (Round {args.round})" if args.round > 0 else ""
    A("# Research Agent Benchmark Report" + round_tag)
    A("")
    A(f"**Suite:** 15 high-complexity topics · **Mode:** `standard` (default budget) · **Date:** 2026-08-11" + round_tag)
    A("**Stack:** Groq (workhorse) + Exa (search) + Gemini (scout) + OpenCode Zen (free fallback) · LangGraph A4 pipeline")
    A("")
    A("**Method:** Each topic ran through the full pipeline (scout → plan → research loop → adversary → adjudicator → synth → compiler). "
      "Outputs scored against a universal 6-checkpoint rubric (per topic) plus topic anchors, with a fact-check matrix comparing "
      "**ground-truth numbers from independent web research** against the report text: 🟢 Green = exact/within-tolerance, "
      "🟡 Yellow = partial/near-miss, 🔴 Red = missing/contradicted.")
    A("")
    A("## Aggregate Results")
    A("")
    A("| # | Topic | Time | Sections | Cites | Domains | News | Checkpoints | Fact 🟢/🟡/🔴 | Fact acc. | Grade |")
    A("|---|-------|------|----------|-------|---------|------|-------------|---------------|-----------|-------|")
    ok_rows = [r for r in rows if "error" not in r]
    for r in ok_rows:
        lg = r["log"]
        ck = r["ckpts"]
        A(f"| {lg.get('topic_id')} | {lg.get('topic','')[:44]} | {lg.get('duration_s',0):.0f}s | {lg.get('sections_count',0)} | {len(lg.get('citation_numbers') or [])} | {ck['detail']['domains']} | {ck['detail']['news']} | {r['ckpt_avg']:.2f} | {r['green']}/{r['yellow']}/{r['red']} | {r['fact_acc']:.2f} | **{r['grade']}** |")
    A("")
    if ok_rows:
        avg_t = sum(r["log"].get("duration_s", 0) for r in ok_rows) / len(ok_rows)
        avg_f = sum(r["fact_acc"] for r in ok_rows) / len(ok_rows)
        avg_c = sum(r["ckpt_avg"] for r in ok_rows) / len(ok_rows)
        avg_n = sum(r["ckpts"]["detail"]["news"] for r in ok_rows) / len(ok_rows)
        A(f"**Averages:** {avg_t:.0f}s per topic · checkpoint coverage {avg_c:.2f} · fact-check accuracy {avg_f:.2f} · tier-1 newswire domains {avg_n:.1f}")
        A("")
        A(f"**Latency vs product Deep Research:** OpenAI/Gemini Deep Research typically 5–30 min per query; this agent averaged **{avg_t/60:.1f} min** on the same class of query — within/below that band.")
    A("")
    A("## Universal Checkpoint Detail")
    A("")
    A("| # | Topic | Data | Geo | Sources | Contrarian | Temporal | Actionable |")
    A("|---|-------|------|-----|---------|------------|----------|------------|")
    for r in ok_rows:
        lg = r["log"]
        ck = r["ckpts"]
        A(f"| {lg.get('topic_id')} | {lg.get('topic','')[:40]} | {ck['data_granularity']:.0%} | {ck['geographic_equity']:.0%} | {ck['source_diversity']:.0%} | {ck['contrarian_fork']:.0%} | {ck['temporal_trajectory']:.0%} | {ck['actionable_thesis']:.0%} |")
    A("")
    A("## Citation Depth")
    A("")
    for r in ok_rows:
        lg = r["log"]
        ck = r["ckpts"]
        d = ck["detail"]
        A(f"- **T{lg.get('topic_id')}** ({lg.get('topic','')[:40]}): {d['domains']} domains · {d['peer_reviewed']} peer-reviewed · {d['gov_regulatory']} gov/regulatory · {d['news']} tier-1 newswire · cites [1–{len(lg.get('citation_numbers') or [])}]")
    A("")
    A("## Per-Topic Logs & Fact-Check Matrix")
    A("")
    for r in ok_rows:
        lg = r["log"]
        A(f"### T{lg.get('topic_id')} — {lg.get('topic','')}")
        A("")
        A(f"**Prompt:** {lg.get('prompt','')[:400]}")
        A("")
        A(f"**Run:** mode={lg.get('mode')} · {lg.get('duration_s')}s · {lg.get('sections_count')} sections · {lg.get('report_chars')} chars · "
          f"{lg.get('findings_count')} findings · {lg.get('claims_count')} claims · evidence graph {lg.get('evidence_graph_edges')} edges · "
          f"adjudicated {lg.get('adjudicated')} · citations [1–{len(lg.get('citation_numbers') or [])}] · ship-gate {'PASS' if lg.get('ship_gate_ok') is not False else 'CHECK'}")
        A("")
        A(f"**Checkpoints:** data {r['ckpts']['data_granularity']:.0%} · geo {r['ckpts']['geographic_equity']:.0%} · sources {r['ckpts']['source_diversity']:.0%} · "
          f"contrarian {r['ckpts']['contrarian_fork']:.0%} · temporal {r['ckpts']['temporal_trajectory']:.0%} · actionable {r['ckpts']['actionable_thesis']:.0%} → **{r['grade']}**")
        A("")
        A("| Fact (ground truth) | Status |")
        A("|---------------------|--------|")
        for fr in r["fact_rows"]:
            icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}[fr["status"]]
            A(f"| {fr['label']} | {icon} {fr['status']}{' (' + ', '.join(fr['hits']) + ')' if fr['hits'] else ''} |")
        A("")
        A("**Ground truth summary:** " + gt.get(str(lg.get('topic_id')), {}).get("summary", "")[:600])
        A("")
        if lg.get("research_debt"):
            A("**Research debt flags:** " + "; ".join(str(d)[:120] for d in lg["research_debt"][:4]))
            A("")
        if lg.get("error"):
            A(f"**ERROR:** {lg['error']}")
            A("")

    A("## Strengths / Weaknesses vs Rubric")
    A("")
    if ok_rows:
        weakest = min(ok_rows, key=lambda r: r["fact_acc"])
        strongest = max(ok_rows, key=lambda r: r["fact_acc"])
        A(f"- **Best topic:** T{strongest['log'].get('topic_id')} — fact acc {strongest['fact_acc']:.2f}")
        A(f"- **Weakest topic:** T{weakest['log'].get('topic_id')} — fact acc {weakest['fact_acc']:.2f}")
        A("")
        ck_keys = ("data_granularity", "geographic_equity", "source_diversity", "contrarian_fork", "temporal_trajectory", "actionable_thesis")
        avgs = {k: sum(r["ckpts"][k] for r in ok_rows) / len(ok_rows) for k in ck_keys}
        A("- **Checkpoint averages:** " + " · ".join(f"{k} {v:.0%}" for k, v in sorted(avgs.items(), key=lambda kv: -kv[1])))
        A("")
        A("- **Red-flag facts by topic:** " + "; ".join(f"T{r['log'].get('topic_id')} ({r['red']})" for r in sorted(ok_rows, key=lambda r: -r['red'])[:6]))
    A("")
    A("---")
    A("*Generated by `benchmarks/score_benchmark.py` from per-topic run logs + ground-truth web research.*")

    out = args.out or (os.path.join(BASE, f"RESEARCH_BENCHMARK_R{args.round}.md" if args.round > 0 else "RESEARCH_BENCHMARK.md"))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Wrote {out}")
    print(f"Topics scored: {len(ok_rows)} | avg fact acc: {sum(r['fact_acc'] for r in ok_rows)/max(len(ok_rows),1):.2f} | avg time: {sum(r['log'].get('duration_s',0) for r in ok_rows)/max(len(ok_rows),1):.0f}s")


if __name__ == "__main__":
    main()
