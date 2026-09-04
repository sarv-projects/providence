"""Compute aggregate + per-topic benchmark stats for the README and reports."""
import glob
import json
import os
import statistics
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from score_benchmark import score_checkpoints, fact_status, _numbers, grade  # noqa: E402

def load():
    gt = json.load(open(os.path.join(BASE, "ground_truth.json")))
    rows = []
    for p in sorted(glob.glob(os.path.join(BASE, "logs", "T*.json"))):
        d = json.load(open(p))
        rp = d.get("report_copy") or ""
        rep = open(rp, encoding="utf-8").read() if rp and os.path.exists(rp) else (d.get("report") or d.get("excerpt") or "")
        rows.append({"log": d, "report": rep})
    return gt, rows

def main():
    gt, rows = load()
    print(f"topics: {len(rows)}")
    durs = [r["log"].get("duration_s", 0) for r in rows]
    secs = [r["log"].get("sections_count", 0) for r in rows]
    cites = [len(r["log"].get("citation_numbers") or []) for r in rows]
    doms = [len(r["log"].get("sources_domains") or []) for r in rows]
    chars = [r["log"].get("report_chars", 0) for r in rows]
    print(f"avg duration: {statistics.mean(durs):.0f}s ({statistics.mean(durs)/60:.1f} min) | min/max: {min(durs):.0f}/{max(durs):.0f}s")
    print(f"avg sections: {statistics.mean(secs):.1f}")
    print(f"avg citations: {statistics.mean(cites):.1f}")
    print(f"avg unique domains: {statistics.mean(doms):.1f}")
    print(f"avg chars: {statistics.mean(chars):.0f}")

    ckpts, faccs, greys, yellows, reds = [], [], [], [], []
    print(f"\n{'T':>3} {'topic':<48} {'ckpt':>5} {'fact':>5} {'G/Y/R':>8} {'grade':>6}")
    for r in rows:
        lg = r["log"]
        rl = r["report"].lower()
        nums = _numbers(r["report"])
        c = score_checkpoints(r["report"])
        ck = sum(c[k] for k in ("data_granularity", "geographic_equity", "source_diversity",
                                "contrarian_fork", "temporal_trajectory", "actionable_thesis")) / 6
        g = y = red = 0
        for f in gt.get(str(lg["topic_id"]), {}).get("facts", []):
            st, _h = fact_status(rl, nums, f)
            if st == "GREEN":
                g += 1
            elif st == "YELLOW":
                y += 1
            else:
                red += 1
        tot = max(g + y + red, 1)
        fa = (g + 0.5 * y) / tot
        ckpts.append(ck); faccs.append(fa); greys.append(g); yellows.append(y); reds.append(red)
        grd = grade(0.5 * ck + 0.5 * fa)
        print(f"{int(lg['topic_id']):>3} {lg['topic'][:48]:48s} {ck:.2f} {fa:.2f} {g}/{y}/{red:>2} {grd}")

    print(f"\navg checkpoint: {statistics.mean(ckpts):.2f}")
    print(f"avg fact acc:   {statistics.mean(faccs):.2f}")
    print(f"avg overall:    {statistics.mean([0.5*c+0.5*f for c, f in zip(ckpts, faccs)]):.2f}")
    print(f"facts: green={sum(greys)} yellow={sum(yellows)} red={sum(reds)}")
    print(f"grades: {len([1 for _ in ckpts])} topics | A={sum(1 for c,f in zip(ckpts,faccs) if grade(0.5*c+0.5*f)=='A')} "
          f"B={sum(1 for c,f in zip(ckpts,faccs) if grade(0.5*c+0.5*f)=='B')} "
          f"C={sum(1 for c,f in zip(ckpts,faccs) if grade(0.5*c+0.5*f)=='C')} "
          f"D={sum(1 for c,f in zip(ckpts,faccs) if grade(0.5*c+0.5*f)=='D')}")

if __name__ == "__main__":
    main()
