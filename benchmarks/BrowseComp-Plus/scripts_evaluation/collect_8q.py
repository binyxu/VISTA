#!/usr/bin/env python3
"""Collect 8-query BrowseComp-Plus baseline results into one table.
Reads judge_summary.json + per-run JSON across the pilot (clear/mask/skeleton) and
the anchors+ablations sweep, reporting accuracy, completion, and both token meters
(budget_used median = active context; usage.total_tokens mean = cumulative API)."""
import json, glob, os, sys, statistics as st

RUNS = "runs"

def latest(prefix):
    ds = sorted(glob.glob(os.path.join(RUNS, prefix)), key=os.path.getmtime)
    return ds[-1] if ds else None

# (label, glob-pattern for the run dir)
ROWS = [
    ("ReAct (truncate)",        "ab8_*_react"),
    ("Tool-result Clearing",    "reactcm_pilot8_*_clear"),
    ("Stale-obs Masking",       "reactcm_pilot8_*_mask"),
    ("Skeleton Compression",    "reactcm_pilot8_*_skeleton"),
    ("SLIM (summary)",          "ab8slim_*_summary"),
    ("Auto-Archive+Recover",    "ab8_*_ab_autoarch"),
    ("VISTA w/o dashboard",     "ab8_*_ab_nodash"),
    ("VISTA w/o recovery",      "ab8_*_ab_norecov"),
    ("VISTA (full)",            "ab8_*_vista_full"),
]

def stats(d):
    js = os.path.join(d, "judge_summary.json")
    acc = compl = None
    if os.path.exists(js):
        j = json.load(open(js))
        acc, compl = j.get("accuracy"), j.get("completion")
    bud, api = [], []
    for f in glob.glob(os.path.join(d, "run_*.json")):
        r = json.load(open(f)); m = r.get("metadata", {}); u = r.get("usage", {})
        if m.get("budget_used") is not None: bud.append(m["budget_used"])
        if u.get("total_tokens") is not None: api.append(u["total_tokens"])
    return acc, compl, bud, api

print(f"{'Method':24} {'Acc':>6} {'Compl':>6} {'BudMed':>8} {'APImean':>9} {'n':>3}")
print("-"*62)
for label, pat in ROWS:
    d = latest(pat)
    if not d:
        print(f"{label:24} {'--':>6} {'--':>6} {'--':>8} {'--':>9} {'0':>3}  (not run)")
        continue
    acc, compl, bud, api = stats(d)
    accs = f"{acc:.3f}" if acc is not None else "--"
    cs = f"{compl:.3f}" if compl is not None else "--"
    bm = f"{st.median(bud):.0f}" if bud else "--"
    am = f"{st.mean(api):.0f}" if api else "--"
    print(f"{label:24} {accs:>6} {cs:>6} {bm:>8} {am:>9} {len(bud):>3}")
