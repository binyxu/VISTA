#!/usr/bin/env python3
"""Aggregate the proprioceptive-blindness probe into a table + figure.

Reads:
  results/probe_results.jsonl      total / block / pairwise, all backbones
  results/preview5_recency.jsonl   quote-based recency null result (gemini)

Writes:
  results/summary.json, results/summary.txt
  results/fig_proprioception.pdf / .png
"""
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
MODEL_ORDER = ["gemini-3-flash-preview", "claude-sonnet-4-5",
               "deepseek-v4-pro", "glm-5"]
SHORT = {"gemini-3-flash-preview": "Gemini-3-Flash",
         "claude-sonnet-4-5": "Claude-Sonnet-4.5",
         "deepseek-v4-pro": "DeepSeek-V4-Pro", "glm-5": "GLM-5"}


def load(fname):
    p = RES / fname
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open() if l.strip()]


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def _find_pair(pairs, a, b):
    if not isinstance(pairs, dict):
        return None
    for k, v in pairs.items():
        ids = set(re.findall(r"B\d+", str(k)))
        if a in ids and b in ids:
            mm = re.search(r"B\d+", str(v))
            return mm.group(0) if mm else None
    return None


def aggregate(rows):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["model"], r["condition"])
        q, sc = r["question"], r["score"]
        agg[k]["parsed"].append(int(r["parsed"]))
        if q == "total" and sc.get("total_rel_err") is not None:
            agg[k]["total"].append(sc["total_rel_err"])
            tp = (r["answer"] or {}).get("total_tokens")
            tg = r["gt"]["total_tokens"]
            if isinstance(tp, (int, float)):
                agg[k]["total_under"].append(int(tp < tg))
        if q == "block":
            agg[k]["block"].extend(sc.get("block_rel_errs", []))
        if q == "pairwise" and r["parsed"]:
            # re-score with tolerant key matching (a key counts if it names both
            # block ids, any separator/order); fixes false misses from claude's
            # alternate key formatting
            pairs = (r["answer"] or {}).get("pairs") or {}
            gtp = r["gt"]["pairs"]
            ratios = r["gt"].get("ratios", {})
            for key, gt in gtp.items():
                a, b = key.split("|")
                pred = _find_pair(pairs, a, b)
                ok = int(pred == gt)
                agg[k]["pw"].append(ok)
                if ratios.get(key, 9) < 2.0:
                    agg[k]["pw_hard"].append(ok)
    out = {}
    for (m, c), d in agg.items():
        out[f"{m}|{c}"] = {
            "model": m, "condition": c,
            "n_calls": len(d["parsed"]),
            "parse_rate": round(st.mean(d["parsed"]), 3) if d["parsed"] else None,
            "total_median_rel_err": _r(med(d["total"])),
            "total_frac_underestimate": _r(st.mean(d["total_under"])) if d["total_under"] else None,
            "block_median_rel_err": _r(med(d["block"])),
            "pairwise_acc": _r(st.mean(d["pw"])) if d["pw"] else None,
            "pairwise_hard_acc": _r(st.mean(d["pw_hard"])) if d["pw_hard"] else None,
            "pairwise_hard_n": len(d["pw_hard"]),
        }
    return out


def recency_summary(rows):
    out = {}
    agg = defaultdict(list)
    for r in rows:
        agg[r["condition"]].extend(r["score"].get("turns_abs_errs", []))
    for c, es in agg.items():
        out[c] = {"median_abs_err": _r(med(es)), "n": len(es)}
    return out


def _r(x):
    return None if x is None else round(x, 4)


def make_table(summ, rec):
    h = (f"{'model':<20}{'cond':<9}{'parse':>6}{'total_err':>10}"
         f"{'(under)':>9}{'block_err':>10}{'pair_acc':>9}{'pair_hard':>10}")
    lines = [h, "-" * len(h)]
    for m in MODEL_ORDER:
        for c in ["noboard", "board"]:
            s = summ.get(f"{m}|{c}")
            if not s:
                continue
            lines.append(
                f"{SHORT.get(m,m):<20}{c:<9}{_p(s['parse_rate']):>6}"
                f"{_p(s['total_median_rel_err']):>10}"
                f"{_p(s['total_frac_underestimate']):>9}"
                f"{_p(s['block_median_rel_err']):>10}"
                f"{_p(s['pairwise_acc']):>9}{_p(s['pairwise_hard_acc']):>10}")
    lines.append("")
    lines.append("recency (quote-based, gemini): " + ", ".join(
        f"{c}=median_abs_err {v['median_abs_err']} (n={v['n']})"
        for c, v in rec.items()))
    return "\n".join(lines)


def _p(x):
    return "-" if x is None else f"{x:.2f}"


def make_figure(rows, summ, rec):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8})
    COLOR = {"gemini-3-flash-preview": "#2E8B57", "claude-sonnet-4-5": "#C0392B",
             "deepseek-v4-pro": "#2E5AA8", "glm-5": "#E08A1E"}
    fig, axR = plt.subplots(1, 1, figsize=(5.0, 4.3))

    # Predicted vs true total tokens, no dashboard, colored by backbone
    pts = defaultdict(lambda: ([], []))
    for r in rows:
        if r["condition"] != "noboard" or r["question"] != "total":
            continue
        p = (r["answer"] or {}).get("total_tokens")
        g = r["gt"]["total_tokens"]
        if isinstance(p, (int, float)) and g:
            pts[r["model"]][0].append(g)
            pts[r["model"]][1].append(p)
    allx = [v / 1000 for m in pts for v in pts[m][0]]
    ally = [v / 1000 for m in pts for v in pts[m][1]]
    if allx:
        for m in MODEL_ORDER:
            if m not in pts:
                continue
            gx = [v / 1000 for v in pts[m][0]]
            gy = [v / 1000 for v in pts[m][1]]
            axR.scatter(gx, gy, s=26, color=COLOR[m], edgecolor="#2b2b2b",
                        lw=0.4, alpha=0.8, zorder=3, label=SHORT[m])
        lim = max(max(allx), max(ally)) * 1.05
        axR.plot([0, lim], [0, lim], ls="--", color="#444", lw=1.0, zorder=2)
        axR.text(lim * 0.97, lim * 0.9, "perfect ($y\\!=\\!x$)", ha="right",
                 va="top", fontsize=8, color="#444", rotation=38,
                 rotation_mode="anchor")
        axR.set_xlim(0, lim)
        axR.set_ylim(0, lim)
        axR.legend(frameon=False, loc="upper left", handletextpad=0.2,
                   labelspacing=0.25)
    axR.set_xlabel("true context size (K tokens)")
    axR.set_ylabel("model self-estimate (K tokens)")
    axR.set_title("Self-estimate vs. truth, no dashboard")
    axR.grid(True, color="#dddddd", lw=0.8)
    axR.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(RES / "fig_proprioception.pdf", bbox_inches="tight")
    fig.savefig(RES / "fig_proprioception.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load("probe_results.jsonl")
    rec_rows = load("preview5_recency.jsonl")
    summ = aggregate(rows)
    rec = recency_summary(rec_rows)
    (RES / "summary.json").write_text(json.dumps(
        {"main": summ, "recency": rec}, indent=2))
    table = make_table(summ, rec)
    (RES / "summary.txt").write_text(table + "\n")
    print(table)
    if rows:
        make_figure(rows, summ, rec)
        print(f"\nWrote summary + fig_proprioception to {RES}")


if __name__ == "__main__":
    main()
