#!/usr/bin/env python3
"""Quasi-exact scorer for GAIA validation-style outputs."""

import argparse
import glob
import json
import os
import re
import string
from pathlib import Path


FINAL_RE = re.compile(r"FINAL ANSWER\s*:\s*(.+)", re.I | re.S)


def extract_final(text: str) -> str:
    text = text or ""
    m = FINAL_RE.search(text)
    if m:
        ans = m.group(1).strip()
        ans = re.split(r"\n\s*\n|(?:^|\n)\s*(?:Reasoning|Explanation)\s*:", ans, maxsplit=1)[0].strip()
        return ans.strip().strip("`")
    # Fallback for older client outputs.
    m = re.search(r"Exact Answer\s*:\s*(.+)", text, re.I)
    if m:
        return m.group(1).strip()
    return text.strip()


def normalize(s: str) -> str:
    return normalize_str(s)


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = str(number_str).replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(s: str, char_list: list[str] = [",", ";"]) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, str(s))


def normalize_str(input_str, remove_punct=True) -> str:
    no_spaces = re.sub(r"\s", "", str(input_str))
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def official_question_scorer(model_answer: str, ground_truth: str) -> bool:
    """Mirror GAIA leaderboard scorer.py question_scorer."""
    if model_answer is None:
        model_answer = "None"
    ground_truth = str(ground_truth)
    if is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)
    if any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                comparisons.append(normalize_number_str(ma_elem) == float(gt_elem))
            else:
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)
    return normalize_str(model_answer) == normalize_str(ground_truth)


def quasi_exact(pred: str, gold: str) -> bool:
    pred = extract_final(pred)
    return official_question_scorer(pred, gold)


def load_gt(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[str(r["task_id"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--queries", default="", help="Optional expected query TSV; missing outputs count incorrect.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--submission-out", default=None)
    args = ap.parse_args()

    gt = load_gt(args.ground_truth)
    rows = []
    submissions = []
    latest_by_qid = {}
    for p in sorted(glob.glob(os.path.join(args.input_dir, "run_*.json")), key=lambda x: os.path.getmtime(x)):
        d = json.load(open(p, encoding="utf-8"))
        qid = str(d.get("query_id"))
        if qid not in gt:
            continue
        latest_by_qid[qid] = d

    seen = set()
    for qid, d in latest_by_qid.items():
        seen.add(qid)
        raw = d["result"][-1].get("output", "") if d.get("result") else ""
        ans = extract_final(raw)
        correct = quasi_exact(raw, gt[qid]["answer"])
        row = {
            "qid": qid,
            "level": str(gt[qid].get("level", "")),
            "status": d.get("status"),
            "model_answer": ans,
            "gold": gt[qid]["answer"],
            "correct": correct,
            "tools": d.get("tool_call_counts", {}),
            "budget_used": d.get("metadata", {}).get("budget_used"),
        }
        rows.append(row)
        submissions.append({
            "task_id": qid,
            "model_answer": ans,
            "reasoning_trace": raw,
        })

    if args.queries:
        with open(args.queries, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                qid = line.split("\t", 1)[0].strip()
                if not qid or qid not in gt or qid in seen:
                    continue
                rows.append({
                    "qid": qid,
                    "level": str(gt[qid].get("level", "")),
                    "status": "missing",
                    "model_answer": "",
                    "gold": gt[qid]["answer"],
                    "correct": False,
                    "tools": {},
                    "budget_used": 0,
                })
                submissions.append({
                    "task_id": qid,
                    "model_answer": "",
                    "reasoning_trace": "",
                })

    n = len(rows)
    by_level = {}
    for lvl in sorted({r["level"] for r in rows}):
        arr = [r for r in rows if r["level"] == lvl]
        by_level[lvl] = {
            "n": len(arr),
            "accuracy": sum(r["correct"] for r in arr) / len(arr) if arr else 0,
            "correct": sum(r["correct"] for r in arr),
        }
    summary = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n if n else 0,
        "by_level": by_level,
        "rows": rows,
    }
    out = Path(args.out or os.path.join(args.input_dir, "official_score.json"))
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    sub = Path(args.submission_out or os.path.join(args.input_dir, "submission.jsonl"))
    with sub.open("w", encoding="utf-8") as f:
        for r in submissions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"WROTE {out}")
    print(f"WROTE {sub}")


if __name__ == "__main__":
    main()
