#!/usr/bin/env python3
"""API judge for GAIA-formatted run directories."""

import argparse
import glob
import json
import os
import re
from pathlib import Path

from openai import OpenAI


PROMPT = """Judge whether the response answers the GAIA question correctly.

[question]: {question}

[response]: {response}

[correct_answer]: {answer}

Return exactly:
extracted_final_answer: ...
reasoning: ...
correct: yes/no
"""


def load_gt(path: str) -> dict[str, dict[str, str]]:
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gt[str(row["query_id"])] = {
                    "question": row.get("query", ""),
                    "answer": row.get("answer", ""),
                }
    return gt


def exactish(response: str, answer: str) -> bool:
    if not answer.strip():
        return False
    m = re.search(r"Exact Answer:\s*(.+)", response, re.I)
    extracted = (m.group(1) if m else response).strip().lower()
    ans = answer.strip().lower()
    return ans == extracted or ans in extracted


def judge_one(client: OpenAI, model: str, question: str, response: str, answer: str) -> bool:
    if exactish(response, answer):
        return True
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(question=question, response=response, answer=answer)}],
        max_tokens=512,
        temperature=0,
    )
    txt = r.choices[0].message.content or ""
    m = re.search(r"correct\s*:\s*(yes|no)", txt, re.I)
    return bool(m and m.group(1).lower() == "yes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ground-truth", default="data/stress/ground_truth.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LOCA_OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LOCA_OPENAI_API_KEY"),
    )
    gt = load_gt(args.ground_truth)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.input_dir, "run_*.json"))):
        data = json.load(open(path, encoding="utf-8"))
        qid = str(data.get("query_id"))
        if qid not in gt:
            continue
        response = ""
        if data.get("result"):
            response = data["result"][-1].get("output", "")
        correct = False
        if response.strip():
            try:
                correct = judge_one(client, args.model, gt[qid]["question"], response, gt[qid]["answer"])
            except Exception as exc:
                print(f"judge failed qid={qid}: {str(exc)[:120]}")
        rows.append({
            "qid": qid,
            "status": data.get("status"),
            "correct": correct,
            "total_tokens": data.get("usage", {}).get("total_tokens", 0),
            "tools": data.get("tool_call_counts", {}),
        })

    n = len(rows)
    summary = {
        "label": args.label or Path(args.input_dir).name,
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n if n else 0,
        "completion": sum(r["status"] == "completed" for r in rows) / n if n else 0,
        "avg_total_tokens": sum(r["total_tokens"] for r in rows) / n if n else 0,
        "archive": sum(r["tools"].get("context_workspace_archive", 0) for r in rows),
        "recover": sum(r["tools"].get("context_workspace_recover", 0) for r in rows),
        "rows": rows,
        "perquery": {r["qid"]: int(r["correct"]) for r in rows},
    }
    out = Path(args.out or os.path.join(args.input_dir, "judge_summary.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
