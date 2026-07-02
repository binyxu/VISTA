"""GPU-free judge for BrowseComp-Plus runs using an API model.

Applies the official GRADER_TEMPLATE (same protocol as evaluate_run.py) via an
OpenAI-compatible endpoint instead of a local vLLM Qwen3-32B, so accuracy can be
computed without a GPU for small experiments. For final numbers, use the official
evaluate_run.py with Qwen3-32B.
"""
import argparse, glob, json, os, re, sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "search_agent"))
from openai import OpenAI
from prompts import GRADER_TEMPLATE


def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            gt[str(o["query_id"])] = {"question": o.get("query", ""), "answer": o.get("answer", "")}
    return gt


def judge_one(client, model, question, response, answer):
    prompt = GRADER_TEMPLATE.format(question=question, response=response, correct_answer=answer)
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=512, temperature=0.0,
    )
    txt = r.choices[0].message.content or ""
    m = re.search(r"correct\s*:\s*(yes|no)", txt, re.I)
    return (m.group(1).lower() == "yes") if m else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--ground_truth", default="data/browsecomp_plus_decrypted.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                    help="Judge model on the endpoint (closest to official Qwen3-32B).")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None,
                    help="Path to persist judge results (json). "
                         "Default: <input_dir>/judge_summary.json")
    args = ap.parse_args()

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LOCA_OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LOCA_OPENAI_API_KEY")
    client = OpenAI(base_url=base_url, api_key=api_key)
    gt = load_gt(args.ground_truth)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.input_dir, "run_*.json"))):
        d = json.load(open(f))
        qid = str(d.get("query_id"))
        if qid not in gt:
            continue
        status = d.get("status")
        resp = d["result"][-1]["output"] if d.get("result") and d["result"][-1]["type"] == "output_text" else ""
        # completion = did it finish within budget on its own (status-based).
        # accuracy is decoupled: grade ANY non-empty final answer, including the
        # forced budget-close answers (status="budget_exhausted"), so a correct
        # answer is never auto-zeroed just because the run ran out of budget.
        completed = status == "completed" and bool((resp or "").strip())
        correct = False
        if (resp or "").strip():
            try:
                correct = judge_one(client, args.model, gt[qid]["question"], resp, gt[qid]["answer"])
            except Exception as e:
                print(f"  judge fail qid={qid}: {str(e)[:80]}")
        rows.append({
            "qid": qid, "status": status, "completed": completed, "correct": correct,
            "total_tokens": d.get("usage", {}).get("total_tokens", 0),
            "tools": d.get("tool_call_counts", {}),
        })

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else 0
    comp = sum(r["completed"] for r in rows) / n if n else 0
    toks = sum(r["total_tokens"] for r in rows) / n if n else 0
    arch = sum(r["tools"].get("context_workspace_archive", 0) for r in rows)
    rec = sum(r["tools"].get("context_workspace_recover", 0) for r in rows)
    label = args.label or Path(args.input_dir).name
    print(f"\n=== {label} (n={n}) ===")
    print(f"accuracy={acc:.3f} | completion={comp:.3f} | avg_total_tokens={toks:,.0f} | archive={arch} recover={rec}")
    for r in sorted(rows, key=lambda x: x["qid"]):
        print(f"  qid={r['qid']} correct={int(r['correct'])} completed={int(r['completed'])} tok={r['total_tokens']:,}")
    # machine-readable line for pairing/McNemar
    print("PERQUERY " + json.dumps({r["qid"]: int(r["correct"]) for r in rows}))

    # persist to disk so the final numbers survive past the terminal session
    out_path = args.out or os.path.join(args.input_dir, "judge_summary.json")
    summary = {
        "label": label,
        "input_dir": os.path.abspath(args.input_dir),
        "judge_model": args.model,
        "n": n,
        "accuracy": acc,
        "completion": comp,
        "avg_total_tokens": toks,
        "archive": arch,
        "recover": rec,
        "perquery": {r["qid"]: int(r["correct"]) for r in rows},
        "rows": sorted(rows, key=lambda x: x["qid"]),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"WROTE {out_path}")


if __name__ == "__main__":
    main()
