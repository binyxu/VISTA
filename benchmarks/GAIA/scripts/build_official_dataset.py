#!/usr/bin/env python3
"""Build GAIA validation/test query files without leaking answers to agents."""

import argparse
import json
import random
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/gaia_2023_all_validation.jsonl")
    ap.add_argument("--out-dir", default="data/official/validation")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-size", type=int, default=0, help="Random sample size after filtering.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for --sample-size.")
    ap.add_argument("--levels", default="", help="Optional comma list, e.g. 1,2")
    ap.add_argument("--task-ids", default="", help="Optional comma-list or file of task ids to include, in order.")
    args = ap.parse_args()

    rows = load_rows(Path(args.input))
    if args.task_ids:
        task_arg = Path(args.task_ids)
        if task_arg.exists():
            ids = [x.strip() for x in task_arg.read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            ids = [x.strip() for x in args.task_ids.split(",") if x.strip()]
        by_id = {str(r.get("task_id")): r for r in rows}
        rows = [by_id[x] for x in ids if x in by_id]
    if args.levels:
        keep = {x.strip() for x in args.levels.split(",") if x.strip()}
        rows = [r for r in rows if str(r.get("Level")) in keep]
    if args.sample_size > 0 and not args.task_ids:
        rng = random.Random(args.seed)
        rows = rng.sample(rows, min(args.sample_size, len(rows)))
    if args.limit > 0:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = out_dir / "queries.tsv"
    gt = out_dir / "ground_truth.jsonl"
    meta = out_dir / "metadata_public.jsonl"

    with queries.open("w", encoding="utf-8") as qf, gt.open("w", encoding="utf-8") as gf, meta.open("w", encoding="utf-8") as mf:
        for row in rows:
            qid = str(row["task_id"])
            question = str(row.get("Question") or "").strip()
            file_name = str(row.get("file_name") or "").strip()
            file_path = str(row.get("file_path") or "").strip()
            if file_name:
                question = (
                    f"{question}\n\nAttached file available to tools: {file_name}. "
                    "Use read_attachment or python_execute if needed."
                )
            qf.write(f"{qid}\t{question.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
            gf.write(json.dumps({
                "task_id": qid,
                "query_id": qid,
                "level": str(row.get("Level") or ""),
                "answer": row.get("Final answer", ""),
            }, ensure_ascii=False) + "\n")
            mf.write(json.dumps({
                "task_id": qid,
                "Question": row.get("Question", ""),
                "Level": row.get("Level", ""),
                "file_name": file_name,
                "file_path": file_path,
            }, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(Path(args.input).resolve()),
        "n_tasks": len(rows),
        "sample_size": args.sample_size,
        "seed": args.seed if args.sample_size > 0 else None,
        "levels": sorted({str(r.get("Level")) for r in rows}),
        "with_files": sum(bool(r.get("file_name")) for r in rows),
        "answer_leakage": "Final answer is only in ground_truth.jsonl; queries and metadata_public omit it.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE {out_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
