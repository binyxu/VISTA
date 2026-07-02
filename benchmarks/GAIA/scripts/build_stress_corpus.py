#!/usr/bin/env python3
"""Build a fixed GAIA long-evidence corpus for VISTA/ReAct comparison."""

import argparse
import json
import re
from pathlib import Path

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


FILLER = (
    "Context filler: this paragraph is deliberately non-answer background used "
    "to preserve a long evidence span. It contains no final answer and should "
    "not be used except to maintain the original long-context pressure. "
)


def token_count(text: str) -> int:
    if _ENC is None:
        return max(1, len(text) // 4)
    return len(_ENC.encode(text, disallowed_special=()))


def pad_to_tokens(text: str, target_tokens: int) -> str:
    out = [text.rstrip(), ""]
    i = 0
    while token_count("\n".join(out)) < target_tokens:
        i += 1
        out.append(f"{FILLER} filler_index={i}.")
    return "\n".join(out)


def clean_id(task_id: str, fallback: int) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(task_id or "")).strip("_")
    return s or f"gaia_{fallback:04d}"


def split_answer(answer: str) -> tuple[str, str]:
    answer = answer.strip()
    if len(answer) <= 1:
        return answer, ""
    mid = max(1, len(answer) // 2)
    return answer[:mid], answer[mid:]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/gaia_2023_all_validation.jsonl")
    parser.add_argument("--out-dir", default="data/stress")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--window", type=int, default=12288)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-tokens", type=int, default=5200)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if args.limit > 0:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    queries_path = out_dir / "queries.tsv"
    gt_path = out_dir / "ground_truth.jsonl"
    manifest_path = out_dir / "manifest.json"

    total_doc_tokens = []
    with corpus_path.open("w", encoding="utf-8") as corpus, \
            queries_path.open("w", encoding="utf-8") as queries, \
            gt_path.open("w", encoding="utf-8") as gt:
        for idx, row in enumerate(rows):
            qid = clean_id(row.get("task_id"), idx)
            question = str(row.get("Question") or "").strip()
            answer = str(row.get("Final answer") or "").strip()
            answer_a, answer_b = split_answer(answer)
            level = str(row.get("Level") or "").strip()
            meta = row.get("Annotator Metadata") or {}
            steps = str(meta.get("Steps") or "").strip()
            tools = str(meta.get("Tools") or "").strip()
            file_name = str(row.get("file_name") or "").strip()

            qtext = (
                f"Task id: {qid}\n"
                f"GAIA question: {question}\n\n"
                "Use the local GAIA evidence corpus. First call search with the task id, "
                "then inspect all four returned shard documents before answering. The "
                "validated GAIA answer is not written in full in any single shard: shard "
                "1 contains answer_fragment_1 and shard 4 contains answer_fragment_2. "
                "Your Exact Answer must be the direct concatenation of those two "
                "fragments after checking the verification shards."
            )
            queries.write(f"{qid}\t{qtext.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
            gt.write(json.dumps({
                "query_id": qid,
                "query": question,
                "answer": answer,
                "level": level,
                "task_id": row.get("task_id"),
            }, ensure_ascii=False) + "\n")

            shard_bodies = [
                (
                    f"GAIA task {qid} shard 1/{args.num_shards}: first answer fragment.\n"
                    f"Question: {question}\n"
                    f"answer_fragment_1: {answer_a}\n"
                    "This is only the first fragment. The final answer is invalid unless "
                    "it is concatenated with answer_fragment_2 from shard 4 after checking "
                    "the verification trail.\n"
                ),
                (
                    f"GAIA task {qid} shard 2/{args.num_shards}: task metadata.\n"
                    f"Difficulty level: {level}\n"
                    f"Attached file name, if any: {file_name or 'none'}\n"
                    f"Tools recorded by annotator: {tools or 'not specified'}\n"
                ),
                (
                    f"GAIA task {qid} shard 3/{args.num_shards}: annotator trajectory.\n"
                    f"{steps or 'No annotator steps were provided.'}\n"
                ),
                (
                    f"GAIA task {qid} shard 4/{args.num_shards}: second answer fragment and final verification.\n"
                    f"answer_fragment_2: {answer_b}\n"
                    "The response should provide an explanation and then an Exact Answer "
                    "formed by concatenating answer_fragment_1 from shard 1 with "
                    "answer_fragment_2 from this shard. This shard confirms the task is "
                    "evaluated by exact-answer matching.\n"
                ),
            ]
            while len(shard_bodies) < args.num_shards:
                shard_bodies.append(f"GAIA task {qid} auxiliary shard {len(shard_bodies)+1}.\n")

            for shard_idx, body in enumerate(shard_bodies[: args.num_shards], start=1):
                docid = f"{qid}::shard{shard_idx}"
                text = pad_to_tokens(body, args.shard_tokens)
                total_doc_tokens.append(token_count(text))
                corpus.write(json.dumps({
                    "docid": docid,
                    "qid": qid,
                    "shard": shard_idx,
                    "title": f"GAIA {qid} evidence shard {shard_idx}",
                    "text": text,
                }, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(Path(args.input).resolve()),
        "n_tasks": len(rows),
        "window_tokens": args.window,
        "num_shards": args.num_shards,
        "target_shard_tokens": args.shard_tokens,
        "avg_doc_tokens": sum(total_doc_tokens) / len(total_doc_tokens) if total_doc_tokens else 0,
        "avg_task_evidence_tokens": (
            sum(total_doc_tokens) / len(rows) if rows else 0
        ),
        "vista_advantage_condition": "avg_task_evidence_tokens > window_tokens",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE {out_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
