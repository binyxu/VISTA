#!/usr/bin/env python3
"""Best-effort runtime/token analysis for completed AMA-Bench runs.

This script analyzes artifacts that already exist after a run. It does not
recover exact provider billing usage, because the harness did not persist API
usage for these runs. The output is still useful for comparing methods under
the same setting: per-question score, episode size, observed episode runtime,
observed judge runtime, and approximate text-token counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


def approx_tokens(text: str) -> int:
    """Small dependency-free token estimate.

    This is intentionally conservative and stable across machines. For mixed
    English/code logs, chars/4 is a reasonable rough proxy; never treat it as
    provider billing truth.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def parse_duration(s: str) -> float:
    parts = [float(p) for p in s.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported duration: {s}")


def load_dataset(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["episode_id"])] = row
    return rows


def load_answers(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["episode_id"])] = row
    return rows


def split_reasoning_trace(trace: str, n: int) -> list[str]:
    if not trace:
        return [""] * n
    matches = list(re.finditer(r"(?:^|\n)Q(\d+) Reasoning:\n", trace))
    if len(matches) < n:
        return [trace] + [""] * (n - 1)
    chunks = []
    for i, m in enumerate(matches[:n]):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(trace)
        chunks.append(trace[start:end].strip())
    while len(chunks) < n:
        chunks.append("")
    return chunks


def parse_generation_times(log_text: str) -> dict[int, float]:
    # tqdm rewrites progress lines with carriage returns. Treat both CR and LF
    # as separators, then use the latest postfix for each completed count.
    lines = re.split(r"[\r\n]+", log_text)
    count_to_latest: dict[int, tuple[float, int]] = {}
    pat = re.compile(
        r"Processing episodes:\s+.*?\|\s*(\d+)/(\d+)\s+\[([0-9:]+)<.*?Episode=(\d+),"
    )
    for line in lines:
        m = pat.search(line)
        if not m:
            continue
        count = int(m.group(1))
        elapsed = parse_duration(m.group(3))
        episode_id = int(m.group(4))
        count_to_latest[count] = (elapsed, episode_id)

    cumulative = {episode_id: elapsed for _, (elapsed, episode_id) in count_to_latest.items()}
    ordered = sorted(count_to_latest.items())
    runtimes: dict[int, float] = {}
    prev_elapsed = 0.0
    for _, (elapsed, episode_id) in ordered:
        runtimes[episode_id] = max(0.0, elapsed - prev_elapsed)
        prev_elapsed = elapsed
    return runtimes


def parse_judge_times(log_text: str) -> list[float]:
    lines = re.split(r"[\r\n]+", log_text)
    count_to_elapsed: dict[int, float] = {}
    pat = re.compile(r"Evaluating QA pairs:\s+.*?\|\s*(\d+)/(\d+)\s+\[([0-9:]+)<")
    for line in lines:
        m = pat.search(line)
        if not m:
            continue
        count = int(m.group(1))
        if count > 0:
            count_to_elapsed[count] = parse_duration(m.group(3))

    times = []
    prev = 0.0
    for count in sorted(count_to_elapsed):
        elapsed = count_to_elapsed[count]
        times.append(max(0.0, elapsed - prev))
        prev = elapsed
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--answers", required=True, type=Path)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out-prefix", required=True, type=Path)
    ap.add_argument("--input-price-per-mtok", type=float, default=0.0)
    ap.add_argument("--output-price-per-mtok", type=float, default=0.0)
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    answers = load_answers(args.answers)
    results_obj = json.loads(args.results.read_text(encoding="utf-8"))
    results = results_obj["results"]
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    episode_runtime = parse_generation_times(log_text)
    judge_times = parse_judge_times(log_text)

    rows = []
    for idx, item in enumerate(results):
        eid = int(item["episode_id"])
        ep = dataset[eid]
        ep_answers = answers.get(eid, {})
        answer_list = ep_answers.get("answer_list", [])
        qa_pairs = ep.get("qa_pairs", [])
        qa_index = sum(1 for prev in results[:idx] if int(prev["episode_id"]) == eid)
        predicted = item.get("predicted_answer", "")
        question = item.get("question", "")
        golden = item.get("golden_answer", "")
        reasoning_chunks = split_reasoning_trace(ep_answers.get("reasoning_trace", ""), len(answer_list) or len(qa_pairs))
        reasoning = reasoning_chunks[qa_index] if qa_index < len(reasoning_chunks) else ""

        gen_ep_sec = episode_runtime.get(eid)
        gen_q_sec = (gen_ep_sec / max(1, len(qa_pairs))) if gen_ep_sec is not None else None
        judge_sec = judge_times[idx] if idx < len(judge_times) else None

        approx_context_tokens = approx_tokens(reasoning)
        approx_output_tokens = approx_tokens(predicted)
        approx_input_cost = approx_context_tokens / 1_000_000 * args.input_price_per_mtok
        approx_output_cost = approx_output_tokens / 1_000_000 * args.output_price_per_mtok

        rows.append(
            {
                "method": args.method,
                "episode_id": eid,
                "qa_index": qa_index,
                "task_type": item.get("task_type"),
                "domain": item.get("domain"),
                "qa_type": item.get("qa_type"),
                "score": item.get("score"),
                "f1": item.get("f1"),
                "episode_total_tokens_dataset": ep.get("total_tokens"),
                "question_tokens_est": approx_tokens(question),
                "golden_answer_tokens_est": approx_tokens(golden),
                "predicted_answer_tokens_est": approx_output_tokens,
                "reasoning_context_tokens_est": approx_context_tokens,
                "generation_episode_seconds": gen_ep_sec,
                "generation_seconds_allocated_per_question": gen_q_sec,
                "judge_seconds": judge_sec,
                "approx_answer_call_cost_usd": approx_input_cost + approx_output_cost,
            }
        )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_prefix.with_suffix(".csv")
    json_path = args.out_prefix.with_suffix(".summary.json")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": args.method,
        "questions": len(rows),
        "episodes": len({r["episode_id"] for r in rows}),
        "accuracy": results_obj["overall"].get("accuracy"),
        "f1": results_obj["overall"].get("f1"),
        "generation_total_seconds_from_log": sum(v for v in episode_runtime.values()),
        "judge_total_seconds_from_log": sum(t for t in judge_times),
        "avg_generation_seconds_per_question_allocated": mean(
            r["generation_seconds_allocated_per_question"] for r in rows
            if r["generation_seconds_allocated_per_question"] is not None
        ),
        "avg_judge_seconds_per_question": mean(t for t in judge_times) if judge_times else None,
        "sum_episode_total_tokens_dataset_over_questions": sum(r["episode_total_tokens_dataset"] for r in rows),
        "sum_reasoning_context_tokens_est": sum(r["reasoning_context_tokens_est"] for r in rows),
        "sum_predicted_answer_tokens_est": sum(r["predicted_answer_tokens_est"] for r in rows),
        "approx_answer_call_cost_usd": sum(r["approx_answer_call_cost_usd"] for r in rows),
        "warning": (
            "Token/cost fields are estimates from saved artifacts, not exact provider usage. "
            "Memory construction, retrieval subcalls, retries, and judge prompts are not fully recoverable "
            "unless the harness logs API usage during the run."
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
