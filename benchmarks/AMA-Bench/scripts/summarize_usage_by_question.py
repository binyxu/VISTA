#!/usr/bin/env python3
"""Aggregate usage_*.jsonl into per-question token/time rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def num(x: Any) -> float:
    return float(x or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage-log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = defaultdict(lambda: {
        "api_calls": 0,
        "elapsed_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_est": 0,
        "completion_tokens_est": 0,
        "errors": 0,
    })

    with args.usage_log.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (
                rec.get("episode_id"),
                rec.get("qa_index"),
                rec.get("stage") or "unknown",
                rec.get("role") or "unknown",
            )
            row = rows[key]
            row["episode_id"], row["qa_index"], row["stage"], row["role"] = key
            row["api_calls"] += 1
            row["elapsed_seconds"] += num(rec.get("elapsed_seconds"))
            row["prompt_tokens"] += int(num(rec.get("prompt_tokens")))
            row["completion_tokens"] += int(num(rec.get("completion_tokens")))
            row["total_tokens"] += int(num(rec.get("total_tokens")))
            row["prompt_tokens_est"] += int(num(rec.get("prompt_tokens_est")))
            row["completion_tokens_est"] += int(num(rec.get("completion_tokens_est")))
            row["errors"] += 1 if rec.get("status") not in ("ok", None) else 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id",
        "qa_index",
        "stage",
        "role",
        "api_calls",
        "elapsed_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_est",
        "completion_tokens_est",
        "errors",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: (
            str(r["episode_id"]),
            -1 if r["qa_index"] is None else int(r["qa_index"]),
            str(r["stage"]),
            str(r["role"]),
        )):
            writer.writerow(row)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
