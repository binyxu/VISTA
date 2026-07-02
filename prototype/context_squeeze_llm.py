#!/usr/bin/env python3
"""Run the Context Squeeze prototype with a real chat-completions LLM.

Credentials are intentionally read from environment variables and are not stored
in this file.

Required env vars:
  LLM_API_KEY
  LLM_BASE_URL  (full /v1/chat/completions endpoint)

Example:
  export LLM_API_KEY='...'
  export LLM_BASE_URL='https://api.example.com/v1/chat/completions'
  python3 prototype/context_squeeze_llm.py --n-tasks 8 --model gpt-5.1-mini
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from context_squeeze import (
    EvalResult,
    FullSqueezeStrategy,
    LivingStateHideStrategy,
    PeriodicSummaryStrategy,
    ResetStrategy,
    SlidingWindowStrategy,
    Task,
    build_tasks,
    token_count,
)


@dataclass
class LLMRunResult(EvalResult):
    raw_answer: str = ""


class ChatClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 120, retries: int = 3):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 64) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(self.base_url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8")
                    last_error = RuntimeError(f"HTTP {exc.code}: {body}")
                except Exception:
                    last_error = exc
                time.sleep(1 + attempt)
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"LLM request failed after {self.retries} attempts: {last_error}")


def make_prompt(task: Task, visible_context: str, living_state: str) -> list[dict[str, str]]:
    system = (
        "You are evaluating a long-running coding-agent session. "
        "Answer only with JSON. Do not include markdown. "
        "The session may contain stale hypotheses, false leads, invalidated notes, and unrelated tasks. "
        "Your job is to identify the CURRENT valid root cause for the requested task."
    )
    user = f"""
Requested task id: T{task.tid:02d}
Requested component/topic: {task.topic}

Living state, if available:
{living_state or '(none)'}

Visible session context:
{visible_context or '(empty)'}

Return JSON exactly like this:
{{"root_cause": "...", "used_stale_or_unrelated_context": false, "evidence": "one short phrase"}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_root(raw: str) -> tuple[str, bool]:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            root = str(obj.get("root_cause", "")).strip()
            stale = bool(obj.get("used_stale_or_unrelated_context", False))
            return root, stale
        except json.JSONDecodeError:
            pass
    # Fallback: first simple token-like phrase.
    cleaned = re.sub(r"[^A-Za-z0-9_\- ]", " ", text).strip()
    return cleaned.split()[0] if cleaned else "", False


def evaluate_answer(task: Task, pred: str, stale_flag: bool) -> tuple[bool, bool]:
    pred_norm = pred.strip().lower()
    success = task.correct_root.lower() in pred_norm
    stale_used = stale_flag or task.stale_root.lower() in pred_norm
    if success:
        stale_used = False
    return success, stale_used


def answer_with_llm(client: ChatClient, task: Task, visible: str, living_state: str) -> tuple[str, bool, str]:
    raw = client.chat(make_prompt(task, visible, living_state), max_tokens=96)
    pred, stale_flag = parse_root(raw)
    return pred, stale_flag, raw


def run_llm_eval(n_tasks: int, seed: int, budget_tokens: int, client: ChatClient) -> list[LLMRunResult]:
    tasks = build_tasks(n_tasks, seed)
    strategies = [
        ResetStrategy(budget_tokens=budget_tokens),
        FullSqueezeStrategy(budget_tokens=budget_tokens),
        SlidingWindowStrategy(window_events=10, budget_tokens=budget_tokens),
        PeriodicSummaryStrategy(summary_every=3, budget_tokens=budget_tokens),
        LivingStateHideStrategy(budget_tokens=budget_tokens),
    ]

    results: list[LLMRunResult] = []
    for strat in strategies:
        if isinstance(strat, ResetStrategy):
            for task in tasks:
                strat.ingest_task(task)
                pred, stale_flag, raw = answer_with_llm(client, task, strat.render_visible(), strat.render_living_state())
                success, stale_used = evaluate_answer(task, pred, stale_flag)
                results.append(
                    LLMRunResult(
                        method=strat.name,
                        tid=task.tid,
                        predicted=pred,
                        correct=task.correct_root,
                        success=success,
                        stale_used=stale_used,
                        active_tokens=strat.visible_tokens() + token_count(strat.render_living_state()),
                        recall_used=False,
                        living_state_ok=(strat.living_state.get(task.tid, {}).get("current") == task.correct_root),
                        raw_answer=raw,
                    )
                )
            continue

        for task in tasks:
            strat.ingest_task(task)
        for task in tasks:
            pred, stale_flag, raw = answer_with_llm(client, task, strat.render_visible(), strat.render_living_state())
            success, stale_used = evaluate_answer(task, pred, stale_flag)
            results.append(
                LLMRunResult(
                    method=strat.name,
                    tid=task.tid,
                    predicted=pred,
                    correct=task.correct_root,
                    success=success,
                    stale_used=stale_used,
                    active_tokens=strat.visible_tokens() + token_count(strat.render_living_state()),
                    recall_used=False,
                    living_state_ok=(strat.living_state.get(task.tid, {}).get("current") == task.correct_root),
                    raw_answer=raw,
                )
            )
    return results


def summarize(results: Iterable[LLMRunResult]) -> list[dict[str, object]]:
    by_method: dict[str, list[LLMRunResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)
    rows = []
    for method, vals in by_method.items():
        rows.append(
            {
                "method": method,
                "n": len(vals),
                "success_rate": round(mean(v.success for v in vals), 3),
                "stale_usage_rate": round(mean(v.stale_used for v in vals), 3),
                "avg_active_tokens": round(mean(v.active_tokens for v in vals), 1),
                "living_state_acc": round(mean(v.living_state_ok for v in vals), 3),
            }
        )
    return rows


def write_outputs(results: list[LLMRunResult], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(results)
    with (out_dir / "llm_prototype_summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=2)
    with (out_dir / "llm_prototype_details.csv").open("w", newline="") as f:
        fieldnames = list(asdict(results[0]).keys()) if results else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    with (out_dir / "llm_prototype_summary.md").open("w") as f:
        f.write("# Real LLM Context Squeeze Prototype Results\n\n")
        f.write("| Method | N | Success ↑ | Stale Usage ↓ | Avg Active Tokens ↓ | Living State Acc |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['method']} | {row['n']} | {row['success_rate']} | {row['stale_usage_rate']} | {row['avg_active_tokens']} | {row['living_state_acc']} |\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget-tokens", type=int, default=900)
    parser.add_argument("--model", default="gpt-5.1-mini")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("Please set LLM_API_KEY and LLM_BASE_URL environment variables.")

    client = ChatClient(api_key=api_key, base_url=base_url, model=args.model)
    results = run_llm_eval(args.n_tasks, args.seed, args.budget_tokens, client)
    write_outputs(results, args.out_dir)
    for row in summarize(results):
        print(row)


if __name__ == "__main__":
    main()
