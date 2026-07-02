#!/usr/bin/env python3
"""Minimal Context Squeeze prototype.

This is intentionally small and synthetic. It tests the core hypothesis:
long-horizon agents can fail not only because they forget, but because stale
or unrelated context remains visible. The simulator is a deliberately simple
contamination-sensitive reader, not a real LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable


TOPICS = [
    "parser", "cache", "router", "scheduler", "logger", "metrics", "auth",
    "database", "renderer", "optimizer", "loader", "serializer", "indexer",
    "notifier", "validator", "compiler", "runtime", "queue", "planner", "search",
]

BAD_ROOTS = [
    "off_by_one", "missing_null_check", "wrong_cache_key", "stale_config",
    "race_condition", "bad_import", "wrong_schema", "timeout", "dtype_mismatch",
    "path_resolution", "permission_error", "incorrect_regex", "overflow",
    "bad_retry", "duplicate_state", "token_budget", "wrong_timezone",
]

GOOD_ROOTS = [
    "boundary_guard", "cache_invalidation", "schema_migration", "async_lock",
    "import_alias", "config_reload", "stable_sort", "batched_query",
    "retry_backoff", "timezone_normalization", "path_canonicalization",
    "dtype_cast", "state_dedup", "context_hiding", "index_refresh",
]


@dataclass
class Event:
    tid: int
    kind: str
    text: str
    active: bool = True
    status: str = "active"  # active, resolved, invalidated, hidden, pinned
    tags: list[str] = field(default_factory=list)

    def render(self) -> str:
        return f"[T{self.tid:02d}|{self.kind}|{self.status}] {self.text}"


@dataclass
class Task:
    tid: int
    topic: str
    stale_root: str
    correct_root: str
    distractor_root: str

    def events(self) -> list[Event]:
        return [
            Event(
                self.tid,
                "goal",
                f"Investigate the {self.topic} component and produce the final root cause.",
                tags=[self.topic],
            ),
            Event(
                self.tid,
                "hypothesis",
                f"Initial hypothesis: the root cause is {self.stale_root}.",
                status="active",
                tags=[self.topic, self.stale_root],
            ),
            Event(
                self.tid,
                "tool_log",
                f"Long debug trace for {self.topic}: many calls point near {self.distractor_root}, but this is noisy intermediate evidence.",
                tags=[self.topic, self.distractor_root],
            ),
            Event(
                self.tid,
                "invalidation",
                f"Correction: {self.stale_root} is a false lead for {self.topic}; do not use it as the final cause.",
                status="active",
                tags=[self.topic, self.stale_root],
            ),
            Event(
                self.tid,
                "decision",
                f"Current valid decision: the final root cause for {self.topic} is {self.correct_root}.",
                status="active",
                tags=[self.topic, self.correct_root],
            ),
            Event(
                self.tid,
                "open_loop",
                f"Open loop: final answer for {self.topic} must mention {self.correct_root} and avoid {self.stale_root}.",
                status="pinned",
                tags=[self.topic, self.correct_root, self.stale_root],
            ),
        ]


@dataclass
class EvalResult:
    method: str
    tid: int
    predicted: str
    correct: str
    success: bool
    stale_used: bool
    active_tokens: int
    recall_used: bool = False
    living_state_ok: bool = False


def token_count(text: str) -> int:
    return len(text.split())


def build_tasks(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    for tid in range(n):
        topic = TOPICS[tid % len(TOPICS)]
        stale, correct, distractor = rng.sample(BAD_ROOTS + GOOD_ROOTS, 3)
        # Keep stale and correct from different pools when possible.
        stale = rng.choice(BAD_ROOTS)
        correct = rng.choice(GOOD_ROOTS)
        while correct == stale:
            correct = rng.choice(GOOD_ROOTS)
        tasks.append(Task(tid, topic, stale, correct, distractor))
    return tasks


class ContextStrategy:
    name = "base"

    def __init__(self, budget_tokens: int = 900, summary_every: int = 5):
        self.budget_tokens = budget_tokens
        self.summary_every = summary_every
        self.visible: list[Event] = []
        self.hidden: list[Event] = []
        self.living_state: dict[int, dict[str, str]] = {}
        self.recall_count = 0

    def reset_for_task(self):
        pass

    def ingest_task(self, task: Task):
        for event in task.events():
            self.add_event(event)

    def add_event(self, event: Event):
        self.visible.append(event)
        self.manage(event)

    def manage(self, event: Event):
        self.trim_to_budget()

    def trim_to_budget(self):
        while self.visible_tokens() > self.budget_tokens and self.visible:
            self.hidden.append(self.visible.pop(0))

    def visible_tokens(self) -> int:
        return token_count(self.render_visible())

    def render_visible(self) -> str:
        return "\n".join(e.render() for e in self.visible)

    def render_living_state(self) -> str:
        if not self.living_state:
            return ""
        lines = ["# Living State"]
        for tid, state in sorted(self.living_state.items()):
            lines.append(
                f"T{tid:02d}: topic={state.get('topic')} current={state.get('current')} invalidated={state.get('invalidated')} open={state.get('open')}"
            )
        return "\n".join(lines)

    def answer(self, task: Task) -> tuple[str, bool]:
        pred, _stale = contamination_sensitive_reader(task, self.render_visible(), self.render_living_state())
        return pred, False


class ResetStrategy(ContextStrategy):
    name = "reset_per_task"

    def ingest_task(self, task: Task):
        self.visible = []
        self.hidden = []
        self.living_state = {}
        super().ingest_task(task)


class FullSqueezeStrategy(ContextStrategy):
    name = "full_squeeze"

    def trim_to_budget(self):
        # Simulates a user/agent that naively keeps everything visible. We still
        # report the bloated token count rather than enforcing a hard budget.
        return


class SlidingWindowStrategy(ContextStrategy):
    name = "sliding_window"

    def __init__(self, window_events: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.window_events = window_events

    def trim_to_budget(self):
        while len(self.visible) > self.window_events:
            self.hidden.append(self.visible.pop(0))


class PeriodicSummaryStrategy(ContextStrategy):
    name = "periodic_summary"

    def __init__(self, summary_every: int = 3, **kwargs):
        super().__init__(summary_every=summary_every, **kwargs)
        self.tasks_seen = 0
        self.summary_events: list[Event] = []

    def ingest_task(self, task: Task):
        super().ingest_task(task)
        self.tasks_seen += 1
        if self.tasks_seen % self.summary_every == 0:
            self.summarize_old_context()

    def summarize_old_context(self):
        # Naive summary: keeps both invalidated and valid info in one compressed
        # event. This intentionally models summary drift / mixing.
        old = self.visible[:-8]
        if not old:
            return
        text = "Summary of previous work: " + "; ".join(e.text for e in old if e.kind in {"hypothesis", "decision", "invalidation"})
        self.hidden.extend(old)
        self.visible = [Event(-1, "summary", text[:1200], status="active")] + self.visible[-8:]


class LivingStateHideStrategy(ContextStrategy):
    name = "living_state_hide"

    def manage(self, event: Event):
        tid = event.tid
        if tid >= 0:
            self.living_state.setdefault(tid, {})
            if event.kind == "goal":
                self.living_state[tid]["topic"] = event.tags[0]
            elif event.kind == "invalidation":
                # Extract stale root from tag list.
                invalidated = event.tags[-1]
                self.living_state[tid]["invalidated"] = invalidated
                self._hide_matching(tid, "hypothesis")
                self._hide_matching(tid, "tool_log")
            elif event.kind == "decision":
                self.living_state[tid]["current"] = event.tags[-1]
                # Current decision supersedes noisy prior context.
                self._hide_matching(tid, "tool_log")
            elif event.kind == "open_loop":
                self.living_state[tid]["open"] = event.text
        self.trim_to_budget()

    def _hide_matching(self, tid: int, kind: str):
        kept: list[Event] = []
        for e in self.visible:
            if e.tid == tid and e.kind == kind:
                e.status = "hidden"
                self.hidden.append(e)
            else:
                kept.append(e)
        self.visible = kept

    def trim_to_budget(self):
        # Prefer hiding old non-pinned events; preserve current task pins/decisions.
        while self.visible_tokens() > self.budget_tokens and self.visible:
            idx = next((i for i, e in enumerate(self.visible) if e.status != "pinned" and e.kind not in {"decision", "open_loop"}), 0)
            event = self.visible.pop(idx)
            event.status = "hidden"
            self.hidden.append(event)

    def answer(self, task: Task) -> tuple[str, bool]:
        # If current task has living state, use it. If not, recall hidden evidence.
        state = self.living_state.get(task.tid, {})
        if "current" in state:
            return state["current"], False
        self.recall_count += 1
        recalled = [e for e in self.hidden if e.tid == task.tid and e.kind == "decision"]
        if recalled:
            return recalled[-1].tags[-1], True
        pred, _stale = contamination_sensitive_reader(task, self.render_visible(), self.render_living_state())
        return pred, False


def contamination_sensitive_reader(task: Task, visible: str, living_state: str = "") -> tuple[str, bool]:
    """A small deterministic model of context contamination.

    Priority:
    1. If living state explicitly has current answer for the task, use it.
    2. Otherwise, if visible context contains both a current decision and visible
       stale hypothesis for the same task, choose stale to simulate over-exposure.
    3. If current decision exists without visible stale hypothesis, choose current.
    4. If neither exists, may pick a same-topic stale or prior global decision.
    """
    current_marker = f"T{task.tid:02d}:"
    if current_marker in living_state and f"current={task.correct_root}" in living_state:
        return task.correct_root, False

    lines = visible.splitlines()
    task_lines = [line for line in lines if f"[T{task.tid:02d}|" in line]
    has_current = any(task.correct_root in line and "decision" in line for line in task_lines)
    visible_stale = any(task.stale_root in line and "hypothesis" in line and "hidden" not in line for line in task_lines)
    visible_invalid = any(task.stale_root in line and "invalidation" in line for line in task_lines)

    if has_current and visible_stale:
        # With a short, clean per-task context, the correction is easy to follow.
        # With long accumulated context or summaries, the stale hypothesis remains
        # salient and can still pull the model toward the false lead.
        if not visible_invalid or len(lines) > 16 or any("summary" in line for line in lines):
            return task.stale_root, True
    if has_current:
        return task.correct_root, False

    # Cross-task contamination: choose the last visible decision from another task.
    for line in reversed(lines):
        if "decision" in line and "final root cause" in line:
            for root in GOOD_ROOTS:
                if root in line:
                    return root, root != task.correct_root

    # Guess stale if only stale is visible.
    if visible_stale:
        return task.stale_root, True
    return "unknown", True


def run_eval(n_tasks: int, seed: int, budget_tokens: int) -> list[EvalResult]:
    tasks = build_tasks(n_tasks, seed)
    strategies: list[ContextStrategy] = [
        ResetStrategy(budget_tokens=budget_tokens),
        FullSqueezeStrategy(budget_tokens=budget_tokens),
        SlidingWindowStrategy(window_events=10, budget_tokens=budget_tokens),
        PeriodicSummaryStrategy(summary_every=3, budget_tokens=budget_tokens),
        LivingStateHideStrategy(budget_tokens=budget_tokens),
    ]
    results: list[EvalResult] = []
    for strat in strategies:
        if isinstance(strat, ResetStrategy):
            # Oracle-ish clean evaluation: each task gets its own fresh session.
            for task in tasks:
                strat.ingest_task(task)
                pred, recall = strat.answer(task)
                success = pred == task.correct_root
                results.append(
                    EvalResult(
                        method=strat.name,
                        tid=task.tid,
                        predicted=pred,
                        correct=task.correct_root,
                        success=success,
                        stale_used=pred == task.stale_root,
                        active_tokens=strat.visible_tokens() + token_count(strat.render_living_state()),
                        recall_used=recall,
                        living_state_ok=(strat.living_state.get(task.tid, {}).get("current") == task.correct_root),
                    )
                )
            continue

        # Squeeze setting: all tasks are processed in one continuous session.
        # We then audit every task at the end, measuring whether the method can
        # recover current valid state after many unrelated/stale traces.
        for task in tasks:
            strat.ingest_task(task)
        for task in tasks:
            pred, recall = strat.answer(task)
            success = pred == task.correct_root
            results.append(
                EvalResult(
                    method=strat.name,
                    tid=task.tid,
                    predicted=pred,
                    correct=task.correct_root,
                    success=success,
                    stale_used=pred == task.stale_root,
                    active_tokens=strat.visible_tokens() + token_count(strat.render_living_state()),
                    recall_used=recall,
                    living_state_ok=(strat.living_state.get(task.tid, {}).get("current") == task.correct_root),
                )
            )
    return results


def summarize(results: Iterable[EvalResult]) -> list[dict[str, object]]:
    rows = []
    by_method: dict[str, list[EvalResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)
    for method, vals in by_method.items():
        rows.append(
            {
                "method": method,
                "n": len(vals),
                "success_rate": round(mean(v.success for v in vals), 3),
                "stale_usage_rate": round(mean(v.stale_used for v in vals), 3),
                "avg_active_tokens": round(mean(v.active_tokens for v in vals), 1),
                "recall_rate": round(mean(v.recall_used for v in vals), 3),
                "living_state_acc": round(mean(v.living_state_ok for v in vals), 3),
            }
        )
    return rows


def write_outputs(results: list[EvalResult], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize(results)
    with (out_dir / "prototype_summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=2)
    with (out_dir / "prototype_details.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r.__dict__)
    with (out_dir / "prototype_summary.md").open("w") as f:
        f.write("# Minimal Context Squeeze Prototype Results\n\n")
        f.write("| Method | N | Success ↑ | Stale Usage ↓ | Avg Active Tokens ↓ | Recall | Living State Acc |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['method']} | {row['n']} | {row['success_rate']} | {row['stale_usage_rate']} | {row['avg_active_tokens']} | {row['recall_rate']} | {row['living_state_acc']} |\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tasks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget-tokens", type=int, default=900)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()

    results = run_eval(args.n_tasks, args.seed, args.budget_tokens)
    write_outputs(results, args.out_dir)
    for row in summarize(results):
        print(row)


if __name__ == "__main__":
    main()
