#!/usr/bin/env python3
"""Generate SWE-Bench predictions with a strict long-context dashboard prompt."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"


@dataclass
class ContextBlock:
    name: str
    kind: str
    content: str
    active: bool = True

    @property
    def chars(self) -> int:
        return len(self.content)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def load_records(dataset_name: str, split: str, instance_ids: set[str] | None) -> list[dict[str, Any]]:
    path = Path(dataset_name)
    if dataset_name.endswith(".jsonl"):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif dataset_name.endswith(".json"):
        loaded = json.loads(path.read_text())
        records = list(loaded.values()) if isinstance(loaded, dict) else loaded
    else:
        from datasets import load_dataset, load_from_disk

        if path.exists() and (path / "dataset_info.json").exists():
            records = list(load_from_disk(str(path)))
        elif path.exists() and (path / split / "dataset_info.json").exists():
            records = list(load_from_disk(str(path / split)))
        else:
            if dataset_name.lower() in {"swe-bench", "swebench", "swe_bench"}:
                dataset_name = "SWE-bench/SWE-bench"
            elif dataset_name.lower() in {
                "swe-bench-lite",
                "swebench-lite",
                "swe_bench_lite",
                "swe-bench_lite",
                "lite",
            }:
                dataset_name = "SWE-bench/SWE-bench_Lite"
            records = list(load_dataset(dataset_name, split=split))

    if instance_ids:
        records = [dict(r) for r in records if r[KEY_INSTANCE_ID] in instance_ids]
    else:
        records = [dict(r) for r in records]
    return records


def read_existing_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()
    ids: set[str] = set()
    for line in output_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line)[KEY_INSTANCE_ID])
        except Exception:
            continue
    return ids


def build_base_context(instance: dict[str, Any]) -> str:
    if instance.get("text"):
        return str(instance["text"])

    parts = [
        "You will be given a GitHub issue from a real repository.",
        "Generate a single unified diff patch that can be applied with git apply.",
        "",
        f"Repository: {instance.get('repo', '')}",
        f"Base commit: {instance.get('base_commit', '')}",
        "",
        "<issue>",
        str(instance.get("problem_statement", "")),
        "</issue>",
    ]
    if instance.get("hints_text"):
        parts.extend(["", "<hints>", str(instance["hints_text"]), "</hints>"])
    if instance.get("FAIL_TO_PASS"):
        parts.extend(["", "<fail_to_pass_tests>", str(instance["FAIL_TO_PASS"]), "</fail_to_pass_tests>"])
    return "\n".join(parts)


def split_context(text: str, target_chars: int) -> list[ContextBlock]:
    if len(text) <= target_chars:
        return [ContextBlock("problem_context", "active_context", text)]

    blocks: list[ContextBlock] = []
    section_pattern = re.compile(r"(?m)^(\[start of .*?\]|<issue>|<code>|diff --git )")
    starts = [m.start() for m in section_pattern.finditer(text)]
    if not starts:
        starts = list(range(0, len(text), target_chars))
    starts = sorted(set([0] + starts + [len(text)]))

    section_idx = 0
    for left, right in zip(starts, starts[1:]):
        chunk = text[left:right].strip()
        if not chunk:
            continue
        if len(chunk) <= target_chars:
            section_idx += 1
            blocks.append(ContextBlock(f"context_{section_idx:03d}", "active_context", chunk))
            continue
        for offset in range(0, len(chunk), target_chars):
            section_idx += 1
            blocks.append(
                ContextBlock(
                    f"context_{section_idx:03d}",
                    "active_context",
                    chunk[offset : offset + target_chars],
                )
            )
    return blocks


def enforce_budget(blocks: list[ContextBlock], max_context_size: int, reserve_output_tokens: int) -> None:
    budget_chars = max(4000, (max_context_size - reserve_output_tokens) * 4)
    dashboard_allowance = 6000
    active_chars = sum(b.chars for b in blocks if b.active)
    while active_chars + dashboard_allowance > budget_chars and sum(b.active for b in blocks) > 1:
        largest = max((b for b in blocks if b.active), key=lambda b: b.chars)
        largest.active = False
        largest.kind = "deferred_context"
        active_chars = sum(b.chars for b in blocks if b.active)


def render_dashboard(blocks: list[ContextBlock], max_context_size: int, preview_chars: int) -> str:
    active = [b for b in blocks if b.active]
    deferred = [b for b in blocks if not b.active]
    lines = [
        "Context Workspace Dashboard",
        f"strict_long_context: enabled",
        f"better_dashboard: enabled",
        f"max_context_size_tokens: {max_context_size}",
        f"active_blocks: {len(active)}",
        f"deferred_blocks: {len(deferred)}",
        f"active_estimated_tokens: {sum(approx_tokens(b.content) for b in active)}",
        "",
        "Active blocks:",
    ]
    for block in active:
        lines.append(f"- {block.name}: {approx_tokens(block.content)} tok, {block.chars} chars")
    if deferred:
        lines.extend(["", "Deferred blocks:"])
        for block in deferred:
            preview = " ".join(block.content[:preview_chars].split())
            lines.append(f"- {block.name}: {approx_tokens(block.content)} tok, preview={preview!r}")
    return "\n".join(lines)


def build_messages(
    instance: dict[str, Any],
    config: dict[str, Any],
    max_context_size: int,
) -> list[dict[str, str]]:
    reserve_output_tokens = int(config.get("reserve_output_tokens", 4096))
    dashboard_cfg = config.get("dashboard", {})
    preview_chars = int(dashboard_cfg.get("max_block_preview_chars", 280))
    base_context = build_base_context(instance)
    target_chars = max(8000, (max_context_size - reserve_output_tokens) * 3)
    blocks = split_context(base_context, target_chars)

    dashboard_enabled = bool(config.get("dashboard", {}).get("enabled", True))
    strict_long_context = bool(config.get("strict_long_context", True)) or os.getenv("SM_STRICT_LONG_CONTEXT") == "1"
    if strict_long_context:
        enforce_budget(blocks, max_context_size, reserve_output_tokens)

    active_context = "\n\n".join(block.content for block in blocks if block.active)
    prompt_parts = []
    if dashboard_enabled:
        dashboard = render_dashboard(blocks, max_context_size, preview_chars)
        prompt_parts.extend([
            "<context_workspace_status>",
            dashboard,
            "</context_workspace_status>",
            "",
        ])
    prompt_parts.extend([
        active_context,
        "",
        "Return only a unified diff patch. Do not include prose outside the patch.",
    ])
    user_prompt = "\n\n".join(prompt_parts)
    if dashboard_enabled:
        system_prompt = (
            "You are a software engineering agent solving SWE-Bench tasks. "
            "Use the context workspace dashboard to track available and deferred context. "
            "Respect the strict long-context budget and produce a minimal correct patch."
        )
    else:
        system_prompt = (
            "You are a software engineering agent solving SWE-Bench tasks. "
            "Think step by step internally, then produce a minimal correct patch."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def make_client() -> OpenAI:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LOCA_OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LOCA_OPENAI_BASE_URL")
    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "180"))
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or LOCA_OPENAI_API_KEY before running inference.")
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def extract_diff(response: str | None) -> str | None:
    if response is None:
        return None
    diff_matches: list[str] = []
    other_matches: list[str] = []
    tag_pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in tag_pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    fence_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in fence_pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0].strip()
    if other_matches:
        return other_matches[0].strip()
    response = response.split("</s>")[0].strip()
    if response.startswith("```diff\n") or response.startswith("```patch\n"):
        response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response[:-3]
    return response.strip()


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    generation: dict[str, Any],
    retries: int,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(generation.get("temperature", 0.0)),
                top_p=float(generation.get("top_p", 1.0)),
                max_tokens=int(generation.get("max_output_tokens", 4096)),
            )
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"Model call failed after {retries} attempts: {last_error}")


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def parse_instance_ids(values: Iterable[str] | None) -> set[str] | None:
    if not values:
        return None
    ids: set[str] = set()
    for value in values:
        p = Path(value)
        if p.exists():
            ids.update(line.strip() for line in p.read_text().splitlines() if line.strip())
        else:
            ids.add(value)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--config", default=str(Path(__file__).with_name("configs") / "strict_lc_better_dashboard.json"))
    parser.add_argument("--max_context_size", type=int, default=None)
    parser.add_argument("--instance_ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    max_context_size = args.max_context_size or int(config.get("max_context_size", 128000))
    generation = dict(config.get("generation", {}))
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_file.exists():
        output_file.unlink()

    instance_ids = parse_instance_ids(args.instance_ids)
    records = load_records(args.dataset_name, args.split, instance_ids)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print("No records selected; nothing to do.")
        return

    existing_ids = read_existing_ids(output_file)
    client = make_client()

    for idx, instance in enumerate(records, start=1):
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id in existing_ids:
            print(f"[{idx}/{len(records)}] skip existing {instance_id}")
            continue
        print(f"[{idx}/{len(records)}] generating {instance_id}")
        messages = build_messages(instance, config, max_context_size)
        response = call_model(client, args.model, messages, generation, args.retries)
        full_output = response.choices[0].message.content or ""
        prediction = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: f"{args.model}__self_managed_strict_lc_better_dashboard",
            KEY_PREDICTION: extract_diff(full_output),
            "full_output": full_output,
            "method_config": {
                "strict_long_context": bool(config.get("strict_long_context", True)),
                "better_dashboard": bool(config.get("better_dashboard", True)),
                "max_context_size": max_context_size,
            },
        }
        write_jsonl(output_file, prediction)


if __name__ == "__main__":
    main()
