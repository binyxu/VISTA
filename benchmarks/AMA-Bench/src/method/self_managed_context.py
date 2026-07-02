"""
Self-managed context workspace method for AMA-Bench.

This method adapts the project prototype's visible-workspace/archive abstraction
to AMA-Bench's two-stage memory interface. It intentionally avoids LLM calls
during construction so it can serve as a low-variance first integration.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.method.base_method import BaseMethod


PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROTOTYPE_DIR = PROJECT_ROOT / "prototype"
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))

from context_workspace import ContextWorkspace  # noqa: E402


_STEP_RE = re.compile(
    r"(?ms)^\s*(?:Step|Turn)\s+(\d+):\s*\n"
    r"Action:\s*(.*?)\n"
    r"Observation:\s*(.*?)(?=^\s*(?:Step|Turn)\s+\d+:\s*\nAction:|\Z)"
)

_IMPORTANT_RE = re.compile(
    r"\b(error|exception|traceback|failed|success|done|final|answer|"
    r"decision|root cause|found|created|updated|deleted|changed|"
    r"key|schema|result|count|total|location|inventory)\b",
    re.IGNORECASE,
)


@dataclass
class SelfManagedMemory:
    workspace: ContextWorkspace
    task: str
    trajectory_text: str
    steps: List[Dict[str, Any]]
    config: Dict[str, Any]
    construction_metrics: Dict[str, Any]


class SelfManagedContextMethod(BaseMethod):
    """
    Context-workspace memory method.

    Construction builds an addressable workspace with explicit visible and
    hidden blocks. Retrieval shows the active workspace plus compact archive
    evidence selected for the current question.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        client: Optional[Any] = None,
        embedding_engine: Optional[Any] = None,
    ):
        config = self._load_config(config_path) if config_path else {}
        self.token_budget = int(config.get("token_budget", 32000))
        self.dashboard_top_k = int(config.get("dashboard_top_k", 8))
        self.archive_top_k = int(config.get("archive_top_k", 5))
        self.archive_max_tokens = int(config.get("archive_max_tokens", 1200))
        self.max_visible_block_tokens = int(config.get("max_visible_block_tokens", 220))
        self.keep_recent_steps = int(config.get("keep_recent_steps", 8))
        self.hide_long_observations = bool(config.get("hide_long_observations", True))
        self.long_observation_tokens = int(config.get("long_observation_tokens", 260))
        self.hide_unimportant_old_steps = bool(config.get("hide_unimportant_old_steps", True))
        self.client = client
        self.embedding_engine = embedding_engine

    def memory_construction(self, traj_text: str, task: str = "") -> SelfManagedMemory:
        start = time.perf_counter()
        workspace = ContextWorkspace(
            token_budget=self.token_budget,
            dashboard_top_k=self.dashboard_top_k,
        )
        episode_id = workspace.start_episode(task or "AMA-Bench trajectory", source="ama_task")
        workspace.blocks[episode_id].status = "pinned"

        steps = self._parse_steps(traj_text)
        if not steps and traj_text.strip():
            steps = [{"turn_idx": 0, "action": "", "observation": traj_text.strip()}]

        step_block_ids: List[str] = []
        for step in steps:
            content = self._format_step(step)
            block_id = workspace.add_react_step(content)
            block = workspace.blocks[block_id]
            block.source = f"turn_{step['turn_idx']}"
            block.summary = workspace.summarizer.summarize_block(
                block,
                task_goal=task,
                max_words=55,
            )
            block.metadata.update(
                {
                    "turn_idx": step["turn_idx"],
                    "action": step.get("action", ""),
                    "important": self._is_important(step),
                }
            )
            step_block_ids.append(block_id)

        self._apply_visibility_policy(workspace, step_block_ids)
        metrics = self._metrics(workspace)
        metrics["construction_latency_ms"] = round((time.perf_counter() - start) * 1000, 3)

        return SelfManagedMemory(
            workspace=workspace,
            task=task,
            trajectory_text=traj_text,
            steps=steps,
            config=self._config_dict(),
            construction_metrics=metrics,
        )

    def memory_retrieve(self, memory: SelfManagedMemory, question: str) -> str:
        if not isinstance(memory, SelfManagedMemory):
            raise ValueError("Memory must be a SelfManagedMemory object")

        start = time.perf_counter()
        workspace = memory.workspace
        archive_matches = workspace.search_archive(question, top_k=self.archive_top_k)
        archive_text, archive_tokens = self._format_archive_matches(workspace, archive_matches)
        metrics = self._metrics(workspace)
        metrics.update(
            {
                "num_recalled_blocks": len(archive_matches),
                "archive_recall_used": bool(archive_matches),
                "archive_tokens": archive_tokens,
                "retrieve_latency_ms": round((time.perf_counter() - start) * 1000, 3),
            }
        )

        visible_context = workspace.assemble_visible_context(
            include_dashboard=True,
            max_block_tokens=self.max_visible_block_tokens,
        )
        metrics_text = "\n".join(f"- {key}: {value}" for key, value in sorted(metrics.items()))

        return (
            f"## Task Description\n{memory.task}\n\n"
            f"## Self-Managed Context Diagnostics\n{metrics_text}\n\n"
            f"## Visible Workspace\n{visible_context}\n\n"
            f"## Retrieved Archive Evidence\n{archive_text or '(no matching archived evidence)'}"
        )

    def _parse_steps(self, traj_text: str) -> List[Dict[str, Any]]:
        steps = []
        for match in _STEP_RE.finditer(traj_text):
            steps.append(
                {
                    "turn_idx": int(match.group(1)),
                    "action": match.group(2).strip(),
                    "observation": match.group(3).strip(),
                }
            )
        return steps

    def _format_step(self, step: Dict[str, Any]) -> str:
        return (
            f"Turn {step['turn_idx']}:\n"
            f"Action: {step.get('action', '')}\n"
            f"Observation: {step.get('observation', '')}"
        )

    def _is_important(self, step: Dict[str, Any]) -> bool:
        text = f"{step.get('action', '')}\n{step.get('observation', '')}"
        return bool(_IMPORTANT_RE.search(text))

    def _apply_visibility_policy(self, workspace: ContextWorkspace, step_block_ids: List[str]) -> None:
        recent = set(step_block_ids[-self.keep_recent_steps :])
        for block_id in step_block_ids:
            block = workspace.blocks[block_id]
            important = bool(block.metadata.get("important"))
            too_long = block.tokens >= self.long_observation_tokens
            old = block_id not in recent

            should_hide = False
            reason = ""
            if self.hide_long_observations and too_long and old:
                should_hide = True
                reason = "long old trajectory detail archived after summarization"
            elif self.hide_unimportant_old_steps and old and not important:
                should_hide = True
                reason = "old non-salient trajectory detail archived"

            if should_hide:
                workspace.hide(block_id, reason)

        # Enforce the global visible-token budget while preserving pinned task and
        # recent turns. Hide oldest non-pinned blocks first.
        while workspace.visible_token_count() > self.token_budget:
            candidates = [
                b for b in workspace.visible_blocks()
                if b.status != "pinned" and b.id not in recent and b.type != "living_state"
            ]
            if not candidates:
                break
            oldest = sorted(candidates, key=lambda b: b.created_at)[0]
            workspace.hide(oldest.id, "visible token budget enforcement")

        workspace._refresh_living_state()

    def _format_archive_matches(
        self,
        workspace: ContextWorkspace,
        matches: List[Any],
    ) -> tuple[str, int]:
        parts: List[str] = []
        used_tokens = 0
        for block in matches:
            excerpt = block.summary or workspace._short(block.content, words=90)
            tokens = workspace.estimate_tokens(excerpt)
            if parts and used_tokens + tokens > self.archive_max_tokens:
                continue
            used_tokens += tokens
            parts.append(
                f"<archived_block id={block.id} type={block.type} source={block.source} "
                f"tokens={block.tokens}>\n{excerpt}\n</archived_block>"
            )
        return "\n\n".join(parts), used_tokens

    def _metrics(self, workspace: ContextWorkspace) -> Dict[str, Any]:
        visible = workspace.visible_blocks()
        hidden = workspace.hidden_blocks()
        living = workspace.blocks.get(workspace.living_state_block_id)
        return {
            "visible_tokens": workspace.visible_token_count(),
            "hidden_tokens": sum(b.tokens for b in hidden),
            "num_visible_blocks": len(visible),
            "num_hidden_blocks": len(hidden),
            "num_pinned_blocks": sum(1 for b in visible if b.status == "pinned"),
            "living_state_tokens": living.tokens if living else 0,
        }

    def _config_dict(self) -> Dict[str, Any]:
        return {
            "token_budget": self.token_budget,
            "dashboard_top_k": self.dashboard_top_k,
            "archive_top_k": self.archive_top_k,
            "archive_max_tokens": self.archive_max_tokens,
            "max_visible_block_tokens": self.max_visible_block_tokens,
            "keep_recent_steps": self.keep_recent_steps,
            "hide_long_observations": self.hide_long_observations,
            "long_observation_tokens": self.long_observation_tokens,
            "hide_unimportant_old_steps": self.hide_unimportant_old_steps,
        }
