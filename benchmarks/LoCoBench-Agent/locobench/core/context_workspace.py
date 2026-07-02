#!/usr/bin/env python3
"""Agentic Context Workspace.

This module implements the core data structure for agent-managed context:

- Visible Workspace: blocks currently shown to the agent.
- Archive: full-fidelity hidden blocks, still searchable/recoverable.
- Dashboard: compact, structured view of context composition and token usage.
- Actions: ABSTRACT, HIDE, ASK_ARCHIVE, SHOW_BLOCK, RESTORE and macro ABSTRACT_AND_HIDE.

The implementation is intentionally model-agnostic. LLMs may propose actions as
JSON, while this module validates and applies them deterministically.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal


BlockStatus = Literal["visible", "hidden", "pinned", "dropped"]
BlockType = Literal[
    "session",
    "episode",
    "react_step",
    "user_message",
    "assistant_message",
    "tool_call",
    "tool_result",
    "file_read",
    "file_write",
    "bash_output",
    "search_result",
    "summary",
    "living_state",
]


class ContextActionError(ValueError):
    """Raised when a context action is invalid."""


class ActionType(str, Enum):
    ABSTRACT = "ABSTRACT"
    HIDE = "HIDE"
    ASK_ARCHIVE = "ASK_ARCHIVE"
    SHOW_BLOCK = "SHOW_BLOCK"
    RESTORE = "RESTORE"
    RECALL = "RECALL"  # Backward-compatible alias for ASK_ARCHIVE/SHOW_BLOCK.
    ABSTRACT_AND_HIDE = "ABSTRACT_AND_HIDE"


@dataclass
class ContextBlock:
    """A typed, addressable context unit."""

    id: str
    type: BlockType
    content: str
    source: str = ""
    status: BlockStatus = "visible"
    parent_id: str | None = None
    summary: str = ""
    tokens: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def short_label(self) -> str:
        src = f" {self.source}" if self.source else ""
        return f"{self.id} {self.type}{src}"

    def to_dashboard_row(self, include_summary: bool = True) -> str:
        summary = f" — {self.summary}" if include_summary and self.summary else ""
        return f"- {self.id}: {self.type}, {self.tokens} tok, {self.status}, source={self.source or 'n/a'}{summary}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextAction:
    action: ActionType
    target: str | None = None
    content: str | None = None
    query: str | None = None
    reason: str = ""
    max_tokens: int | None = None

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "ContextAction":
        if "action" not in obj:
            raise ContextActionError("Action object missing 'action'.")
        try:
            action = ActionType(str(obj["action"]).upper())
        except ValueError as exc:
            raise ContextActionError(f"Unknown action: {obj.get('action')}") from exc
        return cls(
            action=action,
            target=obj.get("target") or obj.get("block_id"),
            content=obj.get("content") or obj.get("summary"),
            query=obj.get("query"),
            reason=obj.get("reason", ""),
            max_tokens=obj.get("max_tokens"),
        )


class SimpleSummarizer:
    """Cheap extractive summarizer used when no LLM summarizer is provided."""

    def summarize_block(self, block: ContextBlock, task_goal: str = "", max_words: int = 45) -> str:
        text = " ".join(block.content.split())
        if not text:
            return ""

        # Prefer lines that look semantically informative for code/tool outputs.
        candidates = re.split(r"(?<=[.!?])\s+|\n+", block.content)
        scored: list[tuple[int, str]] = []
        keywords = [
            "error", "exception", "failed", "function", "class", "def ", "return",
            "log", "request", "response", "config", "route", "handler", "test",
            "todo", "fix", "current", "final", "decision", "root cause",
        ]
        for line in candidates:
            clean = " ".join(line.strip().split())
            if not clean:
                continue
            score = sum(1 for k in keywords if k.lower() in clean.lower())
            if task_goal:
                score += sum(1 for w in task_goal.lower().split() if len(w) > 4 and w in clean.lower())
            # Short informative lines are better than giant raw chunks.
            score -= max(0, len(clean.split()) - 60) // 20
            scored.append((score, clean))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[0][1] if scored and scored[0][0] > 0 else text
        words = chosen.split()
        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

    def summarize_episode(self, blocks: list[ContextBlock], task_goal: str = "", max_words: int = 90) -> str:
        parts = []
        if task_goal:
            parts.append(f"Goal: {task_goal}")
        for block in blocks:
            if block.summary:
                parts.append(f"{block.id}: {block.summary}")
        text = " ".join(parts) or " ".join(b.content for b in blocks[-3:])
        words = text.split()
        return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


class ContextWorkspace:
    """Structured context workspace with deterministic action execution."""

    def __init__(
        self,
        token_budget: int = 32_000,
        dashboard_top_k: int = 6,
        summarizer: SimpleSummarizer | None = None,
    ):
        self.token_budget = token_budget
        self.dashboard_top_k = dashboard_top_k
        self.summarizer = summarizer or SimpleSummarizer()
        self.blocks: dict[str, ContextBlock] = {}
        self.children: dict[str, list[str]] = {}
        self._next_id = 1
        self.current_episode_id: str | None = None
        self.living_state_block_id = self.add_block(
            type="living_state",
            content="# Living State\nNo durable state yet.",
            source="workspace",
            status="visible",
            summary="Current durable task state.",
        )

    # ---------------------------------------------------------------------
    # Block creation and hierarchy
    # ---------------------------------------------------------------------

    def _new_id(self, prefix: str = "B") -> str:
        block_id = f"{prefix}{self._next_id}"
        self._next_id += 1
        return block_id

    def estimate_tokens(self, text: str) -> int:
        # Conservative cheap approximation for dashboard accounting.
        return max(1, math.ceil(len(text.split()) * 1.3)) if text else 0

    def add_block(
        self,
        type: BlockType,
        content: str,
        source: str = "",
        parent_id: str | None = None,
        status: BlockStatus = "visible",
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        block_id: str | None = None,
    ) -> str:
        block_id = block_id or self._new_id("E" if type == "episode" else "B")
        if block_id in self.blocks:
            raise ContextActionError(f"Duplicate block id: {block_id}")
        block = ContextBlock(
            id=block_id,
            type=type,
            content=content,
            source=source,
            status=status,
            parent_id=parent_id,
            summary=summary,
            tokens=self.estimate_tokens(content),
            metadata=metadata or {},
        )
        self.blocks[block_id] = block
        if parent_id:
            self.children.setdefault(parent_id, []).append(block_id)
        return block_id

    def start_episode(self, user_query: str, source: str = "user") -> str:
        episode_id = self.add_block(
            type="episode",
            content=user_query,
            source=source,
            status="visible",
            summary=self.summarizer.summarize_block(
                ContextBlock(id="tmp", type="episode", content=user_query), max_words=25
            ),
        )
        self.current_episode_id = episode_id
        # The episode block itself carries the user query; no separate user_message
        # needed to avoid duplicating content in visible context.
        return episode_id

    def add_react_step(self, content: str = "") -> str:
        parent = self.current_episode_id
        return self.add_block("react_step", content or "ReAct step", source="agent", parent_id=parent)

    def add_tool_call(self, tool_name: str, arguments: dict[str, Any], parent_id: str | None = None) -> str:
        parent_id = parent_id or self.current_episode_id
        content = json.dumps({"tool": tool_name, "arguments": arguments}, ensure_ascii=False)
        return self.add_block("tool_call", content, source=tool_name, parent_id=parent_id, summary=f"Call {tool_name}")

    def add_tool_result(
        self,
        tool_name: str,
        content: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        parent_id = parent_id or self.current_episode_id
        block_type: BlockType = "tool_result"
        if tool_name in {"read_file", "file_system_read_file", "ide_simulator_read_file"}:
            block_type = "file_read"
        elif tool_name in {"bash", "run_shell", "compiler", "debugger"}:
            block_type = "bash_output"
        elif "search" in tool_name:
            block_type = "search_result"
        block_id = self.add_block(block_type, content, source=tool_name, parent_id=parent_id, metadata=metadata)
        # No auto-summary threshold: summaries are produced only when the agent
        # explicitly issues an ABSTRACT action. The extractive summarizer runs
        # inside the workspace, but we never trigger it automatically here.
        return block_id

    # ---------------------------------------------------------------------
    # Dashboard and visible context
    # ---------------------------------------------------------------------

    def current_goal(self) -> str:
        if self.current_episode_id and self.current_episode_id in self.blocks:
            return self.blocks[self.current_episode_id].summary or self.blocks[self.current_episode_id].content
        return ""

    def visible_blocks(self) -> list[ContextBlock]:
        return [b for b in self.blocks.values() if b.status in {"visible", "pinned"}]

    def hidden_blocks(self) -> list[ContextBlock]:
        return [b for b in self.blocks.values() if b.status == "hidden"]

    def visible_token_count(self) -> int:
        return sum(b.tokens for b in self.visible_blocks())

    def mini_dashboard(self) -> str:
        visible = self.visible_blocks()
        hidden = self.hidden_blocks()
        largest = sorted(visible, key=lambda b: b.tokens, reverse=True)[: self.dashboard_top_k]
        by_type = self._token_breakdown(visible)
        unsummarized = [b for b in largest if not b.summary and b.type not in {"living_state", "episode"}]
        archived = [b for b in hidden if b.summary][-self.dashboard_top_k:]
        lines = [
            "# Context Workspace",
            f"Active: {self.visible_token_count()} tokens across {len(visible)} blocks"
            f"  |  {len(hidden)} blocks archived (full content preserved, retrievable anytime)",
            "By type: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())),
            "Largest active blocks (these consume the most of your reasoning space):",
        ]
        lines.extend(b.to_dashboard_row(include_summary=True) for b in largest)
        if unsummarized:
            lines.append(
                "Blocks you've read but not yet summarized"
                " — if you're done with the raw detail, ABSTRACT then HIDE frees up space"
                " while preserving what you learned:"
            )
            lines.extend(
                f"- {b.id}: {b.type}, {b.tokens} tok, source={b.source or 'n/a'}"
                for b in unsummarized
            )
        if archived:
            lines.append("Recently archived (use ASK_ARCHIVE for a specific fact, or RESTORE to bring back in full):")
            lines.extend(
                f"- {b.id}: {b.type}, {b.tokens} tok, source={b.source or 'n/a'} — {b.summary}"
                for b in archived
            )
        return "\n".join(lines)

    def full_dashboard(self) -> str:
        episodes = [b for b in self.blocks.values() if b.type == "episode"]
        visible = self.visible_blocks()
        hidden = self.hidden_blocks()
        lines = [
            "# Context Dashboard (full)",
            f"Visible tokens: {self.visible_token_count()} / {self.token_budget}",
            "Visible token spend by type:",
        ]
        lines.extend(f"- {k}: {v}" for k, v in sorted(self._token_breakdown(visible).items()))
        lines.append("Archive token spend by type:")
        lines.extend(f"- {k}: {v}" for k, v in sorted(self._token_breakdown(hidden).items()))
        lines.append("Context tree:")
        for ep in episodes:
            child_ids = self.children.get(ep.id, [])
            child_tokens = sum(self.blocks[c].tokens for c in child_ids if c in self.blocks)
            lines.append(
                f"- {ep.id}: episode, status={ep.status}, children={len(child_ids)}, child_tokens={child_tokens}, summary={ep.summary or self._short(ep.content)}"
            )
            for child_id in child_ids:
                child = self.blocks.get(child_id)
                if not child:
                    continue
                coverage = "summarized" if child.summary else "raw"
                lines.append(
                    f"  - {child.id}: {child.type}, {child.tokens} tok, {child.status}, {coverage}, source={child.source or 'n/a'}, summary={child.summary or 'n/a'}"
                )
        lines.append("Visible blocks:")
        lines.extend(b.to_dashboard_row(include_summary=True) for b in visible)
        lines.append("Hidden blocks (content not active; use SHOW_BLOCK for details or RESTORE to make visible):")
        lines.extend(b.to_dashboard_row(include_summary=True) for b in hidden)
        suggestions = self.suggest_actions()
        if suggestions:
            lines.append("Workspace awareness (informational, no action required):")
            lines.extend(f"- {s}" for s in suggestions)
        return "\n".join(lines)

    def assemble_visible_context(self, include_dashboard: bool = True, max_block_tokens: int | None = None) -> str:
        parts = []
        if include_dashboard:
            parts.append(self.mini_dashboard())
        for block in self.visible_blocks():
            content = block.content
            if max_block_tokens and block.tokens > max_block_tokens:
                content = " ".join(content.split()[:max_block_tokens]) + " ... [truncated in visible context]"
            parts.append(f"\n<{block.id} type={block.type} source={block.source} status={block.status}>\n{content}\n</{block.id}>")
        return "\n".join(parts)

    # ---------------------------------------------------------------------
    # Actions
    # ---------------------------------------------------------------------

    def apply_actions(self, actions: Iterable[ContextAction | dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for raw_action in actions:
            action = raw_action if isinstance(raw_action, ContextAction) else ContextAction.from_obj(raw_action)
            results.append(self.apply_action(action))
        return results

    def apply_action(self, action: ContextAction) -> dict[str, Any]:
        if action.action == ActionType.ABSTRACT:
            return self.abstract(action.target, action.content, action.reason)
        if action.action == ActionType.HIDE:
            return self.hide(action.target, action.reason)
        if action.action == ActionType.ASK_ARCHIVE:
            return self.ask_archive(query=action.query or action.content or "", max_tokens=action.max_tokens)
        if action.action == ActionType.SHOW_BLOCK:
            return self.show_block(action.target, max_tokens=action.max_tokens)
        if action.action == ActionType.RESTORE:
            return self.restore(action.target)
        if action.action == ActionType.RECALL:
            # Backward-compatible behavior: target means SHOW_BLOCK, query means ASK_ARCHIVE.
            if action.target:
                return self.show_block(action.target, max_tokens=action.max_tokens)
            return self.ask_archive(query=action.query or "", max_tokens=action.max_tokens)
        if action.action == ActionType.ABSTRACT_AND_HIDE:
            abstract_result = self.abstract(action.target, action.content, action.reason)
            hide_result = self.hide(action.target, action.reason or "abstracted and hidden")
            return {"action": "ABSTRACT_AND_HIDE", "abstract": abstract_result, "hide": hide_result}
        raise ContextActionError(f"Unsupported action: {action.action}")

    def abstract(self, target: str | None, content: str | None, reason: str = "") -> dict[str, Any]:
        if not target or target not in self.blocks:
            raise ContextActionError(f"ABSTRACT target not found: {target}")
        if not content or not content.strip():
            raise ContextActionError("ABSTRACT requires non-empty content.")
        block = self.blocks[target]
        block.summary = content.strip()
        block.updated_at = time.time()
        block.metadata["abstract_reason"] = reason

        # Update episode summary if target belongs to an episode.
        episode_id = self._episode_for_block(target)
        if episode_id:
            self._refresh_episode_summary(episode_id)
        self._refresh_living_state()
        return {"ok": True, "action": "ABSTRACT", "target": target, "summary": block.summary}

    def hide(self, target: str | None, reason: str = "") -> dict[str, Any]:
        if not target or target not in self.blocks:
            raise ContextActionError(f"HIDE target not found: {target}")
        block = self.blocks[target]
        if block.status == "pinned":
            raise ContextActionError(f"Cannot hide pinned block: {target}")
        if block.type in {"living_state", "session"}:
            raise ContextActionError(f"Cannot hide block type {block.type}: {target}")
        block.status = "hidden"
        block.updated_at = time.time()
        block.metadata["hide_reason"] = reason
        return {"ok": True, "action": "HIDE", "target": target, "status": block.status}

    def ask_archive(self, query: str, max_tokens: int | None = None) -> dict[str, Any]:
        """Ask hidden archive for compact evidence without changing visibility."""
        if not query:
            raise ContextActionError("ASK_ARCHIVE requires a query.")
        matches = self.search_archive(query, top_k=3)
        budget = max_tokens or 700
        answers = []
        used = 0
        for block in matches:
            snippet = block.summary or self._short(block.content, words=70)
            snippet_tokens = self.estimate_tokens(snippet)
            if answers and used + snippet_tokens > budget:
                continue
            used += snippet_tokens
            answers.append({
                "block_id": block.id,
                "type": block.type,
                "source": block.source,
                "tokens": block.tokens,
                "summary_or_excerpt": snippet,
                "status": block.status,
            })
        return {"ok": True, "action": "ASK_ARCHIVE", "query": query, "answers": answers}

    def show_block(self, target: str | None, max_tokens: int | None = None) -> dict[str, Any]:
        """Return block content/excerpt without changing visibility."""
        if not target or target not in self.blocks:
            raise ContextActionError(f"SHOW_BLOCK target not found: {target}")
        block = self.blocks[target]
        budget = max_tokens or block.tokens  # show full content unless caller specifies a limit
        words = block.content.split()
        content = " ".join(words[:budget]) + (" ... [truncated]" if len(words) > budget else "")
        return {
            "ok": True,
            "action": "SHOW_BLOCK",
            "target": target,
            "block": {
                "id": block.id,
                "type": block.type,
                "source": block.source,
                "tokens": block.tokens,
                "status": block.status,
                "summary": block.summary,
                "content": content,
            },
        }

    def restore(self, target: str | None) -> dict[str, Any]:
        """Make a hidden block visible again. This is intentionally explicit."""
        if not target or target not in self.blocks:
            raise ContextActionError(f"RESTORE target not found: {target}")
        block = self.blocks[target]
        block.status = "visible"
        block.updated_at = time.time()
        return {"ok": True, "action": "RESTORE", "target": target, "status": block.status}

    def recall(self, target: str | None = None, query: str | None = None, max_tokens: int | None = None) -> dict[str, Any]:
        # Backward-compatible alias retained for older demos/tests.
        if target:
            return self.show_block(target, max_tokens=max_tokens)
        return self.ask_archive(query or "", max_tokens=max_tokens)

    # ---------------------------------------------------------------------
    # Search and state updates
    # ---------------------------------------------------------------------

    def search_archive(self, query: str, top_k: int = 5) -> list[ContextBlock]:
        q_terms = self._terms(query)
        candidates = [b for b in self.blocks.values() if b.status == "hidden"]
        scored = []
        for block in candidates:
            text = " ".join([block.source, block.summary, block.content[:2000]])
            terms = self._terms(text)
            overlap = len(q_terms & terms)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(max(1, len(terms)))
            # Prefer summaries and exact source/path matches.
            if query.lower() in block.source.lower():
                score += 2
            if block.summary:
                score += 0.25
            scored.append((score, block))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:top_k]]

    def _token_breakdown(self, blocks: list[ContextBlock]) -> dict[str, int]:
        out: dict[str, int] = {}
        for block in blocks:
            out[block.type] = out.get(block.type, 0) + block.tokens
        return out

    def suggest_actions(self) -> list[str]:
        """Return awareness notes shown in the full dashboard.

        These are informational only — the agent decides whether to act.
        """
        notes = []
        for block in sorted(self.visible_blocks(), key=lambda b: b.tokens, reverse=True):
            if block.type in {"living_state", "episode"}:
                continue
            if not block.summary:
                notes.append(
                    f"{block.id} ({block.type}, {block.tokens} tok) has no summary yet"
                )
            else:
                notes.append(
                    f"{block.id} ({block.type}, {block.tokens} tok) has a summary and could be archived if no longer needed"
                )
            if len(notes) >= self.dashboard_top_k:
                break
        return notes

    def _terms(self, text: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", text)}

    def _episode_for_block(self, block_id: str) -> str | None:
        current = self.blocks.get(block_id)
        while current and current.parent_id:
            parent = self.blocks.get(current.parent_id)
            if parent and parent.type == "episode":
                return parent.id
            current = parent
        return block_id if current and current.type == "episode" else None

    def _refresh_episode_summary(self, episode_id: str) -> None:
        child_ids = self.children.get(episode_id, [])
        children = [self.blocks[c] for c in child_ids if c in self.blocks]
        episode = self.blocks[episode_id]
        episode.summary = self.summarizer.summarize_episode(children, episode.content)
        episode.updated_at = time.time()

    def _refresh_living_state(self) -> None:
        if self.living_state_block_id not in self.blocks:
            return
        episodes = [b for b in self.blocks.values() if b.type == "episode"]
        latest = episodes[-3:]
        lines = ["# Living State"]
        for ep in latest:
            if ep.summary:
                lines.append(f"- {ep.id}: {ep.summary}")
        self.blocks[self.living_state_block_id].content = "\n".join(lines) if len(lines) > 1 else "# Living State\nNo durable state yet."
        self.blocks[self.living_state_block_id].tokens = self.estimate_tokens(self.blocks[self.living_state_block_id].content)
        self.blocks[self.living_state_block_id].updated_at = time.time()

    def _short(self, text: str, words: int = 20) -> str:
        toks = " ".join(text.split()).split()
        return " ".join(toks[:words]) + ("..." if len(toks) > words else "")

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_budget": self.token_budget,
            "next_id": self._next_id,
            "current_episode_id": self.current_episode_id,
            "living_state_block_id": self.living_state_block_id,
            "blocks": {k: v.to_dict() for k, v in self.blocks.items()},
            "children": self.children,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextWorkspace":
        ws = cls(token_budget=data.get("token_budget", 32_000))
        ws._next_id = data.get("next_id", 1)
        ws.current_episode_id = data.get("current_episode_id")
        ws.living_state_block_id = data.get("living_state_block_id", "B1")
        ws.blocks = {k: ContextBlock(**v) for k, v in data.get("blocks", {}).items()}
        ws.children = {k: list(v) for k, v in data.get("children", {}).items()}
        return ws

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "ContextWorkspace":
        return cls.from_dict(json.loads(Path(path).read_text()))


def parse_context_actions(text: str) -> list[ContextAction]:
    """Parse context actions from JSON text or fenced JSON."""
    raw = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    obj = json.loads(raw)
    if isinstance(obj, dict) and "context_actions" in obj:
        obj = obj["context_actions"]
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        raise ContextActionError("Expected a JSON action object or list.")
    return [ContextAction.from_obj(item) for item in obj]


def context_action_prompt() -> str:
    """Prompt snippet explaining context actions to the agent."""
    return f"""
## Managing your context workspace

Your workspace accumulates blocks as you work — file reads, tool outputs,
analysis notes, conversation turns. As you make progress, earlier blocks
become less relevant for what you're doing *right now*, and keeping them
all active makes it harder to focus on the current problem.

You can reshape your workspace whenever it feels useful. Here's why each
action exists and when it helps:

**ABSTRACT(target, content)**
You've read a file or tool output and extracted what you need. Rather than
keeping megabytes of raw content in your active view, write down the key
facts — function signatures, architecture decisions, error causes — and
store them as a durable note. This is like taking notes in a margin so you
can close the book.
  Example: after reading a 500-line source file, ABSTRACT with "exports
  handle_request(); uses epoll for async I/O; no auth middleware"

**HIDE(target, reason)**
A block is no longer relevant to your current line of work. Move it to the
archive — full content is preserved and instantly retrievable. This is not
deletion; it's putting something on the shelf so your desk stays clear.
Prefer doing ABSTRACT before HIDE so your note survives even if you never
look at the raw block again.

**ASK_ARCHIVE(query)**
You archived something earlier and now need one specific fact from it.
Instead of restoring the whole block (which adds noise back), ask a
targeted question and get just the answer. Good for "what was the function
signature I saw in X?" without cluttering your active view.

**SHOW_BLOCK(target)**
A quick peek at an archived block's raw content, without changing its
archived status. Useful for verifying a detail before deciding whether to
restore it fully.

**RESTORE(target)**
Bring an archived block back into your active workspace. Use this when
you need to work with the raw content again for an extended period — for
example, you're now writing code that directly depends on a file you
previously archived.

A useful rhythm: when you finish exploring a file or completing a subtask,
ABSTRACT the key findings and HIDE the raw block. Your workspace stays
focused on what comes next, and nothing is ever lost.

To act, include a JSON block anywhere in your response:
{{"context_actions": [
  {{"action": "ABSTRACT", "target": "B8", "content": "exports handle_request(); uses epoll"}},
  {{"action": "HIDE",     "target": "B8", "reason": "file read complete, key facts abstracted"}}
]}}

If you have nothing to archive right now, omit the JSON entirely.
""".strip()
