"""
Agentic self-managed context method for AMA-Bench.

This adapts the LOCA-Bench context workspace implementation to AMA's offline
trajectory QA shape: replay the trajectory as a growing conversation, let the
model manage the workspace when the context budget is tight, then answer from
the assembled workspace.

Alignment notes (vs LOCA-bench run_strict_lc.sh branch):
- Uses the same WorkspaceManager engine (assemble, preflight_offload, set_overhead)
- Unified tiktoken encoder (cl100k_base) passed to all token-counting calls
- preflight_offload_raw_tool_results fires before LLM management loop (safety net)
- keep_recent_blocks gracefully falls back to full candidate list when exhausted
- Management loop uses warnings instead of RuntimeError (LOCA-style fault tolerance)
- _state_cache cleared at each management round (mirrors run_react.py:1846)

Intentional offline-only differences kept:
- Two-phase memory_construction / memory_retrieve interface
- trajectory replay for-loop (steps arrive offline, not via live tool calls)
- _ask_for_archive_action: code-driven LLM prompt (vs agent calling MCP tool directly)
- _LOCA_TOOL_LOCK + env-var swap for thread-safe Python-function archive call
- externalize_large_observations for huge AMA trajectory observations
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.method.base_method import BaseMethod


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOCA_WORKSPACE_DIR = (
    PROJECT_ROOT
    / "benchmarks"
    / "LOCAbench"
    / "gem"
    / "tools"
    / "mcp_server"
    / "context_workspace"
)
if not LOCA_WORKSPACE_DIR.is_dir():
    raise ImportError(
        "VISTA's LOCA context-workspace core was not found at "
        f"{LOCA_WORKSPACE_DIR}. Run this adapter from the published VISTA "
        "repository layout or set up the benchmark tree first."
    )
if str(LOCA_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCA_WORKSPACE_DIR))

from workspace_manager import WorkspaceManager, count_msg_tokens  # noqa: E402
from server import context_workspace_archive  # noqa: E402


_LOCA_TOOL_LOCK = threading.Lock()


_STEP_RE = re.compile(
    r"(?ms)^\s*(?:Step|Turn)\s+(\d+):\s*\n"
    r"Action:\s*(.*?)\n"
    r"Observation:\s*(.*?)(?=^\s*(?:Step|Turn)\s+\d+:\s*\nAction:|\Z)"
)


@dataclass
class SMSReplayMemory:
    manager: WorkspaceManager
    messages: List[Dict[str, Any]]
    task: str
    workspace_dir: str
    construction_events: List[Dict[str, Any]] = field(default_factory=list)
    construction_warnings: List[str] = field(default_factory=list)


class SelfManagedAgenticMethod(BaseMethod):
    """
    LOCA-style self-managed context replay.

    Construction is online with respect to the trajectory prefix: the future
    question is not shown while the workspace is being managed.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        client: Optional[Any] = None,
        embedding_engine: Optional[Any] = None,
    ):
        config = self._load_config(config_path) if config_path else {}
        self.client = client
        self.embedding_engine = embedding_engine
        self.token_budget = int(config.get("token_budget", 8192))
        self.high_watermark = float(config.get("high_watermark", 0.80))
        self.keep_recent_blocks = int(config.get("keep_recent_blocks", 8))
        self.management_max_tokens = int(config.get("management_max_tokens", 4096))
        self.max_management_rounds = int(config.get("max_management_rounds", 20))
        self.temperature = float(config.get("temperature", 0.0))
        self.externalize_large_observations = bool(config.get("externalize_large_observations", False))
        self.externalize_observation_tokens = int(config.get("externalize_observation_tokens", 24000))
        self.externalized_observation_preview_chars = int(
            config.get("externalized_observation_preview_chars", 6000)
        )
        # Preflight offload configuration (mirrors LOCA SM_PREFLIGHT_TARGET_RATIO /
        # SM_PREFLIGHT_TURN_RESERVE_TOKENS env vars from run_strict_lc.sh)
        self.preflight_target_ratio = float(config.get("preflight_target_ratio", 0.98))
        self.preflight_target_ratio = max(0.50, min(1.00, self.preflight_target_ratio))
        self.preflight_turn_reserve_tokens = int(config.get("preflight_turn_reserve_tokens", 0))

        self.requires_embedding = False
        self.requires_network = True

        # Unified tiktoken encoder — mirrors LOCA run_react.py:1783-1784.
        # All token-counting calls in this class pass self._tkt_enc so that
        # char//3 fallback is never silently triggered mid-run.
        try:
            import tiktoken as _tkt
            self._tkt_enc = _tkt.get_encoding("cl100k_base")
        except Exception:
            self._tkt_enc = None

    def memory_construction(self, traj_text: str, task: str = "") -> SMSReplayMemory:
        if self.client is None:
            raise ValueError("self_managed_agentic requires a model client for construction-time management.")

        workspace_dir = Path(tempfile.mkdtemp(prefix="ama_sms_replay_"))
        public_payload_dir = workspace_dir / "public_payloads"
        manager = WorkspaceManager(
            workspace_dir,
            token_budget=self.token_budget,
            public_payload_dir=public_payload_dir,
        )

        # Register protocol overhead so preflight / budget calculations match
        # LOCA run_react.py:1782-1794 (which measures tool-schema overhead).
        # In AMA offline replay there are no tool schemas, so we measure the
        # fixed management-prompt header instead.
        try:
            protocol_overhead = count_msg_tokens(
                [{"role": "user", "content": self._protocol_text()}],
                tkt_enc=self._tkt_enc,
            )
            manager.set_overhead(protocol_overhead)
        except Exception:
            pass  # overhead stays 0; mirrors LOCA's silent except block

        protocol = self._protocol_text()
        messages: List[Dict[str, Any]] = [{
            "role": "user",
            "content": (
                f"{protocol}\n\n"
                f"## AMA Replay Task\n{task or 'Answer questions about the replayed trajectory.'}\n\n"
                "The trajectory will be replayed incrementally. During replay, manage context only; "
                "do not answer any future question."
            ),
        }]
        manager.register_message(messages[0], 0)

        events: List[Dict[str, Any]] = []
        construction_warnings: List[str] = []
        steps = self._parse_steps(traj_text)
        if not steps and traj_text.strip():
            steps = [{"turn_idx": 0, "action": "", "observation": traj_text.strip()}]

        for step in steps:
            msg, pending_notify, externalize_events = self._build_step_message(manager, step)
            events.extend(externalize_events)
            step_events, step_warnings = self._manage_until_next_fits(
                manager, messages, msg, step["turn_idx"]
            )
            events.extend(step_events)
            construction_warnings.extend(step_warnings)
            messages.append(msg)
            msg_idx = len(messages) - 1
            manager.register_message(msg, msg_idx)
            for block_id in pending_notify:
                manager.set_notify_msg_idx(block_id, msg_idx)
            manager.update_dashboard_cache()

        manager.update_dashboard_cache()
        return SMSReplayMemory(
            manager=manager,
            messages=messages,
            task=task,
            workspace_dir=str(workspace_dir),
            construction_events=events,
            construction_warnings=construction_warnings,
        )

    def memory_retrieve(self, memory: SMSReplayMemory, question: str) -> str:
        if not isinstance(memory, SMSReplayMemory):
            raise ValueError("Memory must be an SMSReplayMemory object")

        memory.manager.update_dashboard_cache()
        assembled = memory.manager.assemble(memory.messages)
        dashboard = memory.manager.get_dashboard()
        state = memory.manager.get_state()
        metrics = self._metrics(memory.manager, memory.messages, state)
        event_lines = "\n".join(
            f"- step {e.get('step')}: {e.get('result')}"
            for e in memory.construction_events[-12:]
        ) or "- no archive actions were needed"
        warning_lines = (
            "\n".join(f"- {w}" for w in memory.construction_warnings[-6:])
            if memory.construction_warnings
            else "- none"
        )

        return (
            "## Self-Managed Context Replay\n"
            "This context was built by replaying the trajectory prefix and letting the model archive "
            "workspace blocks when the LOCA-style context budget was tight.\n\n"
            f"## Task Description\n{memory.task}\n\n"
            f"## Context Workspace Status\n{dashboard}\n\n"
            f"## Construction Events\n{event_lines}\n\n"
            f"## Construction Warnings\n{warning_lines}\n\n"
            f"## Diagnostics\n{metrics}\n\n"
            f"## Assembled Workspace\n{self._messages_to_text(assembled)}\n\n"
            f"## Current Question\n{question}"
        )

    def _manage_until_next_fits(
        self,
        manager: WorkspaceManager,
        messages: List[Dict[str, Any]],
        next_msg: Dict[str, Any],
        step_idx: int,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        events: List[Dict[str, Any]] = []
        step_warnings: List[str] = []
        limit_tokens = int(self.token_budget * self.high_watermark)

        # ── Preflight offload (mirrors LOCA run_react.py:1924-1936) ─────────
        # Deterministic safety pass: replace the largest visible raw tool
        # results with stable offload placeholders before entering the LLM
        # decision loop.  This fires first so the LLM gets a smaller, cleaner
        # candidate list, and so the loop has fewer rounds to converge.
        try:
            state = manager.get_state()
            _fixed_overhead = int(state.get("overhead_tokens", 0) or 0)
            _dashboard_overhead = count_msg_tokens(
                [{"role": "user", "content": f"<context_workspace_status>\n{manager.get_dashboard()}\n</context_workspace_status>"}],
                tkt_enc=self._tkt_enc,
            )
            _target_preflight = (
                int(self.token_budget * self.preflight_target_ratio)
                - _fixed_overhead
                - _dashboard_overhead
                - self.preflight_turn_reserve_tokens
            )
            offloaded = manager.preflight_offload_raw_tool_results(
                messages,
                target_conv_tokens=max(1, _target_preflight),
                tkt_enc=self._tkt_enc,
            )
            if offloaded:
                w = f"[SM-SAFETY] preflight offload fired at step {step_idx} ({offloaded} block(s))"
                step_warnings.append(w)
                warnings.warn(w)
        except Exception as _e:
            step_warnings.append(f"[SM-SAFETY] preflight offload failed at step {step_idx}: {_e}")

        # ── LLM-driven archive loop ──────────────────────────────────────────
        for round_idx in range(1, self.max_management_rounds + 1):
            # Clear cache each round — mirrors LOCA run_react.py:1846
            manager._state_cache = None

            before_tokens = self._current_tokens_with_next(manager, messages, next_msg)
            if before_tokens <= limit_tokens:
                break

            action = self._ask_for_archive_action(manager, messages, step_idx)
            applied = False
            for item in action.get("actions", []):
                block_id = str(item.get("block_id", "")).strip()
                index = str(item.get("index", "")).strip()
                extract = str(item.get("extract", "")).strip()
                if not block_id or not index:
                    continue
                result = self._archive(manager, block_id, index, extract)
                events.append({"step": step_idx, "block_id": block_id, "result": result})
                applied = applied or result.startswith("[ARCHIVED:")

            manager.update_dashboard_cache()

            if not applied:
                # LOCA-style: warn instead of crash (mirrors run_react.py warning log pattern)
                w = (
                    f"[SM-WARN] step {step_idx} round {round_idx}: no archive action applied "
                    f"(context ~{before_tokens:,} tok, limit {limit_tokens:,})"
                )
                step_warnings.append(w)
                warnings.warn(w)
                break

            after_tokens = self._current_tokens_with_next(manager, messages, next_msg)
            if after_tokens >= before_tokens:
                w = (
                    f"[SM-WARN] step {step_idx} round {round_idx}: archive made no progress "
                    f"({before_tokens:,} -> {after_tokens:,} tok)"
                )
                step_warnings.append(w)
                warnings.warn(w)
                break

        final_tokens = self._current_tokens_with_next(manager, messages, next_msg)
        if final_tokens > limit_tokens:
            w = (
                f"[SM-WARN] step {step_idx}: context still exceeds limit after "
                f"{self.max_management_rounds} rounds "
                f"(~{final_tokens:,} tok > limit {limit_tokens:,})"
            )
            step_warnings.append(w)
            warnings.warn(w)

        return events, step_warnings

    def _ask_for_archive_action(
        self,
        manager: WorkspaceManager,
        messages: List[Dict[str, Any]],
        step_idx: int,
    ) -> Dict[str, Any]:
        state = manager.get_state()
        valid_blocks = [
            b for b in sorted(state.get("blocks", {}).values(), key=lambda x: x.get("msg_idx", 0))
            if b.get("status") == "visible"
        ]
        # Dynamic recent-block protection: walk backwards from the newest block,
        # accumulating token estimates until we hit 40% of the total token budget
        # or a hard cap of 10 blocks (whichever comes first).  This prevents
        # dead-locks where a small number of large recent blocks would otherwise
        # consume the full protection quota and leave no viable candidates.
        # If all blocks fall inside the protection zone, fall back to the full
        # list so the loop can still make progress.
        protect_token_limit = self.token_budget * 0.40
        protect_count_limit = min(10, self.keep_recent_blocks) if self.keep_recent_blocks > 0 else 10
        protected_count = 0
        cumulative_tokens = 0
        for b in reversed(valid_blocks):
            tok = int(b.get("tiktoken_tokens") or len(b.get("content", "")) // 3)
            if cumulative_tokens + tok > protect_token_limit or protected_count >= protect_count_limit:
                break
            cumulative_tokens += tok
            protected_count += 1
        candidates = valid_blocks[:-protected_count] if 0 < protected_count < len(valid_blocks) else valid_blocks

        valid_block_lines = "\n".join(
            f"- {b['id']} status={b.get('status')} msg_idx={b.get('msg_idx')} summary={b.get('summary', '')}"
            for b in candidates
        ) or "- none"

        prompt = (
            "You are managing a LOCA-style self-managed context workspace during AMA trajectory replay.\n"
            "The future question is hidden. Your only job is to reduce context usage while preserving "
            "facts that may be needed later.\n\n"
            "Use only archive actions. Do not delete. Do not answer the task.\n"
            "Only target currently valid block IDs from <valid_archive_targets>. Do not invent block IDs.\n"
            "Archive older raw blocks into compact indexes. Preserve exact step numbers, actions, "
            "observed state changes, errors, goals, object names, counts, and causal relations.\n\n"
            "Return JSON only in this schema:\n"
            "{\"actions\":[{\"block_id\":\"B2-B10\","
            "\"index\":\"topic guide: what subjects/info types the archived content covers\","
            "\"extract\":\"verbatim excerpts of critical details (exact IDs, numbers, names) — copy character-for-character, leave empty if not needed\"}]}\n\n"
            f"Current replay step: {step_idx}\n"
            f"Token budget: {self.token_budget}; high-water limit: {int(self.token_budget * self.high_watermark)}\n\n"
            f"<valid_archive_targets>\n{valid_block_lines}\n</valid_archive_targets>\n\n"
            f"<context_workspace_status>\n{manager.get_dashboard()}\n</context_workspace_status>\n\n"
            "<assembled_context>\n"
            f"{self._messages_to_text(manager.assemble(messages))}\n"
            "</assembled_context>"
        )
        raw = self.client.query(
            prompt,
            temperature=self.temperature,
            max_tokens=self.management_max_tokens,
        )
        return self._parse_json_object(raw)

    def _archive(self, manager: WorkspaceManager, target_ids: str, index: str, extract: str = "") -> str:
        # Execute the exact LOCA MCP tool implementation against this replay
        # workspace. The server module is process-global, so protect the env
        # swap for threaded AMA runs.
        with _LOCA_TOOL_LOCK:
            old_workspace = os.environ.get("CONTEXT_WORKSPACE_DIR")
            old_payload = os.environ.get("CONTEXT_WORKSPACE_PAYLOAD_DIR")
            os.environ["CONTEXT_WORKSPACE_DIR"] = str(manager.workspace_dir.resolve())
            if manager.public_payload_dir:
                os.environ["CONTEXT_WORKSPACE_PAYLOAD_DIR"] = str(manager.public_payload_dir.resolve())
            try:
                # The reference LOCA tool calls this field ``replacement``.
                # AMA's replay policy historically emitted an ``index`` plus an
                # optional verbatim ``extract``; keep both in the compact handle
                # while the complete source block remains in the exact payload.
                replacement = index.strip()
                if extract.strip():
                    replacement = f"{replacement}\nCritical extract: {extract.strip()}"
                result = context_workspace_archive(target_ids, replacement=replacement)
            finally:
                if old_workspace is None:
                    os.environ.pop("CONTEXT_WORKSPACE_DIR", None)
                else:
                    os.environ["CONTEXT_WORKSPACE_DIR"] = old_workspace
                if old_payload is None:
                    os.environ.pop("CONTEXT_WORKSPACE_PAYLOAD_DIR", None)
                else:
                    os.environ["CONTEXT_WORKSPACE_PAYLOAD_DIR"] = old_payload
        manager._state_cache = None
        return result

    def _parse_steps(self, traj_text: str) -> List[Dict[str, Any]]:
        return [
            {
                "turn_idx": int(match.group(1)),
                "action": match.group(2).strip(),
                "observation": match.group(3).strip(),
            }
            for match in _STEP_RE.finditer(traj_text)
        ]

    def _format_step(self, step: Dict[str, Any]) -> str:
        return (
            f"Step {step['turn_idx']}:\n"
            f"Action: {step.get('action', '')}\n"
            f"Observation: {step.get('observation', '')}"
        )

    def _build_step_message(
        self,
        manager: WorkspaceManager,
        step: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
        full_msg = {"role": "user", "content": self._format_step(step)}
        full_tokens = count_msg_tokens([full_msg], tkt_enc=self._tkt_enc)
        if not self.externalize_large_observations or full_tokens <= self.externalize_observation_tokens:
            return full_msg, [], []

        raw_block_id = manager.register_message(full_msg, -1, blocked=True)
        manager.update_dashboard_cache()
        state = manager.get_state()
        raw_block = state.get("blocks", {}).get(raw_block_id, {})
        public_path = raw_block.get("public_payload_path") or raw_block.get("payload_path") or ""
        preview = self._observation_preview(step.get("observation", ""))
        placeholder = {
            "role": "user",
            "content": (
                f"Step {step['turn_idx']}:\n"
                f"Action: {step.get('action', '')}\n"
                f"Observation: [EXTERNAL_TOOL_TRANSCRIPT:{raw_block_id}]\n"
                "The full observation/tool transcript for this step was too large to admit verbatim "
                "during AMA replay, so it is stored outside visible context as a LOCA blocked payload.\n"
                f"Payload path: {public_path}\n"
                f"Estimated original step tokens: {full_tokens}\n"
                "Payload semantics: this is the transcript returned at this trajectory step, not a "
                "guarantee of complete source data.\n\n"
                f"{preview}"
            ),
        }
        event = {
            "step": step["turn_idx"],
            "block_id": raw_block_id,
            "result": (
                f"[EXTERNALIZED:{raw_block_id}] large observation stored as blocked payload "
                f"(~{full_tokens} tokens)"
            ),
        }
        return placeholder, [raw_block_id], [event]

    def _observation_preview(self, observation: str) -> str:
        text = str(observation or "")
        max_chars = max(0, self.externalized_observation_preview_chars)
        if max_chars <= 0 or len(text) <= max_chars:
            return f"Observation preview:\n{text}"

        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        head = text[:head_chars].rstrip()
        tail = text[-tail_chars:].lstrip()
        omitted = len(text) - len(head) - len(tail)
        return (
            "Observation preview head:\n"
            f"{head}\n\n"
            f"[... {omitted} characters omitted from visible context; full transcript is in the "
            "external payload ...]\n\n"
            "Observation preview tail:\n"
            f"{tail}"
        )

    def _current_tokens(self, manager: WorkspaceManager, messages: List[Dict[str, Any]]) -> int:
        dashboard_msg = {
            "role": "user",
            "content": f"<context_workspace_status>\n{manager.get_dashboard()}\n</context_workspace_status>",
        }
        return (
            manager.conv_tokens(messages, self._tkt_enc)
            + count_msg_tokens([dashboard_msg], tkt_enc=self._tkt_enc)
        )

    def _current_tokens_with_next(
        self,
        manager: WorkspaceManager,
        messages: List[Dict[str, Any]],
        next_msg: Dict[str, Any],
    ) -> int:
        dashboard_msg = {
            "role": "user",
            "content": f"<context_workspace_status>\n{manager.get_dashboard()}\n</context_workspace_status>",
        }
        return (
            manager.conv_tokens(messages, self._tkt_enc)
            + count_msg_tokens([next_msg], tkt_enc=self._tkt_enc)
            + count_msg_tokens([dashboard_msg], tkt_enc=self._tkt_enc)
        )

    def _metrics(self, manager: WorkspaceManager, messages: List[Dict[str, Any]], state: Dict[str, Any]) -> str:
        blocks = list(state.get("blocks", {}).values())
        compressed = [b for b in blocks if b.get("status") == "compressed"]
        visible = [b for b in blocks if b.get("status") in ("visible", "pinned")]
        data = {
            "workspace_dir": str(manager.workspace_dir),
            "token_budget": self.token_budget,
            "assembled_tokens": self._current_tokens(manager, messages),
            "blocks": len(blocks),
            "visible_or_pinned_blocks": len(visible),
            "compressed_blocks": len(compressed),
            "archive_groups": len(state.get("archive_groups", {})),
        }
        return "\n".join(f"- {k}: {v}" for k, v in data.items())

    def _messages_to_text(self, messages: List[Dict[str, Any]]) -> str:
        parts = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            parts.append(f"[{i}] {role.upper()}\n{content}")
        return "\n\n".join(parts)

    def _parse_json_object(self, raw: str) -> Dict[str, Any]:
        text = raw.strip()
        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            obj = json.loads(match.group(0)) if match else {}
        if not isinstance(obj, dict):
            return {"actions": []}
        actions = obj.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        obj["actions"] = actions
        return obj

    def _protocol_text(self) -> str:
        # Mirrors LOCA run_react.py context_workspace_protocol (archive branch,
        # i.e. SM_DISABLE_ARCHIVE not set — equivalent to run_strict_lc.sh).
        return (
            "CONTEXT MANAGEMENT PROTOCOL:\n"
            "Manage context using the available context workspace tools. "
            "Use context tools only when they help keep the conversation within the window. "
            "Archive completed context that is no longer needed verbatim. "
            "For structured data or calculations, use the source file, source tool, or query directly. "
            "Do not copy table, CSV, or JSON rows from the conversation into code."
        )
