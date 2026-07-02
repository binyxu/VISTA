"""
Context Management System for LoCoBench-Agent

Provides intelligent context window management for multi-turn agent conversations
to handle model context limits gracefully while preserving evaluation integrity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import os
import tiktoken
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkspaceLogger:
    """Append-only per-session workspace log.

    Writes one JSON record per compress_context() call to a .jsonl file so that
    the entire context lifecycle can be replayed offline.

    Each record contains:
    - turn_index     : monotonic call counter (0, 1, 2 …)
    - timestamp      : ISO-8601 wall-clock time
    - session_id     : passed from WorkspaceContextManager
    - visible_tokens : total tokens in the visible workspace
    - visible_blocks : summary list of currently visible blocks
    - hidden_blocks  : summary list of currently archived/hidden blocks
    - dashboard      : the full mini_dashboard() string injected into the LLM
    - workspace_snapshot : complete workspace.to_dict() (blocks, children, etc.)
                           — omitted when log_full_snapshot=False to save space

    Usage::

        wl = WorkspaceLogger(log_dir="/path/to/results/workspace_logs",
                             session_id="eval_..._001",
                             log_full_snapshot=True)
        wl.log(workspace, dashboard_text)
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        session_id: str,
        log_full_snapshot: bool = True,
    ):
        self.session_id = session_id
        self.log_full_snapshot = log_full_snapshot
        self._turn_index = 0
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # One file per session; safe_name strips characters that break filenames.
        safe_sid = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
        self._path = log_dir / f"{safe_sid}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def log(self, workspace: Any, dashboard: str) -> None:
        """Append one turn snapshot to the log file."""
        visible = workspace.visible_blocks()
        hidden = workspace.hidden_blocks()

        record: Dict[str, Any] = {
            "turn_index": self._turn_index,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "visible_tokens": workspace.visible_token_count(),
            "token_budget": workspace.token_budget,
            "visible_blocks": [
                {
                    "id": b.id,
                    "type": b.type,
                    "tokens": b.tokens,
                    "source": b.source,
                    "summary": b.summary,
                }
                for b in visible
            ],
            "hidden_blocks": [
                {
                    "id": b.id,
                    "type": b.type,
                    "tokens": b.tokens,
                    "source": b.source,
                    "summary": b.summary,
                    "hide_reason": b.metadata.get("hide_reason", ""),
                }
                for b in hidden
            ],
            "dashboard": dashboard,
        }
        if self.log_full_snapshot:
            record["workspace_snapshot"] = workspace.to_dict()

        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._turn_index += 1

    def read(self) -> List[Dict[str, Any]]:
        """Read all records from this session's log file."""
        if not self._path.exists():
            return []
        records = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


class ContextManagementStrategy(Enum):
    """Available context management strategies"""
    NONE = "none"              # No management - may fail on long contexts
    BASIC = "basic"            # Simple turn deletion strategy
    ADAPTIVE = "adaptive"      # Intelligent compression and summarization
    LIFECYCLE = "lifecycle"    # Active visibility control with living state
    WORKSPACE = "workspace"    # Block dashboard + archive workspace prototype


@dataclass
class ContextManagementConfig:
    """Configuration for context management"""
    strategy: ContextManagementStrategy = ContextManagementStrategy.ADAPTIVE
    
    # Thresholds (as percentage of max context) - MUCH more aggressive
    early_warning_threshold: float = 0.4    # 40% (very aggressive)
    critical_threshold: float = 0.6         # 60% (very aggressive)
    
    # Basic strategy settings
    preserve_initial_turns: int = 2          # Always keep first N turns
    
    # Adaptive strategy settings
    preserve_recent_turns: int = 3           # Keep last N turns in detail
    enable_conversation_summary: bool = True # Generate summaries of removed turns
    enable_file_compression: bool = True     # Compress inactive project files
    enable_architectural_summary: bool = True # Create architectural summaries
    
    # Token counting settings
    model_name: str = "gpt-4"               # For tiktoken encoding
    max_context_tokens: int = 128000        # Model's context limit
    response_buffer_tokens: int = 4096      # Reserve tokens for response


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    turn_number: int
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "metadata": self.metadata
        }


@dataclass
class ContextState:
    """Current state of the conversation context"""
    conversation_turns: List[ConversationTurn] = field(default_factory=list)
    project_files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    active_files: List[str] = field(default_factory=list)       # currently active files
    conversation_summary: str = ""                               # summary of compressed turns
    architectural_summary: str = ""                             # summary of compressed files
    living_state: str = ""                                      # always-on compact task/project state
    hidden_turns: List[ConversationTurn] = field(default_factory=list)  # full-fidelity hidden history
    pinned_turns: List[ConversationTurn] = field(default_factory=list)  # critical turns kept visible
    metadata: Dict[str, Any] = field(default_factory=dict)      # additional metadata (e.g., file_structure)
    
    total_tokens: int = 0
    last_compression_turn: int = 0
    compression_history: List[Dict[str, Any]] = field(default_factory=list)


class BaseContextManager(ABC):
    """Abstract base class for context management strategies"""
    
    def __init__(self, config: ContextManagementConfig):
        self.config = config
        # Try to get encoding for model, with fallback for unknown models
        try:
            self.encoding = tiktoken.encoding_for_model(config.model_name)
        except KeyError:
            # For unknown/new models, use intelligent fallback based on model name
            # Encoding guide:
            # - o200k_base: o1, o3, o4 series (reasoning models)
            # - cl100k_base: GPT-4, GPT-4o, GPT-4.1, GPT-5 series (standard models)
            fallback_encoding = "cl100k_base"  # Default for most modern models
            
            model_lower = config.model_name.lower()
            
            # Check for o-series reasoning models (use o200k_base)
            if any(prefix in model_lower for prefix in ["o1", "o3", "o4"]):
                fallback_encoding = "o200k_base"
            # Check for GPT-5, GPT-4.x, GPT-4o series (use cl100k_base)
            elif any(prefix in model_lower for prefix in ["gpt-5", "gpt-4.1", "gpt-4o", "gpt-4"]):
                fallback_encoding = "cl100k_base"
            
            logger.info(f"Unknown model '{config.model_name}', using {fallback_encoding} encoding")
            self.encoding = tiktoken.get_encoding(fallback_encoding)
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using character approximation")
            return len(text) // 4  # Rough approximation: 4 chars per token
    
    def calculate_context_usage(self, state: ContextState) -> Tuple[int, float]:
        """Calculate current token usage and percentage"""
        total_tokens = 0
        
        # Count conversation tokens
        for turn in state.conversation_turns:
            if turn.token_count == 0:
                turn.token_count = self.count_tokens(turn.content)
            total_tokens += turn.token_count
        
        # Count project file tokens
        for filename, content in state.project_files.items():
            total_tokens += self.count_tokens(content)
        
        # Count summary tokens
        if state.conversation_summary:
            total_tokens += self.count_tokens(state.conversation_summary)
        if state.architectural_summary:
            total_tokens += self.count_tokens(state.architectural_summary)
        
        state.total_tokens = total_tokens
        usage_percentage = total_tokens / self.config.max_context_tokens
        
        return total_tokens, usage_percentage
    
    @abstractmethod
    def should_compress(self, state: ContextState) -> bool:
        """Check if context compression is needed"""
        pass
    
    @abstractmethod
    def compress_context(self, state: ContextState) -> ContextState:
        """Compress the context to fit within limits"""
        pass
    
    def log_compression(self, state: ContextState, compression_type: str, details: Dict[str, Any]):
        """Log compression event for analysis"""
        compression_event = {
            "timestamp": datetime.now().isoformat(),
            "compression_type": compression_type,
            "turn_number": len(state.conversation_turns),
            "tokens_before": details.get("tokens_before", 0),
            "tokens_after": details.get("tokens_after", 0),
            "details": details
        }
        state.compression_history.append(compression_event)
        
        logger.info(f"Context compression: {compression_type} at turn {len(state.conversation_turns)}")
        logger.info(f"Tokens: {details.get('tokens_before', 0)} → {details.get('tokens_after', 0)}")


class NoContextManager(BaseContextManager):
    """No context management - allows natural overflow (may cause failures)"""
    
    def should_compress(self, state: ContextState) -> bool:
        return False
    
    def compress_context(self, state: ContextState) -> ContextState:
        return state  # No compression


class BasicContextManager(BaseContextManager):
    """Basic context management using simple turn deletion"""
    
    def should_compress(self, state: ContextState) -> bool:
        _, usage_percentage = self.calculate_context_usage(state)
        return usage_percentage >= self.config.early_warning_threshold
    
    def compress_context(self, state: ContextState) -> ContextState:
        """Compress by deleting turns (preserve initial + recent turns)"""
        tokens_before, usage_before = self.calculate_context_usage(state)
        
        if not self.should_compress(state):
            return state
        
        # Find turns to delete (skip initial turns and recent turns)
        turns = state.conversation_turns
        preserve_initial = self.config.preserve_initial_turns
        
        # Find the oldest deletable turn (after preserved initial turns)
        deletable_turns = []
        for i, turn in enumerate(turns):
            if i >= preserve_initial:  # Skip initial preserved turns
                deletable_turns.append(i)
        
        if not deletable_turns:
            logger.warning("No deletable turns found - cannot compress further")
            return state
        
        # Delete the oldest deletable turn
        turn_to_delete = deletable_turns[0]
        deleted_turn = turns.pop(turn_to_delete)
        
        # Update turn numbers
        for i, turn in enumerate(turns):
            turn.turn_number = i + 1
        
        # Recalculate tokens
        tokens_after, usage_after = self.calculate_context_usage(state)
        
        # Log compression
        self.log_compression(state, "basic_turn_deletion", {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "usage_before": usage_before,
            "usage_after": usage_after,
            "deleted_turn": turn_to_delete + 1,
            "deleted_content_preview": deleted_turn.content[:100] + "..."
        })
        
        return state


class AdaptiveContextManager(BaseContextManager):
    """Adaptive context management with intelligent compression"""
    
    def should_compress(self, state: ContextState) -> bool:
        _, usage_percentage = self.calculate_context_usage(state)
        return usage_percentage >= self.config.early_warning_threshold
    
    def compress_context(self, state: ContextState) -> ContextState:
        """Intelligent context compression with multiple strategies"""
        
        # CRITICAL: First enforce maximum message size limits to prevent API errors
        max_message_size = 8_000_000  # 8MB safety margin (OpenAI limit is 10MB)
        for turn in state.conversation_turns:
            if len(turn.content) > max_message_size:
                logger.warning(f"Turn {turn.turn_number} too large ({len(turn.content)} chars), truncating")
                turn.content = turn.content[:max_message_size] + "\n\n[Content truncated due to size limit]"
        
        tokens_before, usage_before = self.calculate_context_usage(state)
        
        if usage_before < self.config.early_warning_threshold:
            return state
        
        compression_actions = []
        
        # Strategy 1: Compress conversation history (80% threshold)
        if usage_before >= self.config.early_warning_threshold:
            state = self._compress_conversation_history(state)
            compression_actions.append("conversation_history")
        
        # Strategy 2: Compress inactive files (still above threshold)
        tokens_mid, usage_mid = self.calculate_context_usage(state)
        if usage_mid >= self.config.early_warning_threshold and self.config.enable_file_compression:
            state = self._compress_inactive_files(state)
            compression_actions.append("inactive_files")
        
        # Strategy 3: Aggressive truncation (95% threshold)
        tokens_after, usage_after = self.calculate_context_usage(state)
        if usage_after >= self.config.critical_threshold:
            state = self._aggressive_truncation(state)
            compression_actions.append("aggressive_truncation")
        
        # Final token count
        tokens_final, usage_final = self.calculate_context_usage(state)
        
        # Log compression
        self.log_compression(state, "adaptive_compression", {
            "tokens_before": tokens_before,
            "tokens_after": tokens_final,
            "usage_before": usage_before,
            "usage_after": usage_final,
            "actions": compression_actions
        })
        
        return state
    
    def _compress_conversation_history(self, state: ContextState) -> ContextState:
        """Compress conversation by summarizing old turns"""
        turns = state.conversation_turns
        preserve_recent = self.config.preserve_recent_turns
        
        if len(turns) <= preserve_recent + 2:  # Not enough turns to compress
            return state
        
        # Identify turns to summarize (all except recent ones)
        turns_to_summarize = turns[:-preserve_recent] if preserve_recent > 0 else turns[:-1]
        recent_turns = turns[-preserve_recent:] if preserve_recent > 0 else [turns[-1]]
        
        # Generate summary of old turns
        if self.config.enable_conversation_summary:
            summary_content = self._generate_conversation_summary(turns_to_summarize)
            state.conversation_summary = summary_content
        
        # Keep only recent turns
        state.conversation_turns = recent_turns
        
        # Update turn numbers
        for i, turn in enumerate(state.conversation_turns):
            turn.turn_number = i + 1
        
        return state
    
    def _compress_inactive_files(self, state: ContextState) -> ContextState:
        """Compress inactive project files to architectural summaries"""
        if not self.config.enable_file_compression:
            return state
        
        # Identify active files (mentioned in recent turns)
        active_files = set(state.active_files)
        recent_turns = state.conversation_turns[-3:]  # Check last 3 turns
        
        for turn in recent_turns:
            # Simple heuristic: look for file extensions in content
            import re
            file_mentions = re.findall(r'[\w/]+\.\w+', turn.content)
            for mention in file_mentions:
                if mention in state.project_files:
                    active_files.add(mention)
        
        # CRITICAL FIX: Always preserve source files (files in src/ directories)
        # and other important files that agents commonly need to access
        protected_patterns = [
            '/src/', '//src//', '/source/', '//source//',
            '.c', '.cpp', '.h', '.hpp', '.py', '.js', '.ts', '.java', '.rs',
            '.go', '.php', '.rb', '.cs', '.swift', '.kt', '.scala'
        ]
        
        protected_files = set()
        for filename in state.project_files.keys():
            # Protect source files and files with common source extensions
            if any(pattern in filename for pattern in protected_patterns):
                protected_files.add(filename)
        
        # Only compress non-protected, truly inactive files (documentation, configs, etc.)
        inactive_files = set(state.project_files.keys()) - active_files - protected_files
        
        # Only compress files that are clearly documentation/config files
        compressible_patterns = [
            'README', 'LICENSE', 'CONTRIBUTING', '.md', '.txt', '.yml', '.yaml',
            '.json', '.xml', '.toml', '.ini', '.cfg', '.conf', '.properties'
        ]
        
        truly_inactive_files = set()
        for filename in inactive_files:
            if any(pattern in filename for pattern in compressible_patterns):
                truly_inactive_files.add(filename)
        
        if truly_inactive_files and self.config.enable_architectural_summary:
            architectural_summary = self._generate_architectural_summary(
                {f: state.project_files[f] for f in truly_inactive_files}
            )
            state.architectural_summary = architectural_summary
            
            # Only remove truly inactive documentation/config files
            for filename in truly_inactive_files:
                del state.project_files[filename]
            
            logger.debug(f"Context compression: protected {len(protected_files)} source files, "
                        f"compressed {len(truly_inactive_files)} documentation files")
        
        return state
    
    def _aggressive_truncation(self, state: ContextState) -> ContextState:
        """Aggressive truncation when approaching critical limits"""
        # Keep only last 2 turns + current project state
        if len(state.conversation_turns) > 2:
            state.conversation_turns = state.conversation_turns[-2:]
            
            # Update turn numbers
            for i, turn in enumerate(state.conversation_turns):
                turn.turn_number = i + 1
        
        # Further compress files if needed
        if len(state.project_files) > 3:
            # Keep only 3 most recently mentioned files
            files_to_keep = list(state.project_files.keys())[:3]
            state.project_files = {f: state.project_files[f] for f in files_to_keep}
        
        return state
    
    def _generate_conversation_summary(self, turns: List[ConversationTurn]) -> str:
        """Generate a summary of conversation turns"""
        if not turns:
            return ""
        
        # Simple extractive summary (in production, could use LLM summarization)
        key_points = []
        for turn in turns:
            # Extract first sentence or key phrases
            sentences = turn.content.split('.')
            if sentences:
                key_points.append(f"Turn {turn.turn_number}: {sentences[0][:100]}...")
        
        summary = "CONVERSATION SUMMARY:\n" + "\n".join(key_points[:5])  # Max 5 key points
        return summary
    
    def _generate_architectural_summary(self, files: Dict[str, str]) -> str:
        """Generate architectural summary of files"""
        if not files:
            return ""
        
        summary_parts = ["ARCHITECTURAL SUMMARY:"]
        
        for filename, content in files.items():
            # Extract key information (classes, functions, imports)
            lines = content.split('\n')
            key_lines = []
            
            for line in lines[:20]:  # Check first 20 lines
                line = line.strip()
                if (line.startswith('class ') or 
                    line.startswith('def ') or 
                    line.startswith('import ') or 
                    line.startswith('from ')):
                    key_lines.append(line)
            
            if key_lines:
                summary_parts.append(f"{filename}: {', '.join(key_lines[:3])}")
        
        return "\n".join(summary_parts)


class LifecycleContextManager(BaseContextManager):
    """Active context visibility control with a compact living state.

    This prototype manager is deliberately heuristic. It is meant to expose a
    clean baseline for our experiments, not to be the final learned policy.
    Unlike AdaptiveContextManager, it is active throughout the session and
    treats resolved/stale details as visibility-management targets rather than
    waiting for context overflow.
    """

    PIN_KEYWORDS = (
        "must", "required", "requirement", "do not", "don't", "critical",
        "constraint", "final", "current", "important", "task", "description",
    )
    INVALIDATE_KEYWORDS = (
        "false lead", "unrelated", "invalid", "invalidated", "superseded",
        "do not use", "wrong", "correction", "obsolete", "stale",
    )
    TOOL_NOISE_KEYWORDS = (
        "traceback", "error", "failed", "debug", "log", "warning", "exception",
    )

    def should_compress(self, state: ContextState) -> bool:
        # Lifecycle management is not only budget-triggered; it is a continuous
        # visibility policy. AgentSession special-cases this strategy to pass the
        # managed conversation every turn.
        return True

    def compress_context(self, state: ContextState) -> ContextState:
        tokens_before, usage_before = self.calculate_context_usage(state)

        # Update living state before hiding details.
        state.living_state = self._build_living_state(state)

        pinned, visible, hidden = [], [], list(state.hidden_turns)
        recent_budget = max(self.config.preserve_recent_turns, 3)
        recent_turn_numbers = {t.turn_number for t in state.conversation_turns[-recent_budget:]}

        seen_pins = set()
        seen_hidden = {(t.turn_number, t.role, t.content[:80]) for t in hidden}

        for turn in state.conversation_turns:
            text = turn.content.lower()
            key = (turn.turn_number, turn.role, turn.content[:80])

            is_pin = self._is_pinned(turn)
            is_recent = turn.turn_number in recent_turn_numbers
            is_tool_noise = turn.role == "tool" or (len(turn.content) > 4000 and any(k in text for k in self.TOOL_NOISE_KEYWORDS))
            is_invalidating = any(k in text for k in self.INVALIDATE_KEYWORDS)

            if is_pin and key not in seen_pins:
                turn.metadata = {**turn.metadata, "lifecycle": "pinned"}
                pinned.append(turn)
                seen_pins.add(key)
                continue

            if is_recent or is_invalidating:
                turn.metadata = {**turn.metadata, "lifecycle": "visible"}
                visible.append(turn)
                continue

            if is_tool_noise or len(turn.content) > 2500:
                turn.metadata = {**turn.metadata, "lifecycle": "hidden", "hide_reason": "resolved_or_noisy_detail"}
                if key not in seen_hidden:
                    hidden.append(turn)
                    seen_hidden.add(key)
                continue

            # Older non-pinned text is hidden once a compact state exists.
            if state.living_state:
                turn.metadata = {**turn.metadata, "lifecycle": "hidden", "hide_reason": "projected_to_living_state"}
                if key not in seen_hidden:
                    hidden.append(turn)
                    seen_hidden.add(key)
            else:
                visible.append(turn)

        # Bound pinned section to avoid growth; retain earliest system/task pins and latest constraints.
        state.pinned_turns = self._dedupe_turns(pinned)[-8:]
        state.hidden_turns = self._dedupe_turns(hidden)
        state.conversation_turns = self._dedupe_turns(state.pinned_turns + visible)

        tokens_after, usage_after = self.calculate_context_usage(state)
        self.log_compression(state, "lifecycle_visibility_control", {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "usage_before": usage_before,
            "usage_after": usage_after,
            "visible_turns": len(state.conversation_turns),
            "hidden_turns": len(state.hidden_turns),
            "pinned_turns": len(state.pinned_turns),
        })
        return state

    def _is_pinned(self, turn: ConversationTurn) -> bool:
        if turn.role == "system" and turn.turn_number <= 2:
            return True
        text = turn.content.lower()
        if turn.role == "user" and any(k in text for k in self.PIN_KEYWORDS):
            return True
        return False

    def _build_living_state(self, state: ContextState) -> str:
        if not state.conversation_turns:
            return ""

        goals, current, invalidated, open_loops, files = [], [], [], [], []
        for turn in state.conversation_turns:
            text = turn.content.strip()
            low = text.lower()
            if turn.role == "system" and ("**task**" in low or "**description**" in low):
                goals.append(self._shorten(text, 700))
            elif turn.role == "user" and any(k in low for k in ("task", "implement", "fix", "debug", "analyze", "based on")):
                open_loops.append(self._shorten(text, 300))
            if any(k in low for k in self.INVALIDATE_KEYWORDS):
                invalidated.append(self._shorten(text, 220))
            if any(k in low for k in ("current", "final", "decision", "therefore", "fixed", "resolved", "implemented")):
                current.append(self._shorten(text, 220))
            # File-like mentions for a lightweight project map.
            import re
            for mention in re.findall(r'[\w./-]+\.(?:py|js|ts|java|cpp|c|h|rs|go|php|rb|cs|json|md)', text):
                files.append(mention)

        lines = ["# Living State (current valid project/task state)"]
        if goals:
            lines.append("Goal / task:")
            lines.extend(f"- {g}" for g in goals[-2:])
        if current:
            lines.append("Current valid conclusions / decisions:")
            lines.extend(f"- {c}" for c in current[-5:])
        if invalidated:
            lines.append("Invalidated or stale information to avoid:")
            lines.extend(f"- {i}" for i in invalidated[-5:])
        if open_loops:
            lines.append("Open loops / active instructions:")
            lines.extend(f"- {o}" for o in open_loops[-5:])
        if files:
            unique_files = []
            for f in files:
                if f not in unique_files:
                    unique_files.append(f)
            lines.append("Mentioned files:")
            lines.extend(f"- {f}" for f in unique_files[-10:])
        return "\n".join(lines)

    def _shorten(self, text: str, max_chars: int) -> str:
        text = " ".join(text.split())
        return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."

    def _dedupe_turns(self, turns: List[ConversationTurn]) -> List[ConversationTurn]:
        out, seen = [], set()
        for turn in turns:
            key = (turn.turn_number, turn.role, turn.content[:120])
            if key in seen:
                continue
            seen.add(key)
            out.append(turn)
        return out


class WorkspaceContextManager(BaseContextManager):
    """Experimental block-dashboard context manager.

    This manager bridges LoCoBench's ConversationTurn state with the standalone
    Agentic Context Workspace. It turns turns/tool outputs into addressable
    blocks, injects a dashboard as living_state, and keeps hidden block content
    in an archive while showing only compact metadata in the dashboard.

    A WorkspaceLogger is lazily initialised on the first compress_context() call.
    It writes one JSONL record per turn to:
        <WORKSPACE_LOG_DIR>/<session_id>.jsonl
    where WORKSPACE_LOG_DIR defaults to "workspace_logs/" (relative to cwd) and
    can be overridden via the WORKSPACE_LOG_DIR environment variable.
    Set WORKSPACE_LOG_DIR="" to disable logging entirely.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ws_logger: Optional[WorkspaceLogger] = None

    def should_compress(self, state: ContextState) -> bool:
        return True

    def _get_or_create_logger(self, state: ContextState) -> Optional[WorkspaceLogger]:
        """Lazily create the WorkspaceLogger the first time we need it."""
        if self._ws_logger is not None:
            return self._ws_logger
        log_dir_env = os.environ.get("WORKSPACE_LOG_DIR", "workspace_logs")
        if not log_dir_env:
            return None  # logging disabled
        session_id = state.metadata.get("session_id", "unknown_session")
        self._ws_logger = WorkspaceLogger(
            log_dir=log_dir_env,
            session_id=session_id,
            log_full_snapshot=os.environ.get("WORKSPACE_LOG_FULL_SNAPSHOT", "1") != "0",
        )
        logger.info(f"[WorkspaceContextManager] workspace log → {self._ws_logger.path}")
        return self._ws_logger

    def compress_context(self, state: ContextState) -> ContextState:
        from .context_workspace import ContextWorkspace

        tokens_before, usage_before = self.calculate_context_usage(state)
        workspace = state.metadata.get("workspace_obj")
        if workspace is None:
            workspace = ContextWorkspace(token_budget=self.config.max_context_tokens)
            state.metadata["workspace_obj"] = workspace
            state.metadata["workspace_synced_turns"] = 0

        synced = state.metadata.get("workspace_synced_turns", 0)
        for turn in state.conversation_turns[synced:]:
            self._add_turn_to_workspace(workspace, turn)
        state.metadata["workspace_synced_turns"] = len(state.conversation_turns)

        # Refresh episode summaries then living state so the dashboard reflects
        # what has actually happened this turn before we decide what to hide.
        if workspace.current_episode_id:
            workspace._refresh_episode_summary(workspace.current_episode_id)
        workspace._refresh_living_state()

        # NOTE: We intentionally do NOT auto-hide blocks here.
        # The workspace philosophy is that the agent itself decides what to
        # abstract, hide, or recall by reading the dashboard and issuing
        # explicit context actions. Forcing auto-hide based on token counts
        # is a heuristic that contradicts the agent-driven design and has
        # been shown to hurt performance by silently removing files the agent
        # is actively using.

        state.living_state = workspace.mini_dashboard()
        state.hidden_turns = self._hidden_turns_from_workspace(workspace)
        state.conversation_turns = self._visible_turns_from_workspace(workspace)

        tokens_after, usage_after = self.calculate_context_usage(state)
        self.log_compression(state, "workspace_dashboard", {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "usage_before": usage_before,
            "usage_after": usage_after,
            "workspace_visible_blocks": len(workspace.visible_blocks()),
            "workspace_hidden_blocks": len(workspace.hidden_blocks()),
            # Snapshot the dashboard that was injected into the LLM context this
            # turn. This lets you grep/inspect the result JSON to see exactly what
            # the agent saw without modifying the stored conversation_history.
            "dashboard_snapshot": state.living_state,
        })

        # Write a full turn snapshot to the per-session workspace log.
        ws_logger = self._get_or_create_logger(state)
        if ws_logger is not None:
            ws_logger.log(workspace, state.living_state)

        return state

    def _add_turn_to_workspace(self, workspace, turn: ConversationTurn):
        role = turn.role
        content = turn.content or ""
        source = turn.metadata.get("source", role) if turn.metadata else role

        if role == "user":
            if workspace.current_episode_id is None:
                workspace.start_episode(content)
            else:
                workspace.add_block("user_message", content, source="user", parent_id=workspace.current_episode_id)
        elif role == "assistant":
            workspace.add_block("assistant_message", content, source="assistant", parent_id=workspace.current_episode_id)
        elif role == "tool":
            workspace.add_tool_result(source or "tool", content, parent_id=workspace.current_episode_id, metadata=turn.metadata)
        elif role == "system":
            # The original system prompt is injected verbatim by
            # _get_managed_conversation_history() from agent.conversation_history.
            # Do NOT add it to the workspace to avoid duplication.
            pass
        else:
            workspace.add_block("summary", content, source=role, parent_id=workspace.current_episode_id)

    def _visible_turns_from_workspace(self, workspace) -> List[ConversationTurn]:
        turns = []
        for idx, block in enumerate(workspace.visible_blocks(), 1):
            if block.type == "living_state":
                continue
            role = "user" if block.type in {"episode", "user_message"} else "assistant" if block.type == "assistant_message" else "system"
            content = block.content
            # If the agent has already abstracted this block, show the summary
            # rather than the raw content to reduce noise — this is the agent's
            # own decision, not a system-imposed truncation.
            if block.summary and block.status != "pinned":
                content = f"[{block.id} {block.type} — agent summary]\n{block.summary}"
            turns.append(ConversationTurn(
                turn_number=idx,
                role=role,
                content=f"<{block.id} type={block.type} source={block.source} status={block.status}>\n{content}\n</{block.id}>",
                timestamp=datetime.now(),
                token_count=self.count_tokens(content),
                metadata={"workspace_block_id": block.id, "workspace_type": block.type},
            ))
        return turns

    def _hidden_turns_from_workspace(self, workspace) -> List[ConversationTurn]:
        turns = []
        for idx, block in enumerate(workspace.hidden_blocks(), 1):
            turns.append(ConversationTurn(
                turn_number=idx,
                role="system",
                content=f"{block.id}: {block.type}, source={block.source}, tokens={block.tokens}, summary={block.summary or 'n/a'}",
                timestamp=datetime.now(),
                token_count=max(1, self.count_tokens(block.summary or block.source or block.type)),
                metadata={"workspace_block_id": block.id, "lifecycle": "hidden", "hide_reason": block.metadata.get("hide_reason", "workspace_hidden")},
            ))
        return turns


def create_context_manager(config: ContextManagementConfig) -> BaseContextManager:
    """Factory function to create appropriate context manager"""
    if config.strategy == ContextManagementStrategy.NONE:
        return NoContextManager(config)
    elif config.strategy == ContextManagementStrategy.BASIC:
        return BasicContextManager(config)
    elif config.strategy == ContextManagementStrategy.ADAPTIVE:
        return AdaptiveContextManager(config)
    elif config.strategy == ContextManagementStrategy.LIFECYCLE:
        return LifecycleContextManager(config)
    elif config.strategy == ContextManagementStrategy.WORKSPACE:
        return WorkspaceContextManager(config)
    else:
        raise ValueError(f"Unknown context management strategy: {config.strategy}")
