"""
Context Workspace Tool — lets the agent actively manage its working context.

The agent can ABSTRACT key facts from a block, HIDE it to the archive,
ASK_ARCHIVE for a specific fact without restoring the block, SHOW_BLOCK
for a peek, and RESTORE when it needs the raw content back in view.

These are the same actions documented in the workspace dashboard. Exposing
them as real tool functions means the agent can call them the same way it
calls read_file or list_directory, which is how agents actually behave.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..core.tool_registry import Tool, ToolCategory, ToolFunction, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class ContextWorkspaceTool(Tool):
    """Manage your context workspace: archive what you've finished with,
    retrieve specific facts, bring content back when you need it again."""

    def __init__(self) -> None:
        super().__init__(
            name="context_workspace",
            description=(
                "Manage your working context. Use this to keep your workspace "
                "focused: archive blocks you've finished reading, retrieve "
                "specific facts without restoring full content, or bring a "
                "block back when you need the raw detail again."
            ),
            category=ToolCategory.SYSTEM,
        )
        self._workspace = None  # injected by AgentSession after workspace is created

    def set_workspace(self, workspace: Any) -> None:
        """Called by AgentSession to link this tool to the live workspace."""
        self._workspace = workspace

    # ------------------------------------------------------------------
    # Tool functions
    # ------------------------------------------------------------------

    @tool_function(
        description=(
            "Record the key facts you have extracted from a block as a short "
            "summary note. Do this when you have understood the content and "
            "want a durable record without keeping the raw detail in view. "
            "A good summary captures function names, data structures, "
            "decisions, or error causes — whatever matters for your task.\n\n"
            "Example: after reading a large C source file, summarize with "
            "'exports handle_request(); uses epoll for async I/O; no auth at "
            "bus layer'. Then HIDE the block to free up space."
        ),
        parameters=[
            ToolParameter(
                "block_id",
                "string",
                "The block ID shown in the context dashboard, e.g. 'B8' or 'E3'.",
            ),
            ToolParameter(
                "summary",
                "string",
                "Concise note capturing the facts that matter for your task. "
                "Keep it short — a sentence or two is usually enough.",
            ),
        ],
        returns="Confirmation that the summary was recorded on the block.",
        category=ToolCategory.SYSTEM,
    )
    def abstract_block(self, block_id: str, summary: str) -> Dict[str, Any]:
        if self._workspace is None:
            return {"success": False, "error": "Workspace not available."}
        try:
            from ..core.context_workspace import ContextAction, ActionType
            self._workspace.apply_action(
                ContextAction(action=ActionType.ABSTRACT, target=block_id, content=summary)
            )
            return {"success": True, "result": f"Summary recorded on {block_id}."}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @tool_function(
        description=(
            "Move a block to the archive so it no longer takes up space in "
            "your active context. The full content is preserved and you can "
            "retrieve it at any time with ask_archive or restore_block.\n\n"
            "Best practice: ABSTRACT the key facts first, then HIDE. That way "
            "your summary survives even if you never look at the raw block again.\n\n"
            "Good candidates for hiding: files you have fully read and understood, "
            "tool outputs from earlier phases, large search results you have "
            "already acted on."
        ),
        parameters=[
            ToolParameter(
                "block_id",
                "string",
                "The block ID shown in the context dashboard, e.g. 'B8'.",
            ),
            ToolParameter(
                "reason",
                "string",
                "Brief reason for archiving — helps you remember later. "
                "E.g. 'file read complete, key facts abstracted'.",
                required=False,
                default="",
            ),
        ],
        returns="Confirmation that the block was moved to the archive.",
        category=ToolCategory.SYSTEM,
    )
    def hide_block(self, block_id: str, reason: str = "") -> Dict[str, Any]:
        if self._workspace is None:
            return {"success": False, "error": "Workspace not available."}
        try:
            from ..core.context_workspace import ContextAction, ActionType
            self._workspace.apply_action(
                ContextAction(action=ActionType.HIDE, target=block_id, reason=reason)
            )
            return {"success": True, "result": f"{block_id} moved to archive."}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @tool_function(
        description=(
            "Ask a question about archived content without bringing the whole "
            "block back into your active view. Use this when you need one "
            "specific fact — a function name, a config value, an error message "
            "— without restoring the noise around it.\n\n"
            "Example: 'What was the epoll timeout value in event_bus.c?' "
            "returns just that answer rather than the full 500-line file."
        ),
        parameters=[
            ToolParameter(
                "query",
                "string",
                "The specific question you want answered from archived content.",
            ),
        ],
        returns="The answer extracted from the archive, or a note that nothing matched.",
        category=ToolCategory.SYSTEM,
    )
    def ask_archive(self, query: str) -> Dict[str, Any]:
        if self._workspace is None:
            return {"success": False, "error": "Workspace not available."}
        try:
            from ..core.context_workspace import ContextAction, ActionType
            results = self._workspace.apply_action(
                ContextAction(action=ActionType.ASK_ARCHIVE, query=query)
            )
            return {"success": True, "result": results}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @tool_function(
        description=(
            "Peek at an archived block's raw content without changing its "
            "archived status. Useful when you want to verify a detail before "
            "deciding whether to restore the block fully."
        ),
        parameters=[
            ToolParameter(
                "block_id",
                "string",
                "The block ID to peek at, e.g. 'B8'.",
            ),
        ],
        returns="The raw content of the archived block.",
        category=ToolCategory.SYSTEM,
    )
    def show_block(self, block_id: str) -> Dict[str, Any]:
        if self._workspace is None:
            return {"success": False, "error": "Workspace not available."}
        try:
            from ..core.context_workspace import ContextAction, ActionType
            result = self._workspace.apply_action(
                ContextAction(action=ActionType.SHOW_BLOCK, target=block_id)
            )
            return {"success": True, "result": result}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @tool_function(
        description=(
            "Bring an archived block back into your active workspace. Use this "
            "when you need to work with the raw content again for an extended "
            "period — for example, you are now writing code that directly "
            "depends on a file you previously archived.\n\n"
            "If you only need a single fact, prefer ask_archive instead — it "
            "answers the question without adding the full block back."
        ),
        parameters=[
            ToolParameter(
                "block_id",
                "string",
                "The block ID to restore, e.g. 'B8'.",
            ),
        ],
        returns="Confirmation that the block is visible in your workspace again.",
        category=ToolCategory.SYSTEM,
    )
    def restore_block(self, block_id: str) -> Dict[str, Any]:
        if self._workspace is None:
            return {"success": False, "error": "Workspace not available."}
        try:
            from ..core.context_workspace import ContextAction, ActionType
            self._workspace.apply_action(
                ContextAction(action=ActionType.RESTORE, target=block_id)
            )
            return {"success": True, "result": f"{block_id} restored to active workspace."}
        except Exception as exc:
            return {"success": False, "error": str(exc)}
