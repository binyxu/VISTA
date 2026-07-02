#!/usr/bin/env python3
"""Smoke tests for Agentic Context Workspace."""

from __future__ import annotations

from context_workspace import ContextWorkspace, ContextActionError


def test_hide_without_abstract() -> None:
    """Agent can HIDE a block directly without ABSTRACT — no forced sequence."""
    ws = ContextWorkspace(token_budget=10_000)
    ep = ws.start_episode("Debug request logging")
    block_id = ws.add_tool_result("read_file", "log_request(req);\n" + "detail\n" * 1000, parent_id=ep)

    # Should succeed even without an ABSTRACT step first
    ws.hide(block_id, "no longer needed")
    assert ws.blocks[block_id].status == "hidden"
    assert block_id not in [b.id for b in ws.visible_blocks()]


def test_abstract_then_hide() -> None:
    """Agent can ABSTRACT then HIDE — still works as before."""
    ws = ContextWorkspace(token_budget=10_000)
    ep = ws.start_episode("Debug request logging")
    block_id = ws.add_tool_result("read_file", "log_request(req);\n" + "detail\n" * 1000, parent_id=ep)

    ws.abstract(block_id, "The file calls log_request(req) after request handling.")
    ws.hide(block_id, "summarized")
    assert ws.blocks[block_id].status == "hidden"
    assert block_id not in [b.id for b in ws.visible_blocks()]

    # ASK_ARCHIVE returns the summary without restoring
    result = ws.ask_archive(query="log_request")
    assert result["answers"][0]["block_id"] == block_id
    assert ws.blocks[block_id].status == "hidden"

    # SHOW_BLOCK lets you peek without restoring
    shown = ws.show_block(block_id, max_tokens=10)
    assert "log_request" in shown["block"]["content"]
    assert ws.blocks[block_id].status == "hidden"

    # RESTORE makes it visible again
    restored = ws.restore(block_id)
    assert restored["status"] == "visible"
    assert ws.blocks[block_id].status == "visible"


def test_dashboard_metadata() -> None:
    ws = ContextWorkspace(token_budget=100)
    ep = ws.start_episode("Inspect logger")
    b = ws.add_tool_result("read_file", "logger implementation" * 100, parent_id=ep)
    ws.abstract(b, "Logger implementation details.")
    ws.hide(b, "not active")
    dashboard = ws.full_dashboard()
    assert b in dashboard
    assert "Logger implementation details" in dashboard
    assert "Visible" in dashboard


def test_no_auto_summary_on_add() -> None:
    """add_tool_result should NOT auto-generate summaries."""
    ws = ContextWorkspace(token_budget=10_000)
    ep = ws.start_episode("Test")
    block_id = ws.add_tool_result("read_file", "content " * 500, parent_id=ep)
    assert ws.blocks[block_id].summary == "", \
        "Block should have no summary until agent explicitly calls ABSTRACT"


def main() -> None:
    test_hide_without_abstract()
    test_abstract_then_hide()
    test_dashboard_metadata()
    test_no_auto_summary_on_add()
    print("context_workspace smoke tests passed")


if __name__ == "__main__":
    main()
