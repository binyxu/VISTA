"""Regression checks for the standalone VISTA release layout."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


class _FastMCPStub:
    """Minimal decorator surface needed to import the context-tool server."""

    def __init__(self, _name: str):
        pass

    def tool(self):
        return lambda function: function


def test_ama_adapter_imports_published_loca_core(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace(FastMCP=_FastMCPStub))

    module = importlib.import_module("src.method.self_managed_agentic")
    assert module.LOCA_WORKSPACE_DIR.is_dir()
    assert module.LOCA_WORKSPACE_DIR.parts[-6:] == (
        "benchmarks",
        "LOCAbench",
        "gem",
        "tools",
        "mcp_server",
        "context_workspace",
    )

    manager = module.WorkspaceManager(tmp_path / "workspace", token_budget=4096)
    manager.register_message({"role": "user", "content": "task"}, 0)
    manager.register_message({"role": "assistant", "content": "exact evidence"}, 1)

    method = module.SelfManagedAgenticMethod()
    result = method._archive(manager, "B2", "evidence index", "exact evidence")

    assert result.startswith("[ARCHIVED:B2")
    block = manager.get_state()["blocks"]["B2"]
    assert block["status"] == "compressed"
    assert block["summary"] == "evidence index\nCritical extract: exact evidence"
    payload_path = manager.workspace_dir / block["payload_path"]
    assert payload_path.read_text(encoding="utf-8") == "exact evidence"
