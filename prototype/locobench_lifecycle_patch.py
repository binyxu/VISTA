#!/usr/bin/env python3
"""Prototype patch helper for LoCoBench-Agent lifecycle context integration.

This script does not modify LoCoBench-Agent automatically. It prints the concrete
files and code locations that need to be patched. Use it as a checklist before
making invasive changes to the external repository.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCO = ROOT / "external" / "LoCoBench-Agent"

CHECKS = [
    (
        LOCO / "locobench/core/context_management.py",
        [
            "class ContextManagementStrategy",
            "class ContextState",
            "class AdaptiveContextManager",
            "def create_context_manager",
        ],
    ),
    (
        LOCO / "locobench/core/agent_session.py",
        [
            "def _initialize_context_manager",
            "def _add_conversation_turn",
            "def _get_managed_conversation_history",
            "use_context_management = usage_pct >",
        ],
    ),
    (
        LOCO / "locobench/cli.py",
        [
            "--context-management",
            "click.Choice(['none', 'basic', 'adaptive'])",
        ],
    ),
]


def main():
    print("LoCoBench-Agent lifecycle integration checklist")
    print(f"Repo: {LOCO}")
    for path, needles in CHECKS:
        print(f"\n{path.relative_to(ROOT)}")
        text = path.read_text() if path.exists() else ""
        for needle in needles:
            print(f"  [{'OK' if needle in text else 'MISSING'}] {needle}")

    print("""
Recommended first patch:
1. Add ContextManagementStrategy.LIFECYCLE = "lifecycle".
2. Add ContextState.living_state, hidden_turns, pinned_turns.
3. Add LifecycleContextManager.
4. Make lifecycle context management always active in AgentSession._execute_phase.
5. Add 'lifecycle' to CLI --context-management choices.
6. Run official smoke eval with --context-management lifecycle.
""")


if __name__ == "__main__":
    main()
