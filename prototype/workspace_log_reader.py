#!/usr/bin/env python3
"""
Workspace Log Reader — inspect the per-turn workspace snapshots written by
WorkspaceLogger during a LoCoBench evaluation run.

Usage examples
--------------
# List all sessions (log files) in the default log directory:
    python3 workspace_log_reader.py list

# Show the dashboard injected at each turn for a specific session:
    python3 workspace_log_reader.py show <session_id_or_path> --mode dashboard

# Show the visible + hidden blocks at every turn:
    python3 workspace_log_reader.py show <session_id_or_path> --mode blocks

# Show the full workspace snapshot at turn N:
    python3 workspace_log_reader.py show <session_id_or_path> --mode snapshot --turn 5

# Replay the full workspace from a snapshot:
    python3 workspace_log_reader.py replay <session_id_or_path> --turn 5

All log files live under WORKSPACE_LOG_DIR (default: workspace_logs/).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = Path("workspace_logs")


def _resolve_log_path(session_id_or_path: str, log_dir: Path) -> Path:
    p = Path(session_id_or_path)
    if p.exists():
        return p
    # Try exact match in log_dir
    direct = log_dir / f"{session_id_or_path}.jsonl"
    if direct.exists():
        return direct
    # Fuzzy: find first file whose stem contains the given string
    for f in log_dir.glob("*.jsonl"):
        if session_id_or_path in f.stem:
            return f
    raise FileNotFoundError(
        f"No log file found for '{session_id_or_path}' in {log_dir}"
    )


def _load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cmd_list(log_dir: Path) -> None:
    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        print(f"No log files found in {log_dir}")
        return
    print(f"{'FILE':<60}  {'TURNS':>5}  {'SESSION_ID'}")
    print("-" * 100)
    for f in files:
        try:
            records = _load_records(f)
            session_id = records[0].get("session_id", "?") if records else "?"
            print(f"{f.name:<60}  {len(records):>5}  {session_id}")
        except Exception as e:
            print(f"{f.name:<60}  ERROR: {e}")


def cmd_show(path: Path, mode: str, turn: Optional[int]) -> None:
    records = _load_records(path)
    if not records:
        print("Log file is empty.")
        return

    to_show = records if turn is None else [r for r in records if r["turn_index"] == turn]
    if not to_show:
        print(f"No record found for turn {turn}. Available: 0–{len(records)-1}")
        return

    for r in to_show:
        t = r["turn_index"]
        ts = r.get("timestamp", "")
        vtok = r.get("visible_tokens", 0)
        budget = r.get("token_budget", "?")
        print(f"\n{'='*70}")
        print(f"Turn {t}  |  {ts}  |  visible_tokens={vtok}/{budget}")
        print("=" * 70)

        if mode == "dashboard":
            print(r.get("dashboard", "(no dashboard)"))

        elif mode == "blocks":
            vis = r.get("visible_blocks", [])
            hid = r.get("hidden_blocks", [])
            print(f"VISIBLE ({len(vis)} blocks):")
            for b in vis:
                summ = f" — {b['summary']}" if b.get("summary") else ""
                print(f"  {b['id']:6s} {b['type']:20s} {b['tokens']:5d} tok  {b['source']}{summ}")
            if hid:
                print(f"\nHIDDEN ({len(hid)} blocks):")
                for b in hid:
                    reason = f" [{b.get('hide_reason','')}]" if b.get("hide_reason") else ""
                    summ = f" — {b['summary']}" if b.get("summary") else ""
                    print(f"  {b['id']:6s} {b['type']:20s} {b['tokens']:5d} tok  {b['source']}{reason}{summ}")

        elif mode == "snapshot":
            snap = r.get("workspace_snapshot")
            if snap is None:
                print("No snapshot stored (run with WORKSPACE_LOG_FULL_SNAPSHOT=1)")
            else:
                print(json.dumps(snap, indent=2, ensure_ascii=False)[:8000])


def cmd_replay(path: Path, turn: int) -> None:
    """Load and print a full workspace from snapshot, then print its full dashboard."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "LoCoBench-Agent"))
    from locobench.core.context_workspace import ContextWorkspace

    records = _load_records(path)
    matches = [r for r in records if r["turn_index"] == turn]
    if not matches:
        print(f"No record for turn {turn}.")
        return
    snap = matches[0].get("workspace_snapshot")
    if snap is None:
        print("No snapshot stored (run with WORKSPACE_LOG_FULL_SNAPSHOT=1)")
        return
    ws = ContextWorkspace.from_dict(snap)
    print(ws.full_dashboard())


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR),
                        help="Directory containing .jsonl log files")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List all session log files")

    p_show = sub.add_parser("show", help="Show turn data for a session")
    p_show.add_argument("session", help="Session id or path to .jsonl file")
    p_show.add_argument("--mode", choices=["dashboard", "blocks", "snapshot"],
                        default="dashboard")
    p_show.add_argument("--turn", type=int, default=None,
                        help="Show only this turn (0-indexed); default=all")

    p_replay = sub.add_parser("replay", help="Replay workspace from a snapshot turn")
    p_replay.add_argument("session", help="Session id or path to .jsonl file")
    p_replay.add_argument("--turn", type=int, required=True)

    args = parser.parse_args()
    log_dir = Path(args.log_dir)

    if args.cmd == "list":
        cmd_list(log_dir)
    elif args.cmd == "show":
        path = _resolve_log_path(args.session, log_dir)
        cmd_show(path, args.mode, args.turn)
    elif args.cmd == "replay":
        path = _resolve_log_path(args.session, log_dir)
        cmd_replay(path, args.turn)


if __name__ == "__main__":
    main()
