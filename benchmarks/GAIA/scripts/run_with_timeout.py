#!/usr/bin/env python3
"""Run a command with a wall-clock timeout and kill its process group."""

import os
import signal
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: run_with_timeout.py SECONDS COMMAND [ARGS...]", file=sys.stderr)
        return 2
    seconds = float(sys.argv[1])
    cmd = sys.argv[2:]
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        return proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        print(f"WALL_TIMEOUT after {seconds:g}s: {' '.join(cmd)}", file=sys.stderr)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
