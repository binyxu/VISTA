#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f ./.env ]]; then
  set -a
  . ./.env
  set +a
fi

cd benchmarks/SWE-Bench
exec bash methods/self_managed_context/run_strict_lc_better_dashboard.sh "$@"

