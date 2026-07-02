#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f ./.env ]]; then
  set -a
  . ./.env
  set +a
fi

cd benchmarks/BrowseComp-Plus
exec bash run_wb_sweep.sh "$@"

