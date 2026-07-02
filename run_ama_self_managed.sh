#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f ./.env ]]; then
  set -a
  . ./.env
  set +a
fi

cd benchmarks/AMA-Bench
exec bash scripts/run_full_sms_gemini.sh "$@"

