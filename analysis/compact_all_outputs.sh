#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT:-${SCRIPT_DIR}/compact_loca_output.py}"

if [[ -n "${COMPACT_ROOTS:-}" ]]; then
  IFS=: read -r -a ROOTS <<< "$COMPACT_ROOTS"
else
  ROOTS=(
    "${LOCA_OUTPUT_DIR:-./outputs}"
  )
fi

MODE="${MODE:-debug-reconstructable}"
PROGRESS="${PROGRESS:-5000}"
FORCE="${FORCE:---force}"
DRY_RUN="${DRY_RUN:-0}"
DELETE_ORIGINAL="${DELETE_ORIGINAL:-1}"
REQUIRE_RESULTS="${REQUIRE_RESULTS:-1}"
MIN_AGE_SECONDS="${MIN_AGE_SECONDS:-3600}"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Missing compaction script: $SCRIPT" >&2
  exit 1
fi

echo "Compaction script: $SCRIPT"
echo "Mode: $MODE"
echo "Progress interval: $PROGRESS"
echo "Force flag: ${FORCE:-<disabled>}"
echo "Dry run: $DRY_RUN"
echo "Delete original after successful compaction: $DELETE_ORIGINAL"
echo "Require top-level results.json: $REQUIRE_RESULTS"
echo "Minimum directory age since last modification: ${MIN_AGE_SECONDS}s"
echo

for root in "${ROOTS[@]}"; do
  if [[ ! -d "$root" ]]; then
    echo "Skipping missing root: $root" >&2
    continue
  fi

  echo "== Root: $root =="
  while IFS= read -r -d '' dir; do
    out="${dir}.compact_debug_reconstructable"

    if [[ "$dir" == *.compact_debug_reconstructable ]]; then
      echo "Skip compact output: $dir"
      continue
    fi

    if [[ -d "$out" && "$FORCE" != "--force" ]]; then
      echo "Skip existing output: $out"
      continue
    fi

    if [[ "$REQUIRE_RESULTS" == "1" && ! -f "$dir/results.json" ]]; then
      echo "Skip incomplete/no-results directory: $dir"
      continue
    fi

    if [[ "$MIN_AGE_SECONDS" != "0" ]]; then
      now_ts="$(date +%s)"
      dir_mtime="$(python3 - "$dir" <<'PY'
import os
import sys

root = sys.argv[1]
latest = os.path.getmtime(root)
for current_root, dirnames, filenames in os.walk(root):
    for name in dirnames + filenames:
        path = os.path.join(current_root, name)
        try:
            latest = max(latest, os.path.getmtime(path))
        except OSError:
            pass
print(int(latest))
PY
)"
      dir_age="$((now_ts - dir_mtime))"
      if (( dir_age < MIN_AGE_SECONDS )); then
        echo "Skip recently modified directory (${dir_age}s old < ${MIN_AGE_SECONDS}s): $dir"
        continue
      fi
    fi

    echo "Compressing:"
    echo "  input : $dir"
    echo "  output: $out"

    if [[ "$DRY_RUN" == "1" ]]; then
      python3 "$SCRIPT" "$dir" "$out" --mode "$MODE" --dry-run --progress "$PROGRESS"
    else
      python3 "$SCRIPT" "$dir" "$out" --mode "$MODE" $FORCE --progress "$PROGRESS"
      if [[ "$DELETE_ORIGINAL" == "1" ]]; then
        if [[ -f "$out/summary.json" && -f "$out/manifest.jsonl" ]]; then
          echo "Deleting original after successful compaction:"
          echo "  $dir"
          rm -rf -- "$dir"
        else
          echo "Refusing to delete original; compact output is missing summary.json or manifest.jsonl: $out" >&2
          exit 1
        fi
      fi
    fi
    echo
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done

echo "Done."
