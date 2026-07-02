#!/usr/bin/env bash
cd ${BROWSECOMP_ROOT:-.}
# Wait for main sweep to finish (DONE marker) or its process gone, up to 90 min.
for i in $(seq 1 540); do
  if grep -q "DONE RUN_ID=" runs/_sweep_main.log 2>/dev/null; then
    echo "[chain] main sweep DONE detected after ${i}0s; starting model sweep" >> runs/_chain.log
    break
  fi
  sleep 10
done
# give MCP a moment to release memory
sleep 5
./run_model_sweep.sh >> runs/_chain.log 2>&1
echo "[chain] model sweep finished" >> runs/_chain.log
