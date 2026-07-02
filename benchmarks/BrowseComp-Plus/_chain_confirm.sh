#!/usr/bin/env bash
cd ${BROWSECOMP_ROOT:-.}
export W=12288 B=163840 PORT=8086
STAMP="$(date +%Y%m%d_%H%M%S)"
wait_port_free(){ for _ in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -n -P >/dev/null 2>&1 || return 0; sleep 3; done; }

# 1) fill missing models on the 8-query subset
for M in "gpt-5.4-mini" "deepseek-v4-flash-qcloud"; do
  wait_port_free
  SAFE="$(echo "$M" | tr '/.' '__')"
  MODEL="$M" RUN_ID="modelsweep_${SAFE}_${STAMP}" QUERIES="topics-qrels/queries_model8.tsv" THREADS=4 \
    ./run_wb_sweep.sh || echo "WARN $M nonzero exit"
done

# 2) replicate the headline W=12288 subset30 result (deepseek-v4-pro)
wait_port_free
MODEL="deepseek-v4-pro-qcloud" RUN_ID="wb12288_b163840_subset30_REP_${STAMP}" \
  QUERIES="topics-qrels/queries_subset30.tsv" THREADS=5 \
  ./run_wb_sweep.sh || echo "WARN replication nonzero exit"

echo "CONFIRM_CHAIN_DONE stamp=$STAMP"
