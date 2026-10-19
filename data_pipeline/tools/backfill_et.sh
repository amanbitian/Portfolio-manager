#!/bin/bash
# Resumable Economic Times full-text backfill, newest month first.
#   RPS=18 WORKERS=22 bash data_pipeline/tools/backfill_et.sh
# Already-stored articles are skipped, so re-running never duplicates.
# Exit codes from one month: 0 done | 3 throttle/block brake (after the 403 retry schedule ran out)
#                            | 4 stalled (nothing finished for FULLTEXT_STALL_SECONDS): retried here.
RPS="${RPS:-8}"
WORKERS="${WORKERS:-10}"
MAX_STALL_RETRIES="${MAX_STALL_RETRIES:-3}"
cd "$(dirname "$0")/../.." || exit 1
for y in $(seq 2026 -1 2016); do
  if [ "$y" = 2026 ]; then months="10 9 8 7 6 5 4 3 2 1"; else months="12 11 10 9 8 7 6 5 4 3 2 1"; fi
  for m in $months; do
    printf -v mm "%02d" $m
    stalls=0
    while true; do
      echo "$(date '+%F %T') START $y-$mm rps=$RPS workers=$WORKERS"
      py -3.14 -m data_pipeline.run news-fulltext --source et --month "$y-$mm" --rps "$RPS" --workers "$WORKERS" 2>&1 | grep -v "it/s\|art/s"
      rc=${PIPESTATUS[0]}
      echo "$(date '+%F %T') END $y-$mm rc=$rc"
      if [ "$rc" = 4 ] && [ "$stalls" -lt "$MAX_STALL_RETRIES" ]; then
        stalls=$((stalls + 1))
        echo "$(date '+%F %T') STALL in $y-$mm: retrying the month ($stalls of $MAX_STALL_RETRIES)"
        continue
      fi
      break
    done
    if [ "$rc" != 0 ]; then echo "$(date '+%F %T') STOPPED rc=$rc (3 = brake/block, 4 = stalled too many times)"; exit "$rc"; fi
  done
done
echo "$(date '+%F %T') ALL DONE"
