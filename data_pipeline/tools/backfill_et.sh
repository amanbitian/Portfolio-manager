#!/bin/bash
# Resumable Economic Times full-text backfill, newest month first.
#   RPS=16 WORKERS=20 bash data_pipeline/tools/backfill_et.sh
# Stops (exit 3) at the first throttle brake; re-run with a lower RPS to resume.
# Already-stored articles are skipped, so re-running never duplicates.
RPS="${RPS:-8}"
WORKERS="${WORKERS:-10}"
cd "$(dirname "$0")/../.." || exit 1
for y in $(seq 2026 -1 2016); do
  if [ "$y" = 2026 ]; then months="10 9 8 7 6 5 4 3 2 1"; else months="12 11 10 9 8 7 6 5 4 3 2 1"; fi
  for m in $months; do
    printf -v mm "%02d" $m
    echo "$(date '+%F %T') START $y-$mm rps=$RPS workers=$WORKERS"
    py -3.14 -m data_pipeline.run news-fulltext --source et --month "$y-$mm" --rps "$RPS" --workers "$WORKERS" 2>&1 | grep -v "it/s\|art/s"
    rc=${PIPESTATUS[0]}
    echo "$(date '+%F %T') END $y-$mm rc=$rc"
    if [ "$rc" != 0 ]; then echo "$(date '+%F %T') STOPPED rc=$rc (brake=3, other errors e.g. HTTP 403)"; exit "$rc"; fi
  done
done
echo "$(date '+%F %T') ALL DONE"
