#!/bin/bash
# Spend guard for a long indexing run, watching from outside the app. Clears the WAITING queue (running jobs
# finish; nothing is deleted) if the spend passes MAX_USD, if the last window cost more than MAX_PER_JOB per
# finished job, or if MAX_NEW_FAILS files fail.
#
# The figure it reads is the first dollar amount on the `cost:` line of `broll status`. Since 13 Oct 2026 that
# is REAL money: the sum of what each model call was billed (usage_metadata from Google, written to the
# api_usage table), counted from when the ledger began. Before that the line showed a formula that ran about
# 5x low, so a ceiling set against it protected nothing.
#
# This is the second line of defence. The first is inside the app: the worker stops claiming files once the
# real spend of a run passes ingest.max_run_usd, or of the last 24 hours passes ingest.max_day_usd
# (config.yaml; the defaults are 15 and 20).
#
# Real cost per file is about $0.003 for a photo and $0.008 for a video, so 0.03 a job means something is wrong.
MAX_USD=${MAX_USD:-15}
MAX_PER_JOB=${MAX_PER_JOB:-0.03}
LOG=/tmp/broll-guard.log
cd /opt/broll-librarian || exit 1
prev_spent=""; prev_done=""; base_failed=""; tick=0; seen_work=0
echo $$ > /tmp/broll-guard.pid
while true; do
  out=$(docker compose exec -T librarian broll status 2>&1)
  spent=$(echo "$out" | grep -oP 'cost:\s+\$\K[0-9.]+')
  done_n=$(echo "$out" | grep -oP 'done=\K[0-9]+')
  queued=$(echo "$out" | grep -oP 'queued=\K[0-9]+')
  failed_n=$(echo "$out" | grep -oP 'failed=\K[0-9]+')
  if [ -n "$spent" ] && [ -n "$done_n" ]; then
    trip=""
    if awk "BEGIN{exit !($spent >= $MAX_USD)}"; then trip="total \$$spent passed the \$$MAX_USD ceiling"; fi
    if [ -n "$prev_spent" ] && [ $((done_n - prev_done)) -ge 40 ]; then
      per=$(awk "BEGIN{printf \"%.4f\", ($spent - $prev_spent) / ($done_n - $prev_done)}")
      if awk "BEGIN{exit !($per > $MAX_PER_JOB)}"; then trip="cost per job \$$per over \$$MAX_PER_JOB in the last window"; fi
    fi
    if [ -n "$failed_n" ]; then
      if [ -z "$base_failed" ] || [ "$failed_n" -lt "$base_failed" ]; then base_failed=$failed_n; fi
      if [ $((failed_n - base_failed)) -ge ${MAX_NEW_FAILS:-15} ]; then trip="$((failed_n - base_failed)) new failures since the guard started (Drive quota or an outage?)"; fi
    fi
    if [ -z "$prev_done" ] || [ $((done_n - prev_done)) -ge 40 ]; then prev_spent=$spent; prev_done=$done_n; fi
    if [ -n "$trip" ]; then
      docker compose exec -T librarian python -c "
from broll.config import load_workspace_config
from broll.db.store import Store
from broll.library_admin import clear_queue
c=load_workspace_config('library'); s=Store.for_config(c)
r=clear_queue(c,s); print('cleared', r.cancelled, 'waiting jobs;', r.still_running, 'still running')" >> $LOG 2>&1
      echo "$(date -u +%H:%M) GUARD TRIPPED: $trip. Queue cleared. spent=\$$spent done=$done_n" >> $LOG
      exit 0
    fi
    tick=$((tick+1))
    if [ $((tick % 15)) -eq 0 ]; then echo "$(date -u +%H:%M) guard ok: spent=\$$spent done=$done_n queued=$queued (ceiling \$$MAX_USD)" >> $LOG; fi
    if [ "${queued:-0}" != "0" ] || ! echo "$out" | grep -q "running=0"; then seen_work=1; fi
    if [ "$seen_work" = "1" ] && [ "$queued" = "0" ] && echo "$out" | grep -q "running=0"; then echo "$(date -u +%H:%M) queue finished: spent=\$$spent done=$done_n" >> $LOG; exit 0; fi
  fi
  sleep 120
done
