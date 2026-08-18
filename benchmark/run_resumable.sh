#!/usr/bin/env bash
# Run one or more evaluate.py stages under a supervisor that restarts them.
#
# evaluate.py journals each scored item, so a stage that dies resumes from where
# it stopped. What this adds is the retry itself, plus a stall check: if a retry
# scores nothing new, retrying again will not help either.
#
#   ./run_resumable.sh qwen3-dapt:v1 dapt-probe:probe dapt-t2-closed:2
set -uo pipefail

MODEL="${1:?usage: run_resumable.sh MODEL TAG:TRACK [TAG:TRACK ...]}"
shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$HERE/data/runs/.ckpt"
LOG_DIR="${LOG_DIR:-$HERE/../training}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
MAX_STALLS="${MAX_STALLS:-3}"
OLLAMA_WAIT="${OLLAMA_WAIT:-1800}"   # seconds to wait for a downed server

say() { echo "$(date '+%F %T') supervisor: $*"; }

journalled() { grep -c '"id"' "$1" 2>/dev/null || echo 0; }

wait_for_ollama() {
    local waited=0
    while ! curl -sf -m 5 http://localhost:11434/api/tags >/dev/null 2>&1; do
        if [ "$waited" = 0 ]; then
            # only works where sudo is configured to allow it; harmless otherwise
            sudo -n systemctl start ollama >/dev/null 2>&1
            say "ollama is not answering; waiting up to ${OLLAMA_WAIT}s."
            say "  start it with: sudo systemctl start ollama"
        fi
        sleep 15
        waited=$((waited + 15))
        if [ "$waited" -ge "$OLLAMA_WAIT" ]; then
            say "ollama still down after ${waited}s, giving up"
            return 1
        fi
    done
    [ "$waited" -gt 0 ] && say "ollama is back after ${waited}s"
    return 0
}

for spec in "$@"; do
    TAG="${spec%%:*}"
    TRACK="${spec#*:}"
    CKPT="$CKPT_DIR/$TAG-$TRACK.jsonl"
    LOG="$LOG_DIR/eval_$TAG.log"
    OUT="$HERE/data/runs/$TAG.json"

    if [ -f "$OUT" ]; then
        say "$TAG already scored ($OUT), skipping"
        continue
    fi

    stalls=0
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        wait_for_ollama || exit 1
        before=$(journalled "$CKPT")
        say "$TAG track $TRACK: attempt $attempt/$MAX_ATTEMPTS (journalled: $before)"
        ( cd "$HERE" && python cb.py eval -m "$MODEL" --tag "$TAG" \
            --tracks "$TRACK" --closed-book ) >>"$LOG" 2>&1
        rc=$?
        if [ "$rc" = 0 ] && [ -f "$OUT" ]; then
            say "$TAG done -> $OUT"
            break
        fi

        after=$(journalled "$CKPT")
        say "$TAG exited rc=$rc after journalling $((after - before)) new item(s)"
        if [ "$after" -le "$before" ]; then
            stalls=$((stalls + 1))
            say "no progress ($stalls/$MAX_STALLS)"
            if [ "$stalls" -ge "$MAX_STALLS" ]; then
                say "$TAG is stuck; stopping so a broken run is not scored"
                exit 1
            fi
        else
            stalls=0
        fi
        sleep 20
    done

    if [ ! -f "$OUT" ]; then
        say "$TAG never finished after $MAX_ATTEMPTS attempt(s)"
        exit 1
    fi
done

say "ALL STAGES DONE"
