#!/usr/bin/env bash
# Restart SFT from the newest checkpoint the interrupted run left behind.
# Optimiser state is not saved, so a resume re-warms for a few steps; the
# alternative is losing every hour since the last save.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/out/qwen3-8b-sft}"
DAPT="$HERE/out/qwen3-8b-dapt"
LOG="$HERE/sft_run.log"
PY=/home/tom/miniforge3/bin/python

LATEST=$(ls -d "$OUT"/step-* 2>/dev/null | sed 's/.*step-//' | sort -n | tail -1)
if [ -z "${LATEST:-}" ]; then
    echo "no checkpoint under $OUT, starting from the DAPT adapter" | tee -a "$LOG"
    exec "$PY" "$HERE/sft.py" --base "$DAPT" -o "$OUT" --save-every 100 >>"$LOG" 2>&1
fi
echo "resuming from step-$LATEST" | tee -a "$LOG"
exec "$PY" "$HERE/sft.py" --resume-from "$OUT/step-$LATEST" -o "$OUT" \
    --save-every 100 >>"$LOG" 2>&1
