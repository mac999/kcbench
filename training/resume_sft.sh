#!/usr/bin/env bash
# Restart SFT from the newest checkpoint the interrupted run left behind.
# Checkpoints since v4.5 carry optimizer.pt and the schedule resumes at its
# step; older checkpoints resume with fresh moments.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/out/qwen3-8b-sft}"
# the data file the run was started with; without it the resume trained on
# the default train_sft.jsonl whatever the original arm used
DATA="${2:-}"; DFLAG=(); [ -n "$DATA" ] && DFLAG=(-d "$DATA")
DAPT="$HERE/out/qwen3-8b-dapt"
LOG="$HERE/sft_run.log"
PY="${KCBENCH_PY:-/home/tom/venvs/cu130/bin/python}"

LATEST=$(ls -d "$OUT"/step-* 2>/dev/null | sed 's/.*step-//' | sort -n | tail -1)
if [ -z "${LATEST:-}" ]; then
    echo "no checkpoint under $OUT, starting from the DAPT adapter" | tee -a "$LOG"
    exec "$PY" "$HERE/sft.py" --base "$DAPT" -o "$OUT" "${DFLAG[@]}" --save-every 100 >>"$LOG" 2>&1
fi
echo "resuming from step-$LATEST" | tee -a "$LOG"
exec "$PY" "$HERE/sft.py" --resume-from "$OUT/step-$LATEST" -o "$OUT" "${DFLAG[@]}" \
    --save-every 100 >>"$LOG" 2>&1
