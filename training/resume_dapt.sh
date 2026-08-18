#!/bin/bash
# resume DAPT from the latest checkpoint; safe to rerun after a crash.
# original run: batch 4, accum 8 -> 836 total steps; keep these so the
# cosine schedule lines up with the checkpoints.
set -u
cd "$(dirname "$0")"

OUT=out/qwen3-8b-dapt
LOG=dapt_run.log
MON=gpu_monitor.log

# unified memory on GB10 fragments under long runs; expandable segments
# lets the allocator grow in place instead of over-reserving
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LATEST=$(ls -d "$OUT"/step-* 2>/dev/null | sed 's/.*step-//' | sort -n | tail -1)
if [ -z "$LATEST" ]; then
    echo "no checkpoint under $OUT" >&2
    exit 1
fi
echo "resuming from step-$LATEST" | tee -a "$LOG"

# temp/mem every 30s so the next crash has a trail
(
    while true; do
        printf '%s temp=%sC mem_avail=%s\n' \
            "$(date '+%F %T')" \
            "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null)" \
            "$(free -h | awk '/^Mem|^메모리/{print $7}')"
        sleep 30
    done >> "$MON"
) &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null' EXIT

python dapt.py \
    --resume-from "$OUT/step-$LATEST" \
    --batch 4 --accum 8 \
    --save-every 100 \
    2>&1 | tee -a "$LOG"
