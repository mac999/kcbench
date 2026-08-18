#!/usr/bin/env bash
# Sample the things that kill a long run: clock throttling, unified-memory
# pressure, temperature. One CSV line a minute; the box shares one memory pool
# between CPU and GPU, so host memory is as load-bearing as GPU memory here.
set -u
OUT="${1:-$(dirname "${BASH_SOURCE[0]}")/gpu_monitor.log}"
INTERVAL="${INTERVAL:-60}"
[ -s "$OUT" ] || echo "time,sm_mhz,power_w,temp_c,util_pct,throttle,mem_used_gb,mem_avail_gb,swap_used_gb" > "$OUT"
while true; do
    read -r sm pw tc ut th <<<"$(nvidia-smi \
        --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.sw_power_cap \
        --format=csv,noheader,nounits | tr -d ',')"
    # by line, not by label: the row labels are localised and padded
    # /proc/meminfo, not free(1): the row labels there are localised and padded
    read -r mt ma st sf <<<"$(awk '/^MemTotal:/{t=$2}/^MemAvailable:/{a=$2}/^SwapTotal:/{s=$2}/^SwapFree:/{f=$2}END{printf "%d %d %d %d", t/1048576, a/1048576, s/1048576, f/1048576}' /proc/meminfo)"
    used=$((mt - ma)); avail=$ma; swap=$((st - sf))
    echo "$(date '+%F %T'),$sm,$pw,$tc,$ut,$th,$used,$avail,$swap" >> "$OUT"
    sleep "$INTERVAL"
done
