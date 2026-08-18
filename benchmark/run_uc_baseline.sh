#!/usr/bin/env bash
# Baseline for the use-case tracks, run strictly serially: this box shares one
# unified memory pool between CPU and GPU, and parallel eval runs have taken
# the whole desktop down. One model loaded at a time, text runs before the
# 30B VLM.
set -u
cd "$(dirname "$0")"
LOG=data/runs/uc-baseline.log
: > "$LOG"

echo "[1/3] qwen3:8b open book" >> "$LOG"
python3 evaluate.py -m qwen3:8b --tag v3-uc-qwen3-8b--open \
    --tracks uc1_safety,uc2_rebar_spec,uc4_faithfulness,uc5_incident \
    --repeats 1 >> "$LOG" 2>&1

echo "[2/3] qwen3:8b closed book" >> "$LOG"
python3 evaluate.py -m qwen3:8b --tag v3-uc-qwen3-8b--closed \
    --tracks uc1_safety,uc2_rebar_spec,uc5_incident \
    --repeats 1 --closed-book >> "$LOG" 2>&1

echo "[3/3] qwen3-vl:30b uc3" >> "$LOG"
python3 evaluate.py -m qwen3-vl:30b --tag v3-uc-qwen3vl-30b--uc3 \
    --tracks uc3_bim_site --repeats 1 >> "$LOG" 2>&1

echo "baseline complete" >> "$LOG"
