#!/usr/bin/env bash
# 루프 3턴 무인 실행: 증강(패러프레이즈 포함) -> SFT v3 -> 후처리 -> 재채점.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${KCBENCH_PY:-/home/tom/venvs/cu130/bin/python}"
LOG="$HERE/run_v3.log"
BM="$HERE/../benchmark"
AUG="$BM/data/train/train_sft_aug2.jsonl"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

if [ ! -f "$AUG" ]; then
    say "1/4 증강 생성 (패러프레이즈 ~5h)"
    $PY "$HERE/augment_sft.py" --closed-ratio 0.4 --paraphrase-n 2 \
        --abstain-ratio 0.15 --enum-max-per-doc 20 --enum-max 8000 \
        -o "$AUG" >>"$LOG" 2>&1 || { say FAIL-augment; exit 1; }
else say "1/4 증강 - 있음, 건너뜀"; fi

if [ ! -f "$HERE/out/qwen3-8b-sft3/run.json" ]; then
    say "2/4 SFT v3 학습"
    $PY "$HERE/sft.py" --base "$HERE/out/qwen3-8b-dapt" -d "$AUG" \
        -o "$HERE/out/qwen3-8b-sft3" --save-every 100 >>"$HERE/sft_run_v3.log" 2>&1
    [ -f "$HERE/out/qwen3-8b-sft3/run.json" ] || { say "학습 중단 -> 재개 1회 시도";
        $PY "$HERE/sft.py" --resume-from "$(ls -d $HERE/out/qwen3-8b-sft3/step-* | sort -V | tail -1)" \
            -d "$AUG" -o "$HERE/out/qwen3-8b-sft3" --save-every 100 >>"$HERE/sft_run_v3.log" 2>&1; }
    [ -f "$HERE/out/qwen3-8b-sft3/run.json" ] || { say FAIL-train; exit 1; }
else say "2/4 학습 - 완료돼 있음"; fi
say "학습 완주"

MERGED="$HERE/out/qwen3-8b-sft3-merged"; Q4="$HERE/out/qwen3-sft3-q4km.gguf"; BF="$HERE/out/qwen3-sft3-bf16.gguf"
[ -f "$MERGED/model.safetensors" ] || { say "3/4 merge";
    $PY "$HERE/merge.py" -a "$HERE/out/qwen3-8b-sft3" -o "$MERGED" >>"$LOG" 2>&1 || { say FAIL-merge; exit 1; }; }
if [ ! -f "$Q4" ]; then
    $PY ~/llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$BF" --outtype bf16 >>"$LOG" 2>&1 || { say FAIL-conv; exit 1; }
    ~/llama.cpp/build/bin/llama-quantize "$BF" "$Q4" Q4_K_M >>"$LOG" 2>&1 || { say FAIL-quant; exit 1; }
    rm -f "$BF"
fi
sed "s|FROM .*|FROM $Q4|" "$HERE/Modelfile.qwen3-dapt" > "$HERE/Modelfile.qwen3-sft3"
ollama create qwen3-sft:v3 -f "$HERE/Modelfile.qwen3-sft3" >>"$LOG" 2>&1 || { say FAIL-create; exit 1; }
ollama show qwen3-sft:v3 --modelfile | grep -q "PARAMETER stop" || { say FAIL-stop; exit 1; }
say "등록 ok"

say "4/4 재채점"
cd "$BM"
run() { say "eval $*"; $PY cb.py eval "$@" >>"$LOG" 2>&1 || say "FAIL-eval $*"; }
run -m qwen3-sft:v3 --tag sft3-probe-nt   --tracks probe --closed-book --think off
run -m qwen3-sft:v3 --tag sft3-t2-nt      --tracks sft   --closed-book --think off
run -m qwen3-sft:v3 --tag sft3-t2-open-nt --tracks sft                 --think off
run -m qwen3-sft:v3 --tag sft3-uc-open-nt --tracks uc1_safety,uc2_rebar_spec,uc4_faithfulness,uc5_incident --think off
$PY cb.py ece -m qwen3-sft:v3 --tag sft3-ece-nt --tracks sft --closed-book --think off >>"$LOG" 2>&1 || say FAIL-ece
say "ALL_DONE"
