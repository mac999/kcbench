#!/usr/bin/env bash
# SFT v2 완주를 기다렸다가 후처리와 재채점까지 무인 실행.
# 각 단계는 산출물이 있으면 건너뛰므로 재실행 안전.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${KCBENCH_PY:-/home/tom/venvs/cu130/bin/python}"
LOG="$HERE/after_sft2.log"
BM="$HERE/../benchmark"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "대기: SFT v2 완주"
while [ ! -f "$HERE/out/qwen3-8b-sft2/run.json" ]; do
    pgrep -f "sft.py --base" >/dev/null || pgrep -f "sft.py --resume" >/dev/null || {
        say "SFT 프로세스 소멸 + run.json 없음 -> resume_sft.sh로 재개"
        setsid "$HERE/resume_sft.sh" "$HERE/out/qwen3-8b-sft2" < /dev/null &
        sleep 120
    }
    sleep 120
done
say "SFT v2 완주 확인"

MERGED="$HERE/out/qwen3-8b-sft2-merged"
Q4="$HERE/out/qwen3-sft2-q4km.gguf"
BF16="$HERE/out/qwen3-sft2-bf16.gguf"

[ -f "$MERGED/model.safetensors" ] || { say "merge";
    $PY "$HERE/merge.py" -a "$HERE/out/qwen3-8b-sft2" -o "$MERGED" >>"$LOG" 2>&1 || { say FAIL-merge; exit 1; }; }
if [ ! -f "$Q4" ]; then
    say "gguf+quant"
    $PY ~/llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$BF16" --outtype bf16 >>"$LOG" 2>&1 || { say FAIL-convert; exit 1; }
    ~/llama.cpp/build/bin/llama-quantize "$BF16" "$Q4" Q4_K_M >>"$LOG" 2>&1 || { say FAIL-quant; exit 1; }
    rm -f "$BF16"
fi
say "ollama 등록"
sed "s|FROM .*|FROM $Q4|" "$HERE/Modelfile.qwen3-dapt" > "$HERE/Modelfile.qwen3-sft2"
ollama create qwen3-sft:v2 -f "$HERE/Modelfile.qwen3-sft2" >>"$LOG" 2>&1 || { say FAIL-create; exit 1; }
ollama show qwen3-sft:v2 --modelfile | grep -q "PARAMETER stop" || { say FAIL-verify-stop; exit 1; }
say "등록 검증 ok"

cd "$BM"
run() { say "eval $*"; $PY cb.py eval "$@" >>"$LOG" 2>&1 || say "FAIL-eval $*"; }
run -m qwen3-sft:v2 --tag sft2-probe-nt   --tracks probe --closed-book --think off
run -m qwen3-sft:v2 --tag sft2-t2-nt      --tracks sft   --closed-book --think off
run -m qwen3-sft:v2 --tag sft2-t2-open-nt --tracks sft                 --think off
run -m qwen3-sft:v2 --tag sft2-uc-open-nt --tracks uc1_safety,uc2_rebar_spec,uc4_faithfulness,uc5_incident --think off
say "ece"
$PY cb.py ece -m qwen3-sft:v2 --tag sft2-ece-nt --tracks sft --closed-book --think off >>"$LOG" 2>&1 || say FAIL-ece
say "ALL_DONE"
