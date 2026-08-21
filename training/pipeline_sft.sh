#!/usr/bin/env bash
# SFT 후처리 체인: merge -> gguf -> quantize -> ollama 등록.
# 각 단계는 산출물이 이미 있으면 건너뛰므로, 중간에 죽어도 재실행하면 이어진다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/tom/miniforge3/bin/python
LOG="$HERE/pipeline_sft.log"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

MERGED="$HERE/out/qwen3-8b-sft-merged"
BF16="$HERE/out/qwen3-sft-bf16.gguf"
Q4="$HERE/out/qwen3-sft-q4km.gguf"

if [ ! -f "$MERGED/model.safetensors" ]; then
    say "1/4 merge"
    $PY "$HERE/merge.py" -a "$HERE/out/qwen3-8b-sft" -o "$MERGED" >>"$LOG" 2>&1 || { say "merge FAILED"; exit 1; }
else say "1/4 merge - 있음, 건너뜀"; fi

if [ ! -f "$Q4" ]; then
    if [ ! -f "$BF16" ]; then
        say "2/4 gguf 변환"
        $PY ~/llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$BF16" --outtype bf16 >>"$LOG" 2>&1 \
            || { say "convert FAILED"; exit 1; }
    fi
    say "3/4 Q4_K_M 양자화"
    ~/llama.cpp/build/bin/llama-quantize "$BF16" "$Q4" Q4_K_M >>"$LOG" 2>&1 || { say "quantize FAILED"; exit 1; }
    rm -f "$BF16"   # 16GB 중간 산출물, 4.7GB 최종본만 유지
else say "2-3/4 gguf - 있음, 건너뜀"; fi

say "4/4 ollama 등록 (base 템플릿 + stop 토큰)"
sed "s|FROM ./out/qwen3-dapt-q4km.gguf|FROM $Q4|" "$HERE/Modelfile.qwen3-dapt" > "$HERE/Modelfile.qwen3-sft"
ollama create qwen3-sft:v1 -f "$HERE/Modelfile.qwen3-sft" >>"$LOG" 2>&1 || { say "ollama create FAILED"; exit 1; }
ollama show qwen3-sft:v1 --modelfile 2>/dev/null | grep -q "PARAMETER stop" \
    && say "등록 검증: stop 토큰 확인됨" || { say "등록 검증 FAILED: stop 토큰 없음"; exit 1; }
say "PIPELINE DONE"
