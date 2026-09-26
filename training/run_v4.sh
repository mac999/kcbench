#!/usr/bin/env bash
# 실험 1: hard-negative 기권 쌍. v3 레시피에서 기권 쌍만 hard로 교체.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${KCBENCH_PY:-/home/tom/venvs/cu130/bin/python}"
LOG="$HERE/run_v4.log"; BM="$HERE/../benchmark"
AUG="$BM/data/train/train_sft_aug3.jsonl"
say(){ echo "$(date '+%F %T') $*" >> "$LOG"; }

[ -f "$AUG" ] || { say "1/4 증강 (hard negatives, 패러프레이즈 재사용 없이 신규 생성)"
  $PY "$HERE/augment_sft.py" --closed-ratio 0.4 --paraphrase-n 2 \
      --abstain-ratio 0.15 --hard-negatives --enum-max-per-doc 20 --enum-max 8000 \
      -o "$AUG" >>"$LOG" 2>&1 || { say FAIL-augment; exit 1; }; }

if [ ! -f "$HERE/out/qwen3-8b-sft4/run.json" ]; then
  say "2/4 SFT v4 학습"
  $PY "$HERE/sft.py" --base "$HERE/out/qwen3-8b-dapt" -d "$AUG" \
      -o "$HERE/out/qwen3-8b-sft4" --save-every 100 >>"$HERE/sft_run_v4.log" 2>&1
  [ -f "$HERE/out/qwen3-8b-sft4/run.json" ] || { say "중단 -> 재개 1회"
    $PY "$HERE/sft.py" --resume-from "$(ls -d $HERE/out/qwen3-8b-sft4/step-* | sort -V | tail -1)" \
        -d "$AUG" -o "$HERE/out/qwen3-8b-sft4" --save-every 100 >>"$HERE/sft_run_v4.log" 2>&1; }
  [ -f "$HERE/out/qwen3-8b-sft4/run.json" ] || { say FAIL-train; exit 1; }
fi
say "학습 완주"

M="$HERE/out/qwen3-8b-sft4-merged"; Q="$HERE/out/qwen3-sft4-q4km.gguf"; B="$HERE/out/qwen3-sft4-bf16.gguf"
[ -f "$M/model.safetensors" ] || $PY "$HERE/merge.py" -a "$HERE/out/qwen3-8b-sft4" -o "$M" >>"$LOG" 2>&1 || { say FAIL-merge; exit 1; }
if [ ! -f "$Q" ]; then
  $PY ~/llama.cpp/convert_hf_to_gguf.py "$M" --outfile "$B" --outtype bf16 >>"$LOG" 2>&1 || { say FAIL-conv; exit 1; }
  ~/llama.cpp/build/bin/llama-quantize "$B" "$Q" Q4_K_M >>"$LOG" 2>&1 || { say FAIL-quant; exit 1; }
  rm -f "$B"
fi
sed "s|FROM .*|FROM $Q|" "$HERE/Modelfile.qwen3-dapt" > "$HERE/Modelfile.qwen3-sft4"
ollama create qwen3-sft:v4 -f "$HERE/Modelfile.qwen3-sft4" >>"$LOG" 2>&1 || { say FAIL-create; exit 1; }
ollama show qwen3-sft:v4 --modelfile | grep -q "PARAMETER stop" || { say FAIL-stop; exit 1; }
say "3/4 등록 ok"

say "4/4 재채점"
cd "$BM"
run(){ say "eval $*"; $PY cb.py eval "$@" >>"$LOG" 2>&1 || say "FAIL $*"; }
run -m qwen3-sft:v4 --tag sft4-uc-open-nt --tracks uc1_safety,uc2_rebar_spec,uc4_faithfulness,uc5_incident --think off
run -m qwen3-sft:v4 --tag sft4-probe-nt   --tracks probe --closed-book --think off
run -m qwen3-sft:v4 --tag sft4-t2-nt      --tracks sft   --closed-book --think off
run -m qwen3-sft:v4 --tag sft4-t2-open-nt --tracks sft                 --think off
$PY cb.py ece -m qwen3-sft:v4 --tag sft4-ece-nt --tracks sft --closed-book --think off >>"$LOG" 2>&1 || say FAIL-ece
say "ALL_DONE"
