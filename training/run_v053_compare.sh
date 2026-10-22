#!/usr/bin/env bash
# Does the v053 data train a better model than v052? Three LoRA SFT arms on
# Qwen3-8B, one recipe, scored on the v053 key against the untrained base:
#   v052   STaR-verified rows of v052, refiltered against the v053 holdout
#   n2202  v053 STaR rows subsampled to the v052 count (quality at equal size)
#   v053   all v053 STaR rows
# Training and scoring share one memory pool, so they never overlap: every
# Ollama model is unloaded before a training step starts.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${KCBENCH_PY:-/home/tom/venvs/cu130/bin/python}"
BM="$HERE/../benchmark"; GT="$HERE/../ground_truth_v053"
CFG="config_kr_v053_ft.json"
LOG="$HERE/run_v053_compare.log"
TRACKS="probe,sft,uc1_safety,uc2_rebar_spec,uc4_faithfulness,uc5_incident,uc6_verdict,uc7_requirement,uc6_verdict_v2,uc7_requirement_v2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ echo "$(date '+%F %T') $*" >> "$LOG"; }

unload(){
  for m in $(curl -s localhost:11434/api/ps | "$PY" -c "import sys,json;print(' '.join(x['name'] for x in json.load(sys.stdin).get('models',[])))"); do
    curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
  done
}

# arm name, data file
train(){
  local arm="$1" data="$2" out="$HERE/out/v053cmp-$1"
  [ -f "$out/run.json" ] && { say "train $arm: already done"; return 0; }
  unload
  local resume=()
  local last; last="$(ls -d "$out"/step-* 2>/dev/null | sort -V | tail -1)"
  [ -n "$last" ] && resume=(--resume-from "$last")
  say "train $arm on $(wc -l < "$data") rows ${resume[*]}"
  "$PY" "$HERE/sft.py" -d "$data" -o "$out" --max-len 3072 --batch 2 --accum 16 \
      --save-every 100 "${resume[@]}" >> "$HERE/sft_v053cmp-$arm.log" 2>&1
  [ -f "$out/run.json" ] || { say "train $arm failed, see sft_v053cmp-$arm.log"; return 1; }
  say "train $arm done"
}

register(){
  local arm="$1" a="$HERE/out/v053cmp-$1"
  local m="$a-merged" q="$HERE/out/v053cmp-$1-q4km.gguf" b="$HERE/out/v053cmp-$1-bf16.gguf"
  ollama show "qwen3-v053cmp:$arm" >/dev/null 2>&1 && { say "register $arm: already"; return 0; }
  [ -f "$m/model.safetensors" ] || [ -f "$m/model.safetensors.index.json" ] || \
    "$PY" "$HERE/merge.py" -a "$a" -o "$m" >> "$LOG" 2>&1 || { say "merge $arm failed"; return 1; }
  if [ ! -f "$q" ]; then
    "$PY" ~/llama.cpp/convert_hf_to_gguf.py "$m" --outfile "$b" --outtype bf16 >> "$LOG" 2>&1 || { say "gguf $arm failed"; return 1; }
    ~/llama.cpp/build/bin/llama-quantize "$b" "$q" Q4_K_M >> "$LOG" 2>&1 || { say "quantize $arm failed"; return 1; }
    rm -f "$b"
  fi
  sed "s|FROM .*|FROM $q|" "$HERE/Modelfile.qwen3-sft4" > "$HERE/out/Modelfile.v053cmp-$arm"
  ollama create "qwen3-v053cmp:$arm" -f "$HERE/out/Modelfile.v053cmp-$arm" >> "$LOG" 2>&1 || { say "create $arm failed"; return 1; }
  rm -rf "$m"
  say "registered qwen3-v053cmp:$arm"
}

score(){
  local model="$1" tag="$2"
  cd "$BM" || return 1
  for book in open closed; do
    [ -f "$GT/runs/$tag-$book.json" ] && { say "eval $tag-$book: already"; continue; }
    local flag=(); [ "$book" = closed ] && flag=(--closed-book)
    say "eval $tag-$book"
    "$PY" cb.py eval -c "$CFG" -m "$model" -t "$tag-$book" --tracks "$TRACKS" \
        --think off --repeats 1 "${flag[@]}" >> "$LOG" 2>&1 || say "eval $tag-$book failed"
  done
}

say "start"
train v052  "$GT/train_from_v052/train_star.jsonl" && register v052
train n2202 "$GT/train/train_star_n2202.jsonl"     && register n2202
score qwen3:8b            v053cmp-base
score qwen3-v053cmp:v052  v053cmp-v052
score qwen3-v053cmp:n2202 v053cmp-n2202
train v053  "$GT/train/train_star.jsonl"           && register v053
score qwen3-v053cmp:v053  v053cmp-v053

cd "$BM"
for arm in v052 n2202 v053; do
  for book in open closed; do
    "$PY" cb.py compare -c "$CFG" -b "v053cmp-base-$book" -a "v053cmp-$arm-$book" \
        --markdown "$GT/runs/compare_v053cmp-$arm-$book.md" >> "$LOG" 2>&1 || say "compare $arm-$book failed"
  done
done
for book in open closed; do
  "$PY" cb.py compare -c "$CFG" -b "v053cmp-v052-$book" -a "v053cmp-v053-$book" \
      --markdown "$GT/runs/compare_v053cmp-v052_vs_v053-$book.md" >> "$LOG" 2>&1
  "$PY" cb.py compare -c "$CFG" -b "v053cmp-v052-$book" -a "v053cmp-n2202-$book" \
      --markdown "$GT/runs/compare_v053cmp-v052_vs_n2202-$book.md" >> "$LOG" 2>&1
done
say "ALL_DONE"
