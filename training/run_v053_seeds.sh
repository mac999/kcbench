#!/usr/bin/env bash
# Second round of the v053 comparison: the refusal cause check and the seed
# spread. One arm trains the full v053 STaR rows with refusal capped at 5%
# (train_star_cap5.jsonl); the three claims-bearing conditions then repeat
# under two more shuffle seeds, so a score difference can be read against
# run-to-run noise. Same recipe and scoring as run_v053_compare.sh.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${KCBENCH_PY:-/home/tom/venvs/cu130/bin/python}"
BM="$HERE/../benchmark"; GT="$HERE/../ground_truth_v053"
CFG="config_kr_v053_ft.json"
LOG="$HERE/run_v053_seeds.log"
TRACKS="probe,sft,uc1_safety,uc2_rebar_spec,uc4_faithfulness,uc5_incident,uc6_verdict,uc7_requirement,uc6_verdict_v2,uc7_requirement_v2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ echo "$(date '+%F %T') $*" >> "$LOG"; }

unload(){
  for m in $(curl -s localhost:11434/api/ps | "$PY" -c "import sys,json;print(' '.join(x['name'] for x in json.load(sys.stdin).get('models',[])))"); do
    curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
  done
}

# arm name, data file, seed
train(){
  local arm="$1" data="$2" seed="$3" out="$HERE/out/v053cmp-$1"
  [ -f "$out/run.json" ] && { say "train $arm: already done"; return 0; }
  unload
  local resume=()
  local last; last="$(ls -d "$out"/step-* 2>/dev/null | sort -V | tail -1)"
  [ -n "$last" ] && resume=(--resume-from "$last")
  say "train $arm on $(wc -l < "$data") rows seed=$seed ${resume[*]}"
  "$PY" "$HERE/sft.py" -d "$data" -o "$out" --max-len 3072 --batch 2 --accum 16 \
      --seed "$seed" --save-every 100 "${resume[@]}" >> "$HERE/sft_v053cmp-$arm.log" 2>&1
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

run_arm(){ # arm data seed
  train "$1" "$2" "$3" && register "$1" && score "qwen3-v053cmp:$1" "v053cmp-$1"
}

say "start"
# the cause check first: does capping refusal rows repair the regression?
run_arm v053cap    "$GT/train/train_star_cap5.jsonl"       20260814
say "MILESTONE v053cap scored"
# then the seed spread, small arms first
run_arm v052-s2    "$GT/train_from_v052/train_star.jsonl"  7
run_arm v052-s3    "$GT/train_from_v052/train_star.jsonl"  42
run_arm n2202-s2   "$GT/train/train_star_n2202.jsonl"      7
run_arm n2202-s3   "$GT/train/train_star_n2202.jsonl"      42
say "MILESTONE small seed arms scored"
run_arm v053cap-s2 "$GT/train/train_star_cap5.jsonl"       7
run_arm v053cap-s3 "$GT/train/train_star_cap5.jsonl"       42

cd "$BM"
for arm in v053cap v052-s2 v052-s3 n2202-s2 n2202-s3 v053cap-s2 v053cap-s3; do
  for book in open closed; do
    "$PY" cb.py compare -c "$CFG" -b "v053cmp-base-$book" -a "v053cmp-$arm-$book" \
        --markdown "$GT/runs/compare_v053cmp-$arm-$book.md" >> "$LOG" 2>&1 || say "compare $arm-$book failed"
  done
done
for book in open closed; do
  "$PY" cb.py compare -c "$CFG" -b "v053cmp-v053-$book" -a "v053cmp-v053cap-$book" \
      --markdown "$GT/runs/compare_v053cmp-v053_vs_v053cap-$book.md" >> "$LOG" 2>&1
done
say "ALL_DONE"
