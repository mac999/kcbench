#!/usr/bin/env bash
# Run a comparison experiment from its settings file (training/experiments/*.json):
# rebuild the declared subsets, then for each arm train (if it names data),
# register the model with ollama and score it open and closed book, and finally
# write the paired comparison reports. Finished steps are skipped, so a rerun
# resumes. --plan prints what would run and exits.
#
#     ./run_experiment.sh experiments/exp_v053cmp.json [--plan]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${KCBENCH_PY:-/home/tom/venvs/cu130/bin/python}"
EXP="${1:?usage: run_experiment.sh experiments/<name>.json [--plan]}"
PLAN="${2:-}"
BM="$HERE/../benchmark"; GT_RUNS=""
LOG="$HERE/$(basename "$EXP" .json).log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ echo "$(date '+%F %T') $*" >> "$LOG"; }

# one key per line: kind name model data seed (tab separated, - for absent)
plan(){ "$PY" - "$HERE/$EXP" <<'PYEOF'
import json, sys
e = json.load(open(sys.argv[1]))
print("\t".join(["cfg", e["benchmark_config"], e["model_tag"], e["run_tag_prefix"],
                 e["tracks"], e["modelfile_template"]]))
r = e["recipe"]
print("\t".join(["recipe", str(r["max_len"]), str(r["batch"]), str(r["accum"]), str(r["save_every"])]))
for a in e["arms"]:
    print("\t".join(["arm", a["name"], a.get("model", "-"), a.get("data", "-"), str(a.get("seed", "-")),
                     str(a.get("epochs", r.get("epochs", 2)))]))
for b, a in e.get("compares", []):
    print("\t".join(["cmp", b, a]))
PYEOF
}
PLANOUT="$(plan)" || exit 1
read -r CFG MODEL_TAG TAG TRACKS MODELFILE <<< "$(grep -P '^cfg\t' <<< "$PLANOUT" | cut -f2- | tr '\t' ' ')"
read -r _ MAXLEN BATCH ACCUM SAVE <<< "$(grep -P '^recipe\t' <<< "$PLANOUT" | tr '\t' ' ')"
GT="$(cd "$BM" && "$PY" -c "
import json,sys
from pathlib import Path
p=Path('$CFG'); cfg={}
chain=[]
while p.is_file():
    d=json.loads(p.read_text()); chain.append((p.parent,d))
    p=(p.parent/d['extends']).resolve() if d.get('extends') else Path('')
for parent,d in reversed(chain):
    if 'out_dir' in d: print((parent/d['out_dir']).resolve())
" | tail -1)"
[ -d "$GT" ] || { echo "cannot resolve out_dir from $CFG" >&2; exit 1; }

if [ "$PLAN" = "--plan" ]; then
  echo "config    $CFG  (answer key $GT)"
  echo "recipe    max-len $MAXLEN batch $BATCH accum $ACCUM save-every $SAVE"
  while IFS=$'\t' read -r _ name model data seed epochs; do
    if [ "$model" != "-" ]; then st="score only ($model)"; else st="train $data seed=$seed epochs=$epochs"; fi
    done=""
    [ "$model" = "-" ] && [ -f "$HERE/out/$TAG-$name/run.json" ] && done=" [trained]"
    [ -f "$GT/runs/$TAG-$name-open.json" ] && done="$done [open]"
    [ -f "$GT/runs/$TAG-$name-closed.json" ] && done="$done [closed]"
    echo "arm $name: $st$done"
  done < <(grep -P '^arm\t' <<< "$PLANOUT")
  exit 0
fi

"$PY" "$HERE/build_subsets.py" "$HERE/$EXP" >> "$LOG" 2>&1 || { say "subset rebuild failed"; exit 1; }

unload(){
  for m in $(curl -s localhost:11434/api/ps | "$PY" -c "import sys,json;print(' '.join(x['name'] for x in json.load(sys.stdin).get('models',[])))"); do
    curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
  done
}

train(){ # arm data seed epochs
  local arm="$1" data="$HERE/$2" seed="$3" epochs="$4" out="$HERE/out/$TAG-$1"
  [ -f "$out/run.json" ] && { say "train $arm: already done"; return 0; }
  unload
  local resume=()
  local last; last="$(ls -d "$out"/step-* 2>/dev/null | sort -V | tail -1)"
  [ -n "$last" ] && resume=(--resume-from "$last")
  say "train $arm on $(wc -l < "$data") rows seed=$seed epochs=$epochs ${resume[*]}"
  "$PY" "$HERE/sft.py" -d "$data" -o "$out" --max-len "$MAXLEN" --batch "$BATCH" \
      --accum "$ACCUM" --seed "$seed" --epochs "$epochs" --save-every "$SAVE" "${resume[@]}" \
      >> "$HERE/sft_$TAG-$arm.log" 2>&1
  [ -f "$out/run.json" ] || { say "train $arm failed, see sft_$TAG-$arm.log"; return 1; }
  say "train $arm done"
}

register(){
  local arm="$1" a="$HERE/out/$TAG-$1"
  local m="$a-merged" q="$HERE/out/$TAG-$1-q4km.gguf" b="$HERE/out/$TAG-$1-bf16.gguf"
  ollama show "$MODEL_TAG:$arm" >/dev/null 2>&1 && { say "register $arm: already"; return 0; }
  [ -f "$m/model.safetensors" ] || [ -f "$m/model.safetensors.index.json" ] || \
    "$PY" "$HERE/merge.py" -a "$a" -o "$m" >> "$LOG" 2>&1 || { say "merge $arm failed"; return 1; }
  if [ ! -f "$q" ]; then
    "$PY" ~/llama.cpp/convert_hf_to_gguf.py "$m" --outfile "$b" --outtype bf16 >> "$LOG" 2>&1 || { say "gguf $arm failed"; return 1; }
    ~/llama.cpp/build/bin/llama-quantize "$b" "$q" Q4_K_M >> "$LOG" 2>&1 || { say "quantize $arm failed"; return 1; }
    rm -f "$b"
  fi
  sed "s|FROM .*|FROM $q|" "$HERE/$MODELFILE" > "$HERE/out/Modelfile.$TAG-$arm"
  ollama create "$MODEL_TAG:$arm" -f "$HERE/out/Modelfile.$TAG-$arm" >> "$LOG" 2>&1 || { say "create $arm failed"; return 1; }
  rm -rf "$m"
  say "registered $MODEL_TAG:$arm"
}

score(){ # model tag-suffix
  local model="$1" rt="$TAG-$2"
  cd "$BM" || return 1
  for book in open closed; do
    [ -f "$GT/runs/$rt-$book.json" ] && { say "eval $rt-$book: already"; continue; }
    local flag=(); [ "$book" = closed ] && flag=(--closed-book)
    say "eval $rt-$book"
    "$PY" cb.py eval -c "$CFG" -m "$model" -t "$rt-$book" --tracks "$TRACKS" \
        --think off --repeats 1 "${flag[@]}" >> "$LOG" 2>&1 || say "eval $rt-$book failed"
  done
}

say "start $EXP"
while IFS=$'\t' read -r _ name model data seed epochs; do
  if [ "$model" != "-" ]; then
    score "$model" "$name"
  else
    train "$name" "$data" "$seed" "$epochs" && register "$name" && score "$MODEL_TAG:$name" "$name"
  fi
done < <(grep -P '^arm\t' <<< "$PLANOUT")

cd "$BM"
while IFS=$'\t' read -r _ b a; do
  for book in open closed; do
    [ -f "$GT/runs/$TAG-$b-$book.json" ] && [ -f "$GT/runs/$TAG-$a-$book.json" ] || continue
    "$PY" cb.py compare -c "$CFG" -b "$TAG-$b-$book" -a "$TAG-$a-$book" \
        --markdown "$GT/runs/compare_$TAG-${b}_vs_${a}-$book.md" >> "$LOG" 2>&1 || say "compare $b vs $a $book failed"
  done
done < <(grep -P '^cmp\t' <<< "$PLANOUT")
say "ALL_DONE"
