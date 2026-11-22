#!/usr/bin/env bash
# Runs the experiments behind the November 2026 manuscript from the frozen
# settings in this folder. Stages, in the order the paper used them:
#
#   data   download the released v0.5.2 and v0.5.3 data
#          (--regenerate runs the generator instead, --us builds the English set)
#   keys   answer keys, training splits and derived subsets
#   score  untrained qwen3:8b on the three keys (Fig. 3, Table 2 base rows, Section 6)
#   dapt   the DAPT model pair of Section 4.2 (Table 2, Fig. 5)
#   train  the ten training arms of Section 5 (Table 4, Figs. 7-8); --plan prints the steps
#
# Scoring and training skip any tag whose run file exists, so a stage resumes
# where it stopped. Output goes to out/run.log next to this script.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BM="$ROOT/benchmark"; TR="$ROOT/training"
V052="$ROOT/ground_truth_v052"; V053="$ROOT/ground_truth_v053"; US="$ROOT/ground_truth_us"
# scoring needs only requests; training needs the CUDA 13 torch build (env/)
BPY="${KCBENCH_PY:-python3}"
TPY="${KCBENCH_TRAIN_PY:-/home/tom/venvs/cu130/bin/python}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
mkdir -p "$HERE/out"; LOG="$HERE/out/run.log"
say(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# greedy, one sample, thinking off; the judge panel is pinned in the *_ft configs
score(){ # KEY_DIR CONFIG MODEL TAG TRACKS [extra eval flags]
  local gt="$1" cfg="$2" model="$3" tag="$4" tracks="$5"; shift 5
  [ -f "$gt/runs/$tag.json" ] && { say "eval $tag: already scored"; return 0; }
  say "eval $tag ($model, $tracks $*)"
  ( cd "$BM" && "$BPY" cb.py eval -c "$cfg" -m "$model" -t "$tag" --tracks "$tracks" \
      --think off --repeats 1 "$@" ) >> "$LOG" 2>&1 || say "eval $tag FAILED"
}
compare(){ # KEY_DIR CONFIG BASE_TAG AFTER_TAG
  local gt="$1" cfg="$2" b="$3" a="$4"
  [ -f "$gt/runs/compare_${b}_vs_${a}.json" ] && return 0
  ( cd "$BM" && "$BPY" cb.py compare -c "$cfg" -b "$b" -a "$a" \
      --markdown "$gt/runs/compare_${b}_vs_${a}.md" ) >> "$LOG" 2>&1 || say "compare $b vs $a FAILED"
}
unload_ollama(){ # training and scoring must not share GPU memory
  for m in $(curl -s localhost:11434/api/ps | "$BPY" -c "import sys,json;print(' '.join(x['name'] for x in json.load(sys.stdin).get('models',[])))"); do
    curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
  done
}

stage_data(){
  case "${1:-}" in
    "")  ( cd "$ROOT" && ./download_dataset.sh v052 && ./download_dataset.sh v053 ) 2>&1 | tee -a "$LOG" ;;
    --regenerate) # qwen3:30b-a3b at temperature 0.5: a new run differs from the release
      ( cd "$ROOT/gen_aec_syn_data" && "$BPY" main.py --config config_v053.json ) 2>&1 | tee -a "$LOG" ;;
    --us)
      ( cd "$ROOT" && "$BPY" scripts/collect_ecfr.py ) 2>&1 | tee -a "$LOG"
      ( cd "$ROOT/gen_aec_syn_data" && "$BPY" main.py --config config_us.json ) 2>&1 | tee -a "$LOG" ;;
    *) echo "usage: $0 data [--regenerate|--us]"; exit 1 ;;
  esac
}

stage_keys(){ # deterministic for a fixed holdout seed; --strict stops on a contamination finding
  cd "$BM" || exit 1
  for cfg in config_kr_v053.json ${1:+config_kr.json config_us.json}; do
    say "build $cfg"
    "$BPY" cb.py build -c "$cfg" --strict >> "$LOG" 2>&1 || say "build $cfg FAILED"
  done
  # the v0.5.2 rows filtered against the v0.5.3 holdout, so both arms of Section 5 exclude the same documents
  "$BPY" cb.py split -c config_kr_v053.json -i ../train_data_v052/train_data \
      --out-train ../ground_truth_v053/train_from_v052 >> "$LOG" 2>&1 || say "split FAILED"
  "$TPY" "$TR/build_subsets.py" "$TR/experiments/exp_v053cmp.json" 2>&1 | tee -a "$LOG"
}

stage_score(){
  local kr_v1=2,probe,uc1_safety,uc2_rebar_spec,uc4_faithfulness,uc5_incident,uc6_verdict,uc7_requirement
  local kr_all=probe,sft,uc1_safety,uc2_rebar_spec,uc4_faithfulness,uc5_incident,uc6_verdict,uc7_requirement,uc6_verdict_v2,uc7_requirement_v2
  local us_tr=2,probe,uc1_safety,uc2_rebar_spec,uc4_faithfulness,uc5_incident,uc6_verdict,uc6_verdict_v2,uc7_requirement
  score "$V052" config_kr_ft.json qwen3:8b v052-open          "$kr_v1"
  score "$V052" config_kr_ft.json qwen3:8b v052-closed        "$kr_v1" --closed-book
  score "$V052" config_kr_ft.json qwen3:8b uc6v2-open-qwen8b  uc6_verdict_v2
  score "$V052" config_kr_ft.json qwen3:8b uc7v2-open         uc7_requirement_v2
  score "$V052" config_kr_ft.json qwen3:8b uc7v1-regrade      uc7_requirement
  score "$V053" config_kr_v053_ft.json qwen3:8b v053cmp-base-open   "$kr_all"
  score "$V053" config_kr_v053_ft.json qwen3:8b v053cmp-base-closed "$kr_all" --closed-book
  score "$US" config_us.json qwen3:8b us-qwen38b-open   "$us_tr" --lang en
  score "$US" config_us.json qwen3:8b us-qwen38b-closed "$us_tr" --lang en --closed-book
}

stage_dapt(){
  # qwen3-dapt:v1 and :v1f share one GGUF (env/ollama_model_blobs.txt); v1f only
  # fixes the chat template. The adapter of the original run is gone, so a
  # retrain gives the same recipe, not the same weights.
  local adapter="$TR/out/qwen3-8b-dapt" merged="$TR/out/qwen3-8b-dapt-merged"
  local q4="$TR/out/qwen3-dapt-q4km.gguf" bf="$TR/out/qwen3-dapt-bf16.gguf"
  if ! ollama show qwen3-dapt:v1f >/dev/null 2>&1; then
    unload_ollama
    # train_dapt.jsonl is the DAPT split of the first full build; its holdout equals the v0.5.2 holdout
    [ -f "$adapter/run.json" ] || "$TPY" "$TR/dapt.py" -d "$ROOT/ground_truth/train/train_dapt.jsonl" \
        -o "$adapter" --save-every 100 >> "$LOG" 2>&1 || { say "dapt FAILED"; exit 1; }
    [ -f "$merged/model.safetensors" ] || [ -f "$merged/model.safetensors.index.json" ] || \
      "$TPY" "$TR/merge.py" -a "$adapter" -o "$merged" >> "$LOG" 2>&1 || { say "merge FAILED"; exit 1; }
    if [ ! -f "$q4" ]; then
      "$TPY" "$LLAMA_CPP/convert_hf_to_gguf.py" "$merged" --outfile "$bf" --outtype bf16 >> "$LOG" 2>&1 || { say "gguf FAILED"; exit 1; }
      "$LLAMA_CPP/build/bin/llama-quantize" "$bf" "$q4" Q4_K_M >> "$LOG" 2>&1 || { say "quantize FAILED"; exit 1; }
      rm -f "$bf"
    fi
    sed "s|^FROM .*|FROM $q4|" "$HERE/config/training/Modelfile.qwen3-dapt" > "$TR/out/Modelfile.qwen3-dapt-v1f"
    ollama create qwen3-dapt:v1f -f "$TR/out/Modelfile.qwen3-dapt-v1f" >> "$LOG" 2>&1 || { say "ollama create FAILED"; exit 1; }
  fi
  score "$V052" config_kr_ft.json qwen3-dapt:v1f ft-dapt-open   uc,probe,2
  score "$V052" config_kr_ft.json qwen3-dapt:v1f ft-dapt-closed uc,probe,2 --closed-book
  compare "$V052" config_kr_ft.json v052-open         ft-dapt-open   # Table 2, key v1
  compare "$V052" config_kr_ft.json uc6v2-open-qwen8b ft-dapt-open   # Table 2, key v2
}

stage_train(){ # about 50 GPU-hours on a GB10; finished steps are skipped
  cd "$TR" || exit 1
  export KCBENCH_PY="$TPY"
  ./run_experiment.sh experiments/exp_v053cmp.json --plan | tee -a "$LOG"
  [ "${1:-}" = "--plan" ] && return 0
  ./run_experiment.sh experiments/exp_v053cmp.json
}

case "${1:-}" in
  data|keys|score|dapt|train) s="$1"; shift; say "stage $s $*"; "stage_$s" "$@"; say "stage $s done" ;;
  *) echo "usage: $0 data [--regenerate|--us] | keys [--all] | score | dapt | train [--plan]"; exit 1 ;;
esac
