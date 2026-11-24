#!/usr/bin/env bash
# Convert the kanana-1.5-8b-instruct checkpoint to GGUF and register it with
# Ollama as kanana1.5:8b, the base of the kanana arms. The trained arms go
# through run_experiment.sh, which reuses Modelfile.kanana-sft with FROM
# pointed at each merged adapter.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${KCBENCH_PY:-/home/tom/venvs/cu130/bin/python}"
SRC="$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('kakaocorp/kanana-1.5-8b-instruct-2505'))")"
Q="$HERE/out/kanana-base-q4km.gguf"; B="$HERE/out/kanana-base-bf16.gguf"
mkdir -p "$HERE/out"
if [ ! -f "$Q" ]; then
  "$PY" ~/llama.cpp/convert_hf_to_gguf.py "$SRC" --outfile "$B" --outtype bf16 || exit 1
  ~/llama.cpp/build/bin/llama-quantize "$B" "$Q" Q4_K_M || exit 1
  rm -f "$B"
fi
ollama show kanana1.5:8b >/dev/null 2>&1 || ollama create kanana1.5:8b -f "$HERE/Modelfile.kanana-sft"
ollama show kanana1.5:8b >/dev/null 2>&1 && echo "kanana1.5:8b registered"
