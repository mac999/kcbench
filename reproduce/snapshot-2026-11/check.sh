#!/usr/bin/env bash
# Compares the working tree with the snapshot: the commits, the live settings
# files against their frozen copies, and every hashed input and output against
# MANIFEST.txt. A difference says which numbers of the paper can no longer be
# regenerated from this tree.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
M="$HERE/MANIFEST.txt"; rc=0

for repo in kcbench gen_aec_syn_data; do
  dir="$ROOT"; [ "$repo" = kcbench ] || dir="$ROOT/$repo"
  head=$(cd "$dir" && git rev-parse HEAD); want=$(awk -v r="$repo" '$1==r{print $2}' "$M")
  [ "$head" = "$want" ] && echo "$repo commit: $head (as recorded)" || { echo "$repo commit: $head, snapshot was $want"; rc=1; }
done

echo; echo "settings files against the frozen copies:"
same(){ cmp -s "$ROOT/$1" "$HERE/config/$2" && echo "  same     $1" || { echo "  CHANGED  $1"; rc=1; }; }
for f in config_kr config_kr_ft config_kr_v053 config_kr_v053_ft config_us; do same "benchmark/$f.json" "benchmark/$f.json"; done
for f in config_v052 config_v053 config_us; do same "gen_aec_syn_data/$f.json" "generator/$f.json"; done
same training/experiments/exp_v053cmp.json training/exp_v053cmp.json
same training/Modelfile.qwen3-dapt training/Modelfile.qwen3-dapt
same training/Modelfile.qwen3-sft4 training/Modelfile.qwen3-sft4

echo; echo "hashed files against MANIFEST.txt:"
( cd "$ROOT" && sed -n '/^\[sha256\]/,$p' "$M" | grep -v '^\[' | sha256sum -c --quiet 2>&1 ) | sed 's/^/  /'
n=$(sed -n '/^\[sha256\]/,$p' "$M" | grep -c '^[0-9a-f]')
echo "  ($n files listed; only differences and missing files are printed)"
exit $rc
