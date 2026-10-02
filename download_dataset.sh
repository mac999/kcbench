#!/usr/bin/env bash
# Download the training dataset from the shared Drive into train_data_v052/,
# the folder the config defaults already point at.
#
#     ./download_dataset.sh              fetch (or resume) the whole dataset
#
# Safe to re-run: finished files are skipped, partial ones resumed. The
# interpreter is picked the way run_cli.sh picks it (KCBENCH_PY, then a venv
# in the repository, then the python on PATH), and gdown is installed into it
# on first use. After it finishes, run_cli.sh works with no path arguments:
# data/ is corpus_dir, train_data/ is generated_dir, metadata/ the catalogues.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDER_URL="https://drive.google.com/drive/folders/1Cz7S-QhXRwQgsajDDyjBAC8vK30jQQTN"
DEST="$HERE/train_data_v052"

if [ -z "${KCBENCH_PY:-}" ]; then
    for candidate in \
        ${KCBENCH_VENV:+"$KCBENCH_VENV/Scripts/python.exe" "$KCBENCH_VENV/bin/python"} \
        "$HERE/venv/Scripts/python.exe"  "$HERE/venv/bin/python" \
        "$HERE/.venv/Scripts/python.exe" "$HERE/.venv/bin/python"; do
        if [ -x "$candidate" ]; then KCBENCH_PY="$candidate"; break; fi
    done
fi
if [ -z "${KCBENCH_PY:-}" ]; then
    for name in python3 python; do
        cand="$(command -v "$name" 2>/dev/null || true)"
        if [ -n "$cand" ] && "$cand" -c "import sys" >/dev/null 2>&1; then
            KCBENCH_PY="$cand"; break
        fi
    done
fi
if [ -z "${KCBENCH_PY:-}" ] || [ ! -x "$KCBENCH_PY" ]; then
    echo "download_dataset.sh: no python found. Make a venv in the repository" >&2
    echo "  root (python -m venv venv) or set KCBENCH_PY / KCBENCH_VENV." >&2
    exit 1
fi

# gdown 5+ walks a shared folder without the old 50-file cap
if ! "$KCBENCH_PY" -c "import gdown" >/dev/null 2>&1; then
    echo "installing gdown into $KCBENCH_PY ..."
    "$KCBENCH_PY" -m pip install --quiet gdown || {
        echo "download_dataset.sh: pip install gdown failed" >&2; exit 1; }
fi

mkdir -p "$DEST"
echo "downloading into $DEST (re-runs resume where they stopped)"
"$KCBENCH_PY" -m gdown --folder "$FOLDER_URL" -O "$DEST" --continue || exit 1

echo
for d in data train_data metadata; do
    if [ -d "$DEST/$d" ]; then
        echo "  $d/  $(find "$DEST/$d" -type f | wc -l) file(s)"
    else
        echo "  $d/  MISSING - the Drive layout may have changed" >&2
    fi
done
echo "done. ./run_cli.sh works with no path arguments now."
