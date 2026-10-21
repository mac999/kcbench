#!/usr/bin/env bash
# Download the training dataset from the shared Drive.
#
#     ./download_dataset.sh              v052 into train_data_v052/ (the
#                                        default config_kr.json)
#     ./download_dataset.sh v053         also the v0.5.3 regeneration into
#                                        train_data_v053/train_data, for
#                                        -c config_kr_v053.json
#
# Both releases share one corpus and catalogue, so data/ and metadata/ always
# go to train_data_v052/, where config_kr_v053.json inherits them from.
#
# Safe to re-run: finished files are skipped, partial ones resumed. The
# interpreter is picked the way run_cli.sh picks it (KCBENCH_PY, then a venv
# in the repository, then the python on PATH), and gdown is installed into it
# on first use. After it finishes, run_cli.sh works with no path arguments:
# data/ is corpus_dir, train_data/ is generated_dir, metadata/ the catalogues.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Subfolders of the shared Drive folder 1Cz7S-QhXRwQgsajDDyjBAC8vK30jQQTN,
# fetched one by one so the release not asked for is not downloaded with it
DRIVE="https://drive.google.com/drive/folders"
DATA_ID="1byKtnR64tYzP8p0AviZQRwFvFwairrjH"
META_ID="1hSZqIjddwi8PmW_Q0KebmD0bb1ctK5qD"
V052_ID="1JqkTvGBbVFHsaK4juMqiJ_9GyUv6C_ft"
V053_ID="1kZAL3ZstGPA5y-SDeFvbnB9CzcnMv2rX"
DEST="$HERE/train_data_v052"

RELEASE="${1:-v052}"
case "$RELEASE" in
    v052) GEN_ID="$V052_ID"; GEN_DEST="$DEST/train_data" ;;
    v053) GEN_ID="$V053_ID"; GEN_DEST="$HERE/train_data_v053/train_data" ;;
    *) echo "usage: ./download_dataset.sh [v052|v053]" >&2; exit 1 ;;
esac

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

echo "downloading $RELEASE (re-runs resume where they stopped)"
for pair in "$DATA_ID $DEST/data" "$META_ID $DEST/metadata" "$GEN_ID $GEN_DEST"; do
    set -- $pair
    mkdir -p "$2"
    # no trailing slash on -O: gdown then writes the folder's contents there
    "$KCBENCH_PY" -m gdown --folder "$DRIVE/$1" -O "$2" --continue || exit 1
done

echo
for d in "$DEST/data" "$DEST/metadata" "$GEN_DEST"; do
    n="$(find "$d" -type f 2>/dev/null | wc -l)"
    if [ "$n" -gt 0 ]; then
        echo "  ${d#$HERE/}/  $n file(s)"
    else
        echo "  ${d#$HERE/}/  EMPTY - the Drive layout may have changed" >&2
    fi
done
if [ "$RELEASE" = v053 ]; then
    echo "done. pass -c config_kr_v053.json to cb.py to use the v053 answer key."
else
    echo "done. ./run_cli.sh works with no path arguments now."
fi
