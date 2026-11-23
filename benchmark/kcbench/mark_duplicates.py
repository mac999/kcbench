"""
Mark held-out items that repeat a trained-on item word for word.

Sibling standards copy clauses from one another: the fire-pump performance
rule (65% of rated pressure, 175% of rated flow) appears in a dozen NFSC
notices. When one copy is in the holdout and another was trained on, the
holdout item asks a question whose answer the model has seen verbatim, and
the document-level split does not catch it because the two documents differ.
This pass compares every holdout-side item's question and answer, normalised,
against every train-side item (probe and the train split of the use-case
tracks) and writes `contamination: "duplicate_of:<id>"` into the matching
rows. Scoring does not read the field; the aggregate reports the clean subset
beside the full track.

    python mark_duplicates.py -c config.json            # report only
    python mark_duplicates.py -c config.json --write    # rewrite the item files
"""
from __future__ import annotations

import argparse
import collections
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

from kcbench.common import (add_common_args, resolve_config, describe, log,
                            read_jsonl, write_jsonl, write_json, track_files)

LOG = log("mark_duplicates")
DUPLICATE = "duplicate_of:"
_WS = re.compile(r"\s+")


def _key(item: dict) -> Tuple[str, str] | None:
    q = item.get("question_ko") or item.get("question_en") or item.get("question") or ""
    a = item.get("answer_ko") or item.get("answer") or item.get("answer_verdict") or ""
    if isinstance(a, list):
        a = "|".join(sorted(str(x) for x in a))
    q, a = str(q), str(a)
    if not q.strip() or not a.strip():
        return None
    # NFKC plus whitespace removal: the same clause arrives from different
    # PDFs with different spacing around particles and full-width digits
    return (_WS.sub("", unicodedata.normalize("NFKC", q)).casefold(),
            _WS.sub("", unicodedata.normalize("NFKC", a)).casefold())


def _trained_side(item: dict) -> bool:
    return item.get("split") == "train"


def mark(cfg, tracks: List[str], write: bool) -> dict:
    files = track_files(cfg)
    rows: Dict[str, List[dict]] = {}
    for t in tracks:
        p = cfg["out_dir"] / files.get(t, f"{t}.jsonl")
        if p.exists():
            rows[t] = read_jsonl(p)
    trained: Dict[Tuple[str, str], str] = {}
    for t, items in rows.items():
        for it in items:
            k = _key(it)
            if k and _trained_side(it):
                trained.setdefault(k, it["id"])

    report: Dict[str, dict] = {}
    changed = 0
    for t, items in rows.items():
        hits = 0
        for it in items:
            if _trained_side(it):
                continue
            k = _key(it)
            dup = trained.get(k) if k else None
            had = str(it.get("contamination") or "")
            if dup and dup != it["id"]:
                hits += 1
                tag = DUPLICATE + dup
                if had != tag:
                    it["contamination"] = tag
                    changed += 1
            elif had.startswith(DUPLICATE):
                # a stale mark from a previous build whose train side has moved
                del it["contamination"]
                changed += 1
        held = sum(1 for it in items if not _trained_side(it))
        report[t] = {"holdout_items": held, "duplicate_of_train": hits}
        if hits:
            LOG.info("track %-18s %4d held-out item(s), %3d repeat a trained item", t, held, hits)
        if write:
            write_jsonl(cfg["out_dir"] / files.get(t, f"{t}.jsonl"), items)
    LOG.info("%d trained-side item(s) indexed, %d row(s) %s",
             len(trained), changed, "rewritten" if write else "would change")
    return {"trained_items": len(trained), "rows_changed": changed, "by_track": report}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mark held-out items that duplicate a trained-on item",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--tracks", default="all", metavar="LIST",
                    help="tracks to scan (default every built track)")
    ap.add_argument("--write", action="store_true",
                    help="rewrite the item files with the contamination field")
    args = ap.parse_args()
    cfg = resolve_config(args)
    describe(cfg)
    files = track_files(cfg)
    tracks = [t for t in files if t not in ("1", "3")] if args.tracks == "all" \
        else [t.strip() for t in args.tracks.split(",") if t.strip()]
    out = mark(cfg, tracks, args.write)
    write_json(cfg["out_dir"] / "duplicate_report.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
