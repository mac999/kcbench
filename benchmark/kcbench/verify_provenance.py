#!/usr/bin/env python3
"""
Prove, per item, where it came from and that nothing trains on it.

    python verify_provenance.py
    python verify_provenance.py --train-dir data/train --strict
    python verify_provenance.py -i /data/ai_ready_v3 -o /tmp/bench
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict

from kcbench.common import (BENCHMARK_NAME, BENCHMARK_VERSION, add_common_args, describe,
                    log, read_jsonl, resolve_config, sha256_text, utc_now,
                    write_json, write_jsonl)

LOG = log("provenance")

TRACK_FILES = {"1": "track1_dapt.jsonl", "2": "track2_sft.jsonl", "3": "track3_vlm.jsonl"}
TRAIN_FILES = ("train_dapt.jsonl", "train_sft.jsonl", "train_vlm.jsonl")


def train_digests(train_dir: Path) -> tuple[set[str], int]:
    """
    Every training row's text, as digests.

    Held as digests rather than text so the whole training split can be checked
    without holding 100 MB of Korean regulation in memory, and so the comparison
    is exact rather than a substring search that would flag any shared boilerplate.
    """
    seen: set[str] = set()
    rows = 0
    for name in TRAIN_FILES:
        path = train_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rows += 1
                for key in ("text", "output", "response", "answer", "completion"):
                    val = rec.get(key)
                    if isinstance(val, str) and val:
                        seen.add(sha256_text(val))
    return seen, rows


def source_row(cfg, prov: Dict[str, Any]) -> tuple[bool, bool, str]:
    """(file exists, row still hashes as recorded, the text found there)."""
    rel = prov.get("dataset_file")
    if not rel:
        return False, False, ""
    path = cfg["generated_dir"] / rel
    if not path.is_file():
        return False, False, ""
    idx = prov.get("row_index")
    if idx is None:                      # track 3 keys on a catalogue, not a row
        return True, True, ""
    try:
        with path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                if n == idx:
                    text = json.loads(line).get("text", "")
                    return True, sha256_text(text) == prov.get("chunk_sha256"), text
    except Exception as exc:
        LOG.warning("unreadable source %s (%s)", path, exc)
        return True, False, ""
    return True, False, ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify item provenance and contamination",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--holdout", metavar="FILE", help="holdout.json (default <out-dir>/holdout.json)")
    ap.add_argument("--train-dir", metavar="DIR", help="training split to check against "
                                                       "(default <out-dir>/train)")
    ap.add_argument("--tracks", default="1,2,3", metavar="LIST", help="tracks to verify")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any check fails")
    args = ap.parse_args()

    cfg = resolve_config(args)
    describe(cfg)

    hp = Path(args.holdout) if args.holdout else cfg["out_dir"] / "holdout.json"
    if not hp.exists():
        LOG.error("no holdout to verify against: %s", hp)
        return 1
    holdout = json.loads(hp.read_text(encoding="utf-8"))
    held_docs = {d["generated_dir"] for d in holdout["pdf_documents"]}
    held_models = {m["model"] for m in holdout["ifc_models"]}

    train_dir = Path(args.train_dir) if args.train_dir else cfg["out_dir"] / "train"
    if train_dir.exists():
        digests, train_rows = train_digests(train_dir)
        LOG.info("training split: %d row(s), %d distinct text digest(s) in %s",
                 train_rows, len(digests), train_dir)
    else:
        digests, train_rows = set(), 0
        LOG.warning("no training split at %s - run make_train_split.py first. "
                    "absent_from_train cannot be checked and is reported as null", train_dir)

    rows: list[dict] = []
    failures: collections.Counter = collections.Counter()
    by_track: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    for t in [x.strip() for x in args.tracks.split(",") if x.strip()]:
        path = cfg["out_dir"] / TRACK_FILES[t]
        if not path.exists():
            LOG.warning("track %s not built - skipping", t)
            continue
        for item in read_jsonl(path):
            prov = item.get("provenance") or {}
            exists, matches, text = source_row(cfg, prov)
            doc_key = prov.get("generated_dir")
            in_holdout = doc_key in held_docs or prov.get("document") in held_models
            digest = prov.get("chunk_sha256") or (sha256_text(text) if text else None)
            absent = None if not digests else (digest not in digests if digest else True)

            checks = {"source_exists": exists, "source_matches": matches,
                      "in_holdout": in_holdout, "absent_from_train": absent}
            for name, ok in checks.items():
                if ok is False:
                    failures[name] += 1
                    by_track[f"track{t}"][name] += 1
            rows.append({
                "id": item["id"], "track": item.get("track"), "eval_type": item.get("eval_type"),
                "dataset": prov.get("dataset"),
                "dataset_file": prov.get("dataset_file"),
                "row_index": prov.get("row_index"),
                "char_start": prov.get("char_start"), "char_end": prov.get("char_end"),
                "chunk_sha256": prov.get("chunk_sha256"),
                "source_document": prov.get("document"),
                "source_name": prov.get("source_name"),
                "source_ifc": prov.get("source_ifc"),
                "category": prov.get("category"),
                "checks": checks,
                "verified": all(v is not False for v in checks.values()),
            })
            by_track[f"track{t}"]["items"] += 1

    verified = sum(1 for r in rows if r["verified"])
    LOG.info("%d item(s) checked, %d fully verified", len(rows), verified)
    for name in ("source_exists", "source_matches", "in_holdout", "absent_from_train"):
        n = failures[name]
        (LOG.error if n else LOG.info)("  %-18s %s", name,
                                       f"{len(rows) - n}/{len(rows)} pass" if rows else "no items")

    out = write_jsonl(cfg["out_dir"] / "provenance.jsonl", rows)
    LOG.info("wrote %s", out)
    report_path = write_json(cfg["out_dir"] / "contamination_report.json", {
        "benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION, "checked_at": utc_now(),
        "generated_dir": str(cfg["generated_dir"]),
        "holdout_file": str(hp), "train_dir": str(train_dir),
        "train_rows": train_rows, "train_text_digests": len(digests),
        "items": len(rows), "fully_verified": verified,
        "failures": dict(failures),
        "by_track": {k: dict(v) for k, v in by_track.items()},
        "method_en": ("Each item records the generated file and row it was mined from. "
                      "source_matches re-hashes that row; absent_from_train looks the same "
                      "digest up in every text field of the training split. A document-name "
                      "check would pass a file that had been renamed or re-chunked."),
        "method_ko": ("각 문항은 출처 생성 파일과 행 번호를 기록합니다. source_matches 는 해당 행을 "
                      "다시 해시하고, absent_from_train 은 같은 다이제스트를 학습셋 전체 텍스트에서 "
                      "찾습니다. 문서명 대조만으로는 파일명이 바뀌거나 재청킹된 경우를 놓칩니다."),
    })
    LOG.info("wrote %s", report_path)

    if args.strict and failures:
        LOG.error("strict mode: %d check(s) failed", sum(failures.values()))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
