#!/usr/bin/env python3
"""
Assemble the training split with every held-out document removed.

    python make_train_split.py
    python make_train_split.py -i /data/ai_ready_v3 --out-train /data/train_v3
    python make_train_split.py --check    # report contamination, write nothing
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from corpusbench.common import (add_common_args, describe, log, resolve_config, sha256_text,
                    write_json, write_jsonl)

LOG = log("split")

KINDS = {
    "sft": "sllm_training_data.jsonl",
    "dapt": "dapt_training_data.jsonl",
    "vlm": "vlm_training_data.jsonl",
}


def held_keys(holdout: dict) -> tuple[set[str], set[str]]:
    """
    Held-out documents, keyed by their folder path under generated_dir.

    The path rather than the name, because names repeat across categories:
    excluding by name would drop a document the split never held out, and the
    benchmark would then be scored against text nothing trained on.
    """
    pdf = {d.get("generated_dir") or d.get("stem") or Path(d["path"]).stem
           for d in holdout["pdf_documents"]}
    ifc = {m["model"] for m in holdout["ifc_models"]}
    return pdf, ifc


def held_text_digests(holdout: dict) -> set[str]:
    """
    Digests of every reserved chunk, for the exclusion the folder path misses.

    The corpus collects the same regulation more than once — an amendment
    alongside the text it amends, a file captured twice under names differing by
    a suffix, sibling standards sharing whole clauses. Dropping the held-out
    *document* then leaves its sentences in the training data under a different
    document's name, and the benchmark quietly measures recall again.
    """
    return {h for d in holdout["pdf_documents"] for h in d.get("chunk_sha256") or []}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the contamination-free training split",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--holdout", metavar="FILE",
                    help="holdout.json to exclude (default <out-dir>/holdout.json)")
    ap.add_argument("--out-train", metavar="DIR",
                    help="where the training split is written (default <out-dir>/train)")
    ap.add_argument("--check", action="store_true",
                    help="report contamination and exit without writing")
    args = ap.parse_args()

    cfg = resolve_config(args)
    describe(cfg)

    hp = Path(args.holdout) if args.holdout else cfg["out_dir"] / "holdout.json"
    if not hp.exists():
        LOG.error("no holdout to exclude - run build_holdout.py first (%s)", hp)
        return 1
    holdout = json.loads(hp.read_text(encoding="utf-8"))
    pdf_held, ifc_held = held_keys(holdout)
    held_digests = held_text_digests(holdout)
    LOG.info("holdout: %d document(s), %d model(s), %d reserved chunk digest(s)",
             len(pdf_held), len(ifc_held), len(held_digests))

    kept: dict[str, list[dict]] = {k: [] for k in KINDS}
    dropped_docs: set[str] = set()
    dropped_rows: collections.Counter = collections.Counter()
    dropped_dupes: collections.Counter = collections.Counter()
    dupe_docs: set[str] = set()
    kept_docs: set[str] = set()

    for kind, fname in KINDS.items():
        for path in sorted(cfg["generated_dir"].rglob(fname)):
            folder = str(path.parent.relative_to(cfg["generated_dir"]))
            stem = path.parent.name
            excluded = folder in pdf_held or stem in ifc_held
            try:
                rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            except Exception as exc:
                LOG.warning("unreadable, skipping: %s (%s)", path, exc)
                continue
            if excluded:
                dropped_docs.add(folder)
                dropped_rows[kind] += len(rows)
                continue
            kept_docs.add(folder)
            src = str(path.relative_to(cfg["generated_dir"]))
            for r in rows:
                # A row from a document the split kept, carrying text the split
                if held_digests and any(
                        isinstance(r.get(k), str) and r[k] and sha256_text(r[k]) in held_digests
                        for k in ("text", "output", "response", "answer", "completion")):
                    dropped_dupes[kind] += 1
                    dupe_docs.add(folder)
                    continue
                r.setdefault("source_file", src)
                r.setdefault("source_doc", folder)
                kept[kind].append(r)

    LOG.info("excluded %d generated document folder(s)", len(dropped_docs))
    for kind in KINDS:
        LOG.info("  %-4s kept %7d row(s), dropped %6d by document, %5d by duplicate text",
                 kind, len(kept[kind]), dropped_rows[kind], dropped_dupes[kind])
    if dupe_docs:
        LOG.warning("%d kept document(s) carried text reserved for evaluation and were "
                    "filtered row by row - the corpus holds duplicate or amended copies",
                    len(dupe_docs))

    if args.check:
        if dropped_docs:
            LOG.warning("CONTAMINATION: %d held-out document(s) have generated training data. "
                        "Train from this split, not from %s directly.",
                        len(dropped_docs), cfg["generated_dir"].name)
        else:
            LOG.info("clean - no held-out document has generated training data")
        return 0

    out_dir = Path(args.out_train) if args.out_train else cfg["out_dir"] / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, rows in kept.items():
        if rows:
            LOG.info("wrote %s", write_jsonl(out_dir / f"train_{kind}.jsonl", rows))

    write_json(out_dir / "manifest.json", {
        "version": "train-split-v1",
        "purpose_en": "Training data with all benchmark documents removed.",
        "purpose_ko": "벤치마크 문서를 제외한 학습용 데이터.",
        "generated_dir": str(cfg["generated_dir"]),
        "holdout_file": str(hp),
        "kept_documents": len(kept_docs),
        "excluded_documents": sorted(dropped_docs),
        "rows_kept": {k: len(v) for k, v in kept.items()},
        "rows_dropped": dict(dropped_rows),
        "rows_dropped_duplicate_text": dict(dropped_dupes),
        "documents_filtered_for_duplicate_text": sorted(dupe_docs),
        "exclusion_rule_en": ("A row is dropped when its document is reserved, or when any of its "
                              "text fields hashes to a reserved chunk. The second rule catches "
                              "duplicate and amended copies of a held-out document filed under "
                              "another name."),
        "exclusion_rule_ko": ("문서가 홀드아웃이거나, 행의 텍스트가 홀드아웃 청크와 동일한 해시일 때 "
                              "제외합니다. 두 번째 규칙이 다른 이름으로 중복 수집되었거나 개정 전후로 "
                              "나뉜 사본을 잡습니다."),
    })
    LOG.info("training split ready in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
