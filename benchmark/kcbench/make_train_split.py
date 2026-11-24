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
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator

from kcbench.common import (add_common_args, describe, log, resolve_config, sha256_text,
                    write_json, write_jsonl, generated_documents)

LOG = log("split")

# Everything the generator writes that someone could train on. star, dpo and
# rlvr are derived from the SFT samples of the same run, so a held-out
# document reaches them by the same route it reaches SFT — leaving them out of
# the split does not make them safe, it makes the leak invisible.
#
# rag_corpus.jsonl is deliberately absent: it is a retrieval index, and the
# documents in it are there precisely because the generator decided they
# should not be trained on. Splitting it would be splitting the wrong thing.
KINDS = {
    "sft": "sllm_training_data.jsonl",
    "dapt": "dapt_training_data.jsonl",
    "vlm": "vlm_training_data.jsonl",
    "star": "star_training_data.jsonl",
    "dpo": "dpo_training_data.jsonl",
    "rlvr": "rlvr_training_data.jsonl",
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


# Rows the digest check used to read: the top-level string fields. An SFT row
# keeps its passage under input.context and its answer under output.answer, a
# DPO row under chosen/rejected, so a held-out sentence inside them passed.
TOP_TEXT_FIELDS = ("text", "output", "response", "answer", "completion")
_WS = re.compile(r"\s+")


def _texts(row: dict, nested: bool, min_chars: int = 40) -> Iterator[str]:
    """Every string a model could train on in this row."""
    if not nested:
        for k in TOP_TEXT_FIELDS:
            v = row.get(k)
            if isinstance(v, str) and v:
                yield v
        return

    def walk(v):
        if isinstance(v, str):
            if len(v) >= min_chars:
                yield v
        elif isinstance(v, dict):
            for k, x in v.items():
                if k not in ("id", "images", "source_doc_ids", "domain_tags"):
                    yield from walk(x)
        elif isinstance(v, list):
            for x in v:
                yield from walk(x)
    yield from walk(row)


def _canon(text: str) -> str:
    return _WS.sub("", unicodedata.normalize("NFKC", text))


def _shingles(text: str, n: int) -> set:
    t = _canon(text)
    if len(t) < n:
        return {hash(t)} if t else set()
    return {hash(t[i:i + n]) for i in range(len(t) - n + 1)}


def held_shingles(cfg, pdf_held: set, n: int) -> set:
    """
    Character n-gram shingles of every held-out chunk, for the near-duplicate
    check. The exact digest catches a copy; it does not catch the same clause
    with a date changed, a line break moved or a particle respaced, which is
    what an amendment notice next to the text it amends looks like.
    """
    out: set = set()
    docs = 0
    for d in generated_documents(cfg):
        if d.get("generated_dir") not in pdf_held:
            continue
        docs += 1
        for c in d["chunk_rows"]:
            out |= _shingles(c["text"], n)
    LOG.info("near-duplicate index: %d shingle(s) of %d-grams from %d held-out document(s)",
             len(out), n, docs)
    return out


def contained(text: str, held: set, n: int) -> float:
    """Share of this text's shingles that occur in held-out text."""
    s = _shingles(text, n)
    if not s:
        return 0.0
    return sum(1 for x in s if x in held) / len(s)


def valid_side(folder: str, fraction: float, seed: int) -> bool:
    """Document-level validation split, decided by a hash so it is stable."""
    if fraction <= 0:
        return False
    h = hashlib.sha256(f"{seed}:{folder}".encode("utf-8")).hexdigest()[:8]
    return int(h, 16) / 16 ** 8 < fraction


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
    # Settings of the split itself. All default to the v0.5.3 behaviour, so a
    # config without a "split" section rebuilds the released training files.
    sp = cfg.get("split") or {}
    nested = bool(sp.get("nested_fields", False))
    ngram = int(sp.get("near_duplicate_ngram") or 0)
    threshold = float(sp.get("near_duplicate_threshold", 0.5))
    min_chars = int(sp.get("near_duplicate_min_chars", 60))
    valid_fraction = float(sp.get("valid_fraction", 0.0))
    valid_seed = int(sp.get("valid_seed", cfg.get("holdout", {}).get("seed", 0)))
    held_grams = held_shingles(cfg, pdf_held, ngram) if ngram else set()
    dropped_near: collections.Counter = collections.Counter()
    near_docs: set[str] = set()
    valid: dict[str, list[dict]] = {k: [] for k in KINDS}
    valid_docs: set[str] = set()

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
            to_valid = valid_side(folder, valid_fraction, valid_seed)
            for r in rows:
                texts = list(_texts(r, nested))
                # A row from a document the split kept, carrying text the split
                if held_digests and any(sha256_text(t) in held_digests for t in texts):
                    dropped_dupes[kind] += 1
                    dupe_docs.add(folder)
                    continue
                if held_grams and any(len(t) >= min_chars and
                                      contained(t, held_grams, ngram) >= threshold
                                      for t in texts):
                    dropped_near[kind] += 1
                    near_docs.add(folder)
                    continue
                r.setdefault("source_file", src)
                r.setdefault("source_doc", folder)
                if valid_fraction:
                    r["split"] = "valid" if to_valid else "train"
                if to_valid:
                    valid[kind].append(r)
                    valid_docs.add(folder)
                else:
                    kept[kind].append(r)

    LOG.info("excluded %d generated document folder(s)", len(dropped_docs))
    for kind in KINDS:
        LOG.info("  %-4s kept %7d row(s), dropped %6d by document, %5d by duplicate text",
                 kind, len(kept[kind]), dropped_rows[kind], dropped_dupes[kind])
    if dupe_docs:
        LOG.warning("%d kept document(s) carried text reserved for evaluation and were "
                    "filtered row by row - the corpus holds duplicate or amended copies",
                    len(dupe_docs))
    if near_docs:
        LOG.warning("%d kept document(s) carried near-duplicates of held-out text "
                    "(%d-gram containment >= %.2f): %s row(s) dropped",
                    len(near_docs), ngram, threshold, dict(dropped_near))
    if valid_fraction:
        LOG.info("validation split: %d document(s), rows %s",
                 len(valid_docs), {k: len(v) for k, v in valid.items()})

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
    for kind, rows in valid.items():
        if rows:
            LOG.info("wrote %s", write_jsonl(out_dir / f"valid_{kind}.jsonl", rows))

    write_json(out_dir / "manifest.json", {
        "version": "train-split-v2" if (nested or ngram or valid_fraction) else "train-split-v1",
        "split_settings": {"nested_fields": nested, "near_duplicate_ngram": ngram,
                           "near_duplicate_threshold": threshold,
                           "valid_fraction": valid_fraction, "valid_seed": valid_seed},
        "valid_documents": len(valid_docs),
        "rows_valid": {k: len(v) for k, v in valid.items()},
        "rows_dropped_near_duplicate": dict(dropped_near),
        "documents_filtered_for_near_duplicate": sorted(near_docs),
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
