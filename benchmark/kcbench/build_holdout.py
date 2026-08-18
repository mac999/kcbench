#!/usr/bin/env python3
"""
Reserve a slice of the dataset for evaluation and freeze it.

    python build_holdout.py
    python build_holdout.py -i /data/ai_ready_v3 -o /tmp/bench
    python build_holdout.py --config my.json --seed 42
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from kcbench.common import (BENCHMARK_NAME, BENCHMARK_VERSION, PROJECT, add_common_args,
                    corpus_path_for, describe, generated_documents, log, rel,
                    resolve_config, sha256_file, source_metadata, utc_now,
                    write_json)

LOG = log("holdout")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Reserve documents for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--fraction", type=float, metavar="F",
                    help="target share of chunks to hold out per category (default 0.10)")
    ap.add_argument("--mirror", metavar="FILE", default=str(PROJECT / "BENCHMARK_HOLDOUT.json"),
                    help="also write here, where run_corpus.py looks for exclusions")
    args = ap.parse_args(argv)

    cfg = resolve_config(args)
    if args.fraction:
        cfg["holdout"]["chunk_fraction"] = args.fraction
    describe(cfg)
    hp = cfg["holdout"]
    meta = source_metadata(cfg)

    LOG.info("reading generated documents under %s", cfg["generated_dir"])
    docs = generated_documents(cfg)
    if not docs:
        LOG.error("no generated training data found - point -i at a dataset like ai_ready_full")
        return 1
    total = sum(len(d["chunks"]) for d in docs)
    LOG.info("%d document(s), %d chunk(s)", len(docs), total)

    by_cat: dict[str, list] = collections.defaultdict(list)
    for d in docs:
        by_cat[d["category"]].append(d)

    rng = random.Random(hp["seed"])
    held = []
    for cat in sorted(by_cat):
        entries = by_cat[cat]
        target = sum(len(d["chunks"]) for d in entries) * hp["chunk_fraction"]
        eligible = [d for d in entries if len(d["chunks"]) <= hp["max_chunks_per_doc"]] or entries
        eligible = sorted(eligible, key=lambda d: d["stem"])
        rng.shuffle(eligible)
        picked, acc = [], 0
        for d in eligible:
            if acc >= target and picked:
                break
            picked.append(d)
            acc += len(d["chunks"])
        held.extend(picked or entries[:1])

    held_chunks = sum(len(d["chunks"]) for d in held)
    LOG.info("holding out %d document(s), %d chunk(s) - %.1f%% of the dataset",
             len(held), held_chunks, held_chunks / total * 100)

    pdf_rows, unresolved = [], 0
    for d in sorted(held, key=lambda d: (d["category"], d["stem"])):
        src = corpus_path_for(cfg, d)
        unresolved += src is None
        m = meta.get(src, {}) if src else {}
        pdf_rows.append({
            # Keyed on the folder path, not the bare name: some documents share
            "doc_id": "H-" + hashlib.sha1(d["generated_dir"].encode()).hexdigest()[:10],
            "stem": d["stem"],
            "path": src,
            "generated_dir": d["generated_dir"],
            "dataset_file": d["dataset_file"],
            "category": d["category"],
            "chunks": len(d["chunks"]),
            # Digests of the exact chunk texts reserved here. A training split
            "chunk_sha256": [c["sha256"] for c in d["chunk_rows"]],
            "sha256": m.get("sha256", ""),
            "title": m.get("title", d["stem"]),
            "promulgation_date": str(m.get("promulgation_date", d["source_date"])),
            "ministry": m.get("ministry", d["source_org"]),
        })
    if unresolved:
        LOG.warning("%d held-out document(s) have no source PDF in %s - they are still "
                    "excluded by stem", unresolved, cfg["corpus_dir"].name)

    # Every IFC model carries an element catalogue, so every one *could* serve as
    excluded_cats = set(hp.get("ifc_exclude_categories") or [])
    min_types = hp.get("ifc_min_element_types", 1)
    candidates, rejected = [], collections.Counter()
    for cat_file in sorted((cfg["generated_dir"] / "40_bim_models_ifc").rglob("bim_elements.json")):
        payload = json.loads(cat_file.read_text(encoding="utf-8"))
        elements = payload.get("elements") or {}
        flat = [v for v in (elements.values() if isinstance(elements, dict) else elements)
                if isinstance(v, dict)]
        if not flat:
            continue
        types = collections.Counter(e["ifc_type"] for e in flat if e.get("ifc_type"))
        model = cat_file.parent.name
        category = cat_file.parent.parent.name
        if category in excluded_cats:
            rejected["excluded category"] += 1
            continue
        if len(types) < min_types:
            rejected[f"fewer than {min_types} element types"] += 1
            continue
        src = next((s for s in (cfg["corpus_dir"] / "40_bim_models_ifc").rglob(model + ".ifc")), None)
        candidates.append({
            "doc_id": "HV-" + hashlib.sha1(model.encode()).hexdigest()[:10],
            "model": model,
            "category": category,
            "source_ifc": rel(cfg, src) if src else None,
            "catalogue_file": str(cat_file.relative_to(cfg["generated_dir"])),
            "catalogue_sha256": sha256_file(cat_file),
            "element_total": len(flat),
            "element_types": dict(types),
        })

    for reason, n in rejected.items():
        LOG.info("IFC models not eligible for evaluation - %s: %d", reason, n)

    ifc_fraction = hp.get("ifc_fraction", 0.25)
    candidates.sort(key=lambda r: r["model"])
    random.Random(hp["seed"]).shuffle(candidates)
    take = max(1, round(len(candidates) * ifc_fraction)) if candidates else 0
    ifc_rows = sorted(candidates[:take], key=lambda r: r["model"])
    if candidates:
        LOG.info("holding out %d of %d eligible IFC model(s) - the rest stay available "
                 "for VLM training", len(ifc_rows), len(candidates))

    out = {
        "version": f"{BENCHMARK_NAME}-holdout-{BENCHMARK_VERSION}",
        "built_at": utc_now(),
        "purpose_en": ("Reserved for benchmark evaluation. These documents must not be used "
                       "to train. Use make_train_split.py to build a training set without them."),
        "purpose_ko": ("벤치마크 평가 전용. 학습에 사용하면 안 됩니다. "
                       "make_train_split.py 로 이 문서들을 제외한 학습셋을 만드십시오."),
        "split_rule": (f"~{hp['chunk_fraction']:.0%} of chunks per category, documents of at most "
                       f"{hp['max_chunks_per_doc']} chunks preferred, seed {hp['seed']}"),
        "ifc_split_rule": (f"{hp.get('ifc_fraction', 0.25):.0%} of eligible models; categories "
                           f"{hp.get('ifc_exclude_categories') or []} excluded, at least "
                           f"{hp.get('ifc_min_element_types', 1)} element type(s) required"),
        "generated_dir": str(cfg["generated_dir"]),
        "totals": {"pdf_documents": len(pdf_rows), "pdf_chunks": held_chunks,
                   "dataset_documents": len(docs), "dataset_chunks": total,
                   "ifc_models": len(ifc_rows)},
        "pdf_documents": pdf_rows,
        "ifc_models": ifc_rows,
    }
    LOG.info("wrote %s", write_json(cfg["out_dir"] / "holdout.json", out))
    if args.mirror:
        LOG.info("mirrored to %s for run_corpus.py", write_json(Path(args.mirror), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
