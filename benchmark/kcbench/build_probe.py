#!/usr/bin/env python3
"""
Probe set: the same questions, mined from documents the model was trained on.

    python build_probe.py
    python build_probe.py --target 400 -i /data/ai_ready_v3

Track 2 asks about held-out regulation, so 75 percent of its answers appear
nowhere in the training data and no amount of fine-tuning can teach them. That
makes it a generalisation test, which is worth having and is not the same
question as "did the model learn what we trained it on".

This set answers that one. Items come from documents on the training side of the
split, so every answer is present in the data the model saw. Contamination here
is deliberate and is the point: if closed-book accuracy does not rise on these
after training, the training did not take, whatever the loss curve says.

Never report a probe score as a benchmark result. Every row carries
split="train" and contamination="intentional" so the two cannot be confused.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Dict, List

from kcbench.build_tracks import (Rejects, _base, _instr, _nameset_candidate,
                          _numeric_candidates, _provenance, CLAUSE_RE, josa,
                          load_reviews, reviewed_out)
from kcbench.common import (QUALIFIER_EN, UNIT_EN, add_common_args, describe,
                    generated_documents, item_id, log, resolve_config, utc_now,
                    write_json, write_jsonl)

LOG = log("probe")


def train_documents(cfg, holdout) -> List[dict]:
    """Documents on the training side, in a deterministic shuffled order."""
    held = {d["generated_dir"] for d in holdout["pdf_documents"]}
    docs = [d for d in generated_documents(cfg) if d["generated_dir"] not in held]
    docs.sort(key=lambda d: d["generated_dir"])
    random.Random(cfg["probe"]["seed"]).shuffle(docs)
    return docs


def build(cfg, holdout, rej: Rejects) -> Path:
    p = cfg["probe"]
    want, cap = p["target_items"], p["max_items_per_doc"]
    # Reserve room for the nameset half up front; mining numeric first would
    # otherwise fill the quota on its own and the set would be single-shaped.
    numeric_want = want - int(want * p.get("nameset_share", 0.2))
    reviews = load_reviews(cfg)
    docs = train_documents(cfg, holdout)
    LOG.info("%d document(s) on the training side", len(docs))

    rows: List[dict] = []
    seen: set = set()

    for d in docs:
        if len(rows) >= numeric_want:
            break
        made = 0
        for chunk in d["chunk_rows"]:
            if made >= cap or len(rows) >= numeric_want:
                break
            text = chunk["text"]
            clause = CLAUSE_RE.search(text)
            for f in _numeric_candidates(text, rej, cfg):
                subject, value, unit, qual = f["subject"], f["value"], f["unit"], f["qualifier"]
                key = (subject, value, unit, qual)
                if key in seen:
                    continue
                answer_ko = f"{value} {unit}"
                if f"{value}{unit}" not in text.replace(" ", ""):
                    continue
                ident = item_id("p", chunk["sha256"], subject, value, unit, qual)
                if reviewed_out(reviews, ident, rej):
                    continue
                seen.add(key)
                topic = josa(subject, "은", "는")
                rows.append({
                    **_base("probe", "numeric", d),
                    "id": ident,
                    "split": "train",
                    "contamination": "intentional",
                    "lang": "ko",
                    "context": text,
                    "question_ko": f"{subject}{topic} 몇 {unit} {qual}이어야 하는가?",
                    "question_en": f"What is the stipulated threshold for '{subject}', "
                                   f"in {UNIT_EN[unit]} ({QUALIFIER_EN[qual]})?",
                    **_instr("numeric"),
                    "answer": answer_ko,
                    "answer_ko": answer_ko,
                    "answer_en": f"{value} {UNIT_EN[unit]}",
                    "answer_value": float(value.replace(",", "")),
                    "answer_unit": unit,
                    "answer_unit_en": UNIT_EN[unit],
                    "qualifier": qual,
                    "qualifier_en": QUALIFIER_EN[qual],
                    "clause": (f"제{clause.group(1)}조" if clause else None),
                    "verified_en": "threshold stated in a document used for training",
                    "verified_ko": "학습에 사용한 문서에 명시된 기준값.",
                    "provenance": _provenance(cfg, d, chunk, f["span"]),
                })
                made += 1
                if made >= cap or len(rows) >= numeric_want:
                    break

    numeric_n = len(rows)
    LOG.info("probe: %d numeric item(s)", numeric_n)

    share = cfg["probe"].get("nameset_share", 0.2)
    nameset_want = int(want * share)
    for d in docs:
        if len(rows) - numeric_n >= nameset_want or len(rows) >= want:
            break
        made = 0
        for n, chunk in enumerate(d["chunk_rows"]):
            if made:
                break
            prev_tail = d["chunk_rows"][n - 1]["text"][-400:] if n else ""
            found = _nameset_candidate(chunk["text"], rej, cfg, prev_tail)
            if not found:
                continue
            lead, kind, items, span, prefix = found
            key = (lead, tuple(items))
            if key in seen:
                continue
            ident = item_id("p", chunk["sha256"], lead)
            if reviewed_out(reviews, ident, rej):
                continue
            seen.add(key)
            if kind == "heading":
                q_ko = f"조문 {lead}에서 정한 항목을 모두 나열하시오."
                q_en = f"List every item stipulated in {lead} of the passage."
            else:
                q_ko = f"조문에 따르면 {lead}에 해당하는 항목을 모두 나열하시오."
                q_en = f"According to the passage, list every item that falls under '{lead}'."
            rows.append({
                **_base("probe", "nameset", d),
                "id": ident,
                "split": "train",
                "contamination": "intentional",
                "lang": "ko",
                "context": prefix + chunk["text"],
                "question_ko": q_ko,
                "question_en": q_en,
                **_instr("nameset"),
                "answer": items,
                "answer_ko": items,
                "answer_lang": "ko",
                "lead_in": lead,
                "lead_in_kind": kind,
                "verified_en": "list from a document used for training",
                "verified_ko": "학습에 사용한 문서의 각 호 목록.",
                "provenance": _provenance(cfg, d, chunk, span),
            })
            made += 1

    out = write_jsonl(cfg["out_dir"] / "probe_trained.jsonl", rows)
    kinds = collections.Counter(r["eval_type"] for r in rows)
    LOG.info("probe: %d item(s) %s over %d document(s)",
             len(rows), dict(kinds), len({r["doc"] for r in rows}))
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the probe set from training documents",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--target", type=int, help="item count (default from config)")
    ap.add_argument("--holdout", metavar="FILE",
                    help="holdout.json naming the documents to avoid")
    ap.add_argument("--report", metavar="FILE", help="rejection tally")
    args = ap.parse_args(argv)

    cfg = resolve_config(args)
    if args.target:
        cfg["probe"]["target_items"] = args.target
    describe(cfg)

    hp = Path(args.holdout) if args.holdout else cfg["out_dir"] / "holdout.json"
    if not hp.exists():
        LOG.error("run build_holdout.py first, %s is missing", hp)
        return 1
    holdout = json.loads(hp.read_text(encoding="utf-8"))

    rej = Rejects()
    build(cfg, holdout, rej)
    path = Path(args.report) if args.report else cfg["out_dir"] / "probe_rejections.json"
    write_json(path, {"built_at": utc_now(), "reports": [rej.report("probe")]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
