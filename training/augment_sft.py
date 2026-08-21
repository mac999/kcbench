#!/usr/bin/env python3
"""
Augment the SFT pairs toward closed-book recall and complete enumeration.

    python augment_sft.py                             # defaults from the flags below
    python augment_sft.py --closed-ratio 0.5 --enum-max 3000
    python augment_sft.py --no-enum                   # closed-book mixture only

Why this exists: the generated pairs are open-book by construction -- 95% carry
the source clause in the prompt -- so training on them teaches extraction, not
recall, and the benchmark showed exactly that (closed-book flat, p~0.9). Two
transformations, both configurable:

  closed-book variants   a share of pairs duplicated with the clause removed,
                         so the same question is also asked from memory. The
                         open-book originals stay in, because extraction is a
                         skill worth keeping.
  enumeration pairs      clauses in the DAPT chunks that enumerate items become
                         "list them all" pairs whose output is the full list,
                         countering the short-single-answer style that damaged
                         nameset F1 (median output was 56 chars, 18% list-form).

Contamination: inputs are the already-split training files, which exclude the
held-out documents by digest, so everything derived here is clean by
construction. Nothing in this script reads the holdout or the evaluation sets.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "benchmark" / "data" / "train"

ITEM_RE = re.compile(r"^\s*(?:(\d{1,2})\.|([가나다라마바사아자차])\.)\s+(.+)$")
LEAD_RE = re.compile(r"(다음\s*각\s*호|다음과\s*같다|다음\s*사항|아래와\s*같다|다음\s*기준)")


def load(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def closed_variants(rows, ratio: float, rng: random.Random):
    """Duplicate a share of pairs with the clause stripped from the input."""
    out = []
    for r in rows:
        inp = r.get("input")
        ctx = (inp or {}).get("context", "") if isinstance(inp, dict) else ""
        if len(ctx.strip()) < 30 or rng.random() > ratio:
            continue
        v = json.loads(json.dumps(r, ensure_ascii=False))
        v["id"] = f"{r.get('id', 'sft')}_cb"
        v["input"] = {k: val for k, val in (inp or {}).items() if k != "context"}
        v["input"]["context"] = ""
        v["task_type"] = "regulation_recall"
        out.append(v)
    return out


def enum_pairs(chunks, max_pairs: int, min_items: int, max_per_doc: int,
               rng: random.Random):
    """
    Mine full enumerations out of the training-side chunks.

    A pair is admitted only when a lead-in sentence introduces the list and the
    list does not run to the chunk edge, for the same reasons the benchmark's
    miner demands both: without a lead-in the question is unanswerable, and a
    cut list makes the answer key incomplete.
    """
    out, per_doc = [], {}
    for c in chunks:
        doc = c.get("doc_id", "")
        if per_doc.get(doc, 0) >= max_per_doc:
            continue
        lines = c.get("text", "").splitlines()
        i = 0
        while i < len(lines):
            m = ITEM_RE.match(lines[i])
            if not m or m.group(1) not in ("1", None) and not m.group(2):
                i += 1
                continue
            # collect the run of consecutive items
            items, j = [], i
            while j < len(lines):
                mj = ITEM_RE.match(lines[j])
                if not mj:
                    break
                items.append(mj.group(3).strip())
                j += 1
            ended_inside = j < len(lines)          # the list did not hit the edge
            lead = lines[i - 1].strip() if i > 0 else ""
            if (len(items) >= min_items and ended_inside
                    and lead and LEAD_RE.search(lead)):
                out.append({
                    "id": f"enum_{c['id']}_{i}",
                    "task_type": "regulation_enumeration",
                    "doc_id": doc,
                    "source_doc_ids": [doc],
                    "instruction": f"「{doc}」에서 다음을 모두 나열하시오: {lead}",
                    "input": {"context": "", "metadata": {"language": "ko"}},
                    "output": {"answer": ", ".join(items)},
                })
                per_doc[doc] = per_doc.get(doc, 0) + 1
                if per_doc[doc] >= max_per_doc:
                    break
            i = j if j > i else i + 1
    rng.shuffle(out)
    return out[:max_pairs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sft", type=Path, default=TRAIN / "train_sft.jsonl")
    ap.add_argument("--dapt", type=Path, default=TRAIN / "train_dapt.jsonl")
    ap.add_argument("-o", "--out", type=Path, default=TRAIN / "train_sft_aug.jsonl")
    ap.add_argument("--closed-ratio", type=float, default=0.4,
                    help="share of open-book pairs also emitted as closed-book "
                         "variants (default 0.4)")
    ap.add_argument("--no-enum", action="store_true",
                    help="skip the enumeration pairs")
    ap.add_argument("--enum-max", type=int, default=4000,
                    help="cap on mined enumeration pairs (default 4000)")
    ap.add_argument("--enum-min-items", type=int, default=3,
                    help="shortest list worth asking about (default 3)")
    ap.add_argument("--enum-max-per-doc", type=int, default=6,
                    help="per-document cap, so one code does not dominate (default 6)")
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = list(load(args.sft))
    closed = closed_variants(base, args.closed_ratio, rng)
    enums = [] if args.no_enum else enum_pairs(
        load(args.dapt), args.enum_max, args.enum_min_items,
        args.enum_max_per_doc, rng)

    merged = base + closed + enums
    rng.shuffle(merged)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"open-book originals {len(base)}")
    print(f"closed-book variants {len(closed)}  (ratio {args.closed_ratio})")
    print(f"enumeration pairs    {len(enums)}")
    print(f"total -> {args.out}  {len(merged)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
