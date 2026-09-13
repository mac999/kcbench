"""Mine the verdict track from threshold items already in the benchmark.

A threshold item states a limit -- "5 년 이상", "40 mm 이하". Pairing that limit
with a measured figure makes a compliance judgement whose answer follows from
the qualifier, so the key is derived rather than written, the same way the rest
of the benchmark is built.

    python build_verdict.py                       # from track2_sft and uc1
    python build_verdict.py --sources track2_sft.jsonl --out verdict.jsonl

Answers follow the m1_model output contract -- verdict, answerable, value,
evidence -- so one label schema serves both the training data and this track.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from kcbench.common import add_common_args, log, read_jsonl, resolve_config, write_jsonl

LOG = log("verdict")

# Which side of the limit satisfies it. margin is applied to build the measured
# figure: a satisfying one sits inside the limit, a violating one outside.
QUALIFIERS = {
    "이상": ("ge", "at least"),
    "이하": ("le", "at most"),
    "미만": ("lt", "less than"),
    "이내": ("le", "within"),
    "초과": ("gt", "greater than"),
}

ENTAIL, CONTRADICT = "entail", "contradict"


def satisfies(measured: float, limit: float, rule: str) -> bool:
    if rule == "ge":
        return measured >= limit
    if rule == "gt":
        return measured > limit
    if rule == "le":
        return measured <= limit
    return measured < limit


def measured_for(limit: float, rule: str, want_pass: bool, rng: random.Random) -> float:
    """A figure on the required side of the limit, at a readable distance from it."""
    step = max(abs(limit) * rng.choice([0.1, 0.2, 0.3]), 1.0)
    inside = rule in ("ge", "gt")
    if want_pass:
        value = limit + step if inside else limit - step
    else:
        value = limit - step if inside else limit + step
    # A limit stated as a whole number counts something whole -- floors, days,
    # lanes -- and a measured 2.8 floors reads as a generator artefact.
    value = round(value) if float(limit).is_integer() else round(value, 2)
    # Rounding can land a strict comparison back on the limit itself.
    while satisfies(value, limit, rule) != want_pass:
        value = value + (1 if want_pass == (rule in ("ge", "gt")) else -1)
    return value


def build_item(src: dict, want_pass: bool, rng: random.Random) -> dict | None:
    qualifier = src.get("qualifier")
    limit = src.get("answer_value")
    if qualifier not in QUALIFIERS or not isinstance(limit, (int, float)):
        return None
    rule, qualifier_en = QUALIFIERS[qualifier]
    measured = measured_for(float(limit), rule, want_pass, rng)
    verdict = ENTAIL if want_pass else CONTRADICT
    unit = src.get("answer_unit") or ""
    unit_en = src.get("answer_unit_en") or ""
    clause = src.get("clause") or ""
    subject = src.get("question_ko", "")

    key = f"{src['id']}|{verdict}"
    item_id = "verdict-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    evidence = f"{src.get('doc', '')}#{clause}" if clause else src.get("doc", "")

    return {
        "benchmark": "kcbench",
        "benchmark_version": src.get("benchmark_version", "v2"),
        "schema": "kcbench-item-v2",
        "track": "usecase",
        "usecase": "uc6_verdict",
        "eval_type": "verdict",
        "cognitive_level": "application",
        "id": item_id,
        "doc": src.get("doc", ""),
        "category": src.get("category", ""),
        "split": src.get("split", "holdout"),
        "lang": "ko",
        "context": src.get("context", ""),
        "field_data": {"측정값": measured, "단위": unit},
        "question_ko": f"측정값이 {measured}{unit}로 확인되었다. 규정에 적합한지 판정하시오.",
        "question_en": (f"The measured value is {measured} {unit_en}. "
                        f"Judge whether it complies with the clause."),
        "instruction_ko": ("주어진 조문만 근거로 판정하시오. 아래 JSON 객체 하나만 반환한다. "
                           '{"answer":"판정 이유","answerable":true|false,'
                           '"verdict":"entail"|"contradict"|"neutral"|null,'
                           '"value":수치|null,"evidence":["조문 id"]}'),
        "instruction_en": ("Judge using only the clause supplied. Return one JSON object: "
                           '{"answer","answerable","verdict","value","evidence"}.'),
        "answer": verdict,
        "answer_verdict": verdict,
        "answer_answerable": True,
        "answer_value": measured,
        "answer_evidence": [evidence] if evidence else [],
        "threshold": {"limit": limit, "qualifier": qualifier,
                      "qualifier_en": qualifier_en, "unit": unit, "rule": rule},
        "subject": subject,
        "verified_ko": "조문에 명시된 기준값과 측정값의 대소 비교로 판정이 결정된다.",
        "verified_en": "The verdict follows from comparing the measured figure with the stated limit.",
        "provenance": {**(src.get("provenance") or {}), "derived_from": src["id"]},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--sources", nargs="+",
                    default=["track2_sft.jsonl", "uc1_safety_qa.jsonl"],
                    help="item files to mine thresholds from, under the output directory")
    ap.add_argument("--out", default="uc6_verdict.jsonl", help="output file name")
    ap.add_argument("--holdout-only", action="store_true",
                    help="skip items whose split is train")
    args = ap.parse_args(argv)

    cfg = resolve_config(args)
    out_dir = Path(cfg["out_dir"])
    rng = random.Random(cfg.get("holdout", {}).get("seed", 0))

    items, skipped = [], 0
    for name in args.sources:
        path = out_dir / name
        if not path.exists():
            LOG.warning("%s not found, skipping", path)
            continue
        rows = list(read_jsonl(path))
        made = 0
        for row in rows:
            if args.holdout_only and row.get("split") == "train":
                continue
            # One compliant and one violating item per threshold, so a model that
            # always answers the same way scores 0.5 rather than looking capable.
            for want_pass in (True, False):
                item = build_item(row, want_pass, rng)
                if item is None:
                    skipped += 1
                    continue
                items.append(item)
                made += 1
        LOG.info("%s: %d source row(s) -> %d item(s)", name, len(rows), made)

    rng.shuffle(items)
    write_jsonl(out_dir / args.out, items)
    entail = sum(1 for i in items if i["answer_verdict"] == ENTAIL)
    LOG.info("%d item(s) -> %s (entail %d / contradict %d), %d source row(s) unusable",
             len(items), out_dir / args.out, entail, len(items) - entail, skipped // 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
