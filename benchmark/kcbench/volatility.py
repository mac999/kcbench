#!/usr/bin/env python3
"""
Mark which items ask for a figure that a revision will change.

    python cb.py volatility                      classify and report
    python cb.py volatility --write              stamp the item files
    python cb.py volatility --tracks uc2,sft

A benchmark that reports one closed-book number is answering two questions at
once. "What is the definition of a load-bearing wall" is knowledge a model can
hold. "What is the minimum clear width" is a figure the ministry reissues, and
a model that holds it will one day be confidently wrong. Training the second
into the weights is the mistake; scoring the failure to hold it as a knowledge
gap repeats the mistake in the measurement.

So the classifier does not judge difficulty. It judges whether the answer is
the kind of thing that gets amended, and the split is reported rather than
folded into a single score: `volatile` items are measured on retrieval and
citation, `stable` items on recall.

Rules first, and only rules for now. Each one is a property of the item that a
person can check against the source, which is the same standard `cb.py verify`
holds. Where the rules disagree or say nothing the item is left `unknown`
rather than guessed at — an LLM adjudicator for that band is a separate step
and has to be validated against hand labels before its output is believed.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kcbench.common import add_common_args, describe, log, resolve_config

LOG = log("volatility")

# Titles of instruments that are reissued. A standard's name carries its own
# revision history in this corpus: 개정 (amended), 고시 (ministerial notice),
# 제정 (enacted), 시행 (in force from), and the notice number that goes with them.
REVISED_TITLE = re.compile(r"개정|고시|제정|시행|공고|훈령|예규")
NOTICE_NUMBER = re.compile(r"제\s*\d{2,4}\s*-?\s*\d*\s*호")

# Tables are amended more often than the body text around them: a numeric limit
# usually lives in 별표 and is replaced wholesale when the instrument is reissued.
TABLE_CLAUSE = re.compile(r"별표|별지|부표|서식")

# Units that quantify a requirement. A figure carrying one of these is a
# threshold someone set, not a constant of nature.
REGULATED_UNITS = {
    "mm", "cm", "m", "㎡", "m2", "㎥", "m3", "kg", "톤", "t", "kN", "MPa",
    "%", "일", "개월", "년", "명", "회", "배", "층", "시간", "분", "원",
}

# Definitional language. A clause that defines a term states what something is
# rather than how much of it is required, and survives the next amendment.
DEFINITION_CUE = re.compile(r"이란|이라\s*함은|란\s*[^\n]{0,20}말한다|정의|용어의\s*뜻")


def signals(item: Dict[str, Any]) -> Dict[str, bool]:
    """Every rule's verdict on one item, kept separate so a call can be audited."""
    doc = str(item.get("doc") or item.get("source_name") or "")
    clause = str(item.get("clause") or "")
    context = str(item.get("context") or "")
    unit = str(item.get("answer_unit") or "")
    return {
        "revised_title": bool(REVISED_TITLE.search(doc)),
        "notice_number": bool(NOTICE_NUMBER.search(doc)),
        "table_clause": bool(TABLE_CLAUSE.search(clause) or TABLE_CLAUSE.search(doc)),
        "regulated_figure": bool(item.get("answer_value") is not None
                                 or unit in REGULATED_UNITS),
        "has_qualifier": bool(item.get("qualifier")),
        # a list of required measures and a compliance judgement are as exposed
        # to an amendment as a figure is: reissuing the clause rewrites both
        "requirement_text": item.get("eval_type") in ("nameset", "verdict"),
        "definitional": bool(DEFINITION_CUE.search(context[:400])),
    }


def classify(item: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """
    Returns (volatility, routing, the rules that fired).

    A figure with a regulated unit and a threshold qualifier, published in an
    instrument that names its own amendment, is as clear a case as the corpus
    offers. A definition with no figure is the clear case the other way. The
    band between them is left alone.
    """
    s = signals(item)
    fired = [k for k, v in s.items() if v]

    amendable = s["revised_title"] or s["notice_number"] or s["table_clause"]
    quantified = (s["regulated_figure"] or s["has_qualifier"]
                  or s["requirement_text"])

    if s["definitional"] and not s["regulated_figure"]:
        # a definition in an amended instrument is still a definition
        return "stable", "parametric", fired
    if quantified and amendable:
        return "volatile", "rag", fired
    if quantified and not amendable:
        # a figure is still a figure; without a revision signal on the document
        # the call is weaker, so it is flagged rather than decided
        return "unknown", "rag", fired
    return "unknown", "parametric", fired


def classify_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        vol, route, fired = classify(r)
        out.append({**r, "volatility": vol, "routing": route,
                    "volatility_basis": "rule", "volatility_rules": fired})
    return out


def report(tagged: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_track = collections.defaultdict(collections.Counter)
    by_rule = collections.Counter()
    for r in tagged:
        by_track[r.get("usecase") or r.get("track") or "?"][r["volatility"]] += 1
        for k in r["volatility_rules"]:
            by_rule[k] += 1
    total = collections.Counter(r["volatility"] for r in tagged)
    return {"total": dict(total), "by_track": {k: dict(v) for k, v in by_track.items()},
            "rules_fired": dict(by_rule), "n": len(tagged)}


def _print(rep: Dict[str, Any]) -> None:
    n = rep["n"] or 1
    print(f"items {rep['n']}")
    for k in ("volatile", "stable", "unknown"):
        c = rep["total"].get(k, 0)
        print(f"  {k:9s} {c:5d}  {c / n:5.1%}")
    print("\nby track")
    for t, v in sorted(rep["by_track"].items()):
        tot = sum(v.values())
        print(f"  {t:16s} {tot:5d}  " +
              "  ".join(f"{k} {v.get(k, 0)}" for k in ("volatile", "stable", "unknown")))
    print("\nrules fired")
    for k, c in sorted(rep["rules_fired"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:18s} {c:5d}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="classify items by whether a revision will change the answer",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--tracks", default="uc1,uc2,uc4,uc5,uc6,sft,probe",
                    help="comma-separated item sets to classify")
    ap.add_argument("--write", action="store_true",
                    help="stamp volatility onto the item files")
    ap.add_argument("--out", metavar="FILE", help="write the report as JSON")
    args = ap.parse_args(argv)
    cfg = resolve_config(args)
    describe(cfg)

    data = Path(cfg["out_dir"])
    files = {p.stem.split("_")[0]: p for p in data.glob("*.jsonl")}
    picked = []
    for t in [x.strip() for x in args.tracks.split(",") if x.strip()]:
        hit = [p for k, p in files.items() if k.startswith(t) or t in p.stem]
        if not hit:
            LOG.warning("no item file matches %r", t)
        picked.extend(hit)

    tagged_all, per_file = [], {}
    for p in sorted(set(picked)):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if r.get("eval_type")]
        tagged = classify_rows(rows)
        per_file[p] = tagged
        tagged_all.extend(tagged)

    rep = report(tagged_all)
    _print(rep)

    if args.write:
        for p, tagged in per_file.items():
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in tagged) + "\n",
                         encoding="utf-8")
            LOG.info("stamped %d item(s) in %s", len(tagged), p.name)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
