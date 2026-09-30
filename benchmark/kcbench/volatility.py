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
import functools
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kcbench.common import (add_common_args, describe, log, resolve_config,
                            track_files)

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

# An answer written as something that must be done or supplied. A 각 호 list of
# submissions ("설계도 1부") is as exposed to an amendment as a numeric limit.
OBLIGATION = re.compile(r"할 것|하여야|해야|이상|이하|미만|초과|이내|\d+\s*부(?:\s|$|,)")


# The patterns above are the Korean-regulation defaults. Every one of them is
# a property of the corpus, not of the method, so `config.json` replaces any
# of them under "volatility.rules" -- a different jurisdiction announces its
# amendments with different words, and a different domain measures in
# different units.
RULE_DEFAULTS: Dict[str, Any] = {
    "revised_title": REVISED_TITLE.pattern,
    "notice_number": NOTICE_NUMBER.pattern,
    "table_clause": TABLE_CLAUSE.pattern,
    "definition_cue": DEFINITION_CUE.pattern,
    "obligation": OBLIGATION.pattern,
    "regulated_units": sorted(REGULATED_UNITS),
    "requirement_eval_types": ["nameset", "verdict"],
    "definition_scan_chars": 400,
}


@functools.lru_cache(maxsize=8)
def _rules(frozen: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    over = dict(frozen)
    out: Dict[str, Any] = {}
    for k, default in RULE_DEFAULTS.items():
        v = over.get(k, default)
        if isinstance(default, str):
            try:
                out[k] = re.compile(v)
            except re.error as exc:
                LOG.error("volatility.rules.%s is not a valid regex (%s); "
                          "using the default", k, exc)
                out[k] = re.compile(default)
        elif k == "regulated_units":
            out[k] = set(v)
        else:
            out[k] = v
    return out


def rules(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """The compiled rule set, with any config override applied."""
    over = ((cfg or {}).get("volatility") or {}).get("rules") or {}
    return _rules(tuple(sorted(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in over.items())))


def _answer_text(item: Dict[str, Any]) -> str:
    a = item.get("answer_ko") or item.get("answer")
    if isinstance(a, (list, tuple)):
        return " ".join(str(x) for x in a)
    return str(a or "")


def signals(item: Dict[str, Any], cfg: Dict[str, Any] | None = None) -> Dict[str, bool]:
    """Every rule's verdict on one item, kept separate so a call can be audited."""
    r = rules(cfg)
    doc = str(item.get("doc") or item.get("source_name") or "")
    clause = str(item.get("clause") or "")
    context = str(item.get("context") or "")
    unit = str(item.get("answer_unit") or "")
    return {
        "revised_title": bool(r["revised_title"].search(doc)),
        "notice_number": bool(r["notice_number"].search(doc)),
        "table_clause": bool(r["table_clause"].search(clause)
                             or r["table_clause"].search(doc)),
        "regulated_figure": bool(item.get("answer_value") is not None
                                 or unit in r["regulated_units"]),
        "has_qualifier": bool(item.get("qualifier")),
        # a list of required measures and a compliance judgement are as exposed
        # to an amendment as a figure is: reissuing the clause rewrites both
        "requirement_text": item.get("eval_type") in r["requirement_eval_types"],
        "obligation_answer": bool(r["obligation"].search(_answer_text(item))),
        "definitional": bool(r["definition_cue"].search(
            context[:r["definition_scan_chars"]])),
    }


def classify(item: Dict[str, Any],
             cfg: Dict[str, Any] | None = None) -> Tuple[str, str, List[str]]:
    """
    Returns (volatility, routing, the rules that fired).

    A figure with a regulated unit and a threshold qualifier, published in an
    instrument that names its own amendment, is as clear a case as the corpus
    offers. A definition with no figure is the clear case the other way. The
    band between them is left alone.
    """
    s = signals(item, cfg)
    fired = [k for k, v in s.items() if v]

    amendable = s["revised_title"] or s["notice_number"] or s["table_clause"]
    quantified = (s["regulated_figure"] or s["has_qualifier"]
                  or s["requirement_text"])

    if s["definitional"] and not s["regulated_figure"] and not s["obligation_answer"]:
        # A definition in an amended instrument is still a definition. The cue
        # is read off the surrounding passage though, and a 각 호 list of
        # required documents sits under definitional phrasing often enough
        # that this branch was claiming them — 17 of 30 disagreements with an
        # independent labeller were that shape. So the answer decides: one
        # written as an obligation is a requirement whatever the passage above
        # it reads like.
        return "stable", "parametric", fired
    if quantified and amendable:
        return "volatile", "rag", fired
    if quantified and not amendable:
        # a figure is still a figure; without a revision signal on the document
        # the call is weaker, so it is flagged rather than decided
        return "unknown", "rag", fired
    return "unknown", "parametric", fired


def classify_rows(rows: List[Dict[str, Any]],
                  cfg: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        vol, route, fired = classify(r, cfg)
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


def _print(rep: Dict[str, Any], data: Path | None = None) -> None:
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
    # A classification nobody has checked is a hypothesis. Print how far it has
    # been checked next to the counts, so the two are never read apart.
    val = (data or Path(".")) / "volatility_validation.json"
    if val.is_file():
        import json as _json
        v = _json.loads(val.read_text(encoding="utf-8"))
        print(f"\nvalidation  kappa {v['kappa']} against {v['validated_against']}, "
              f"n={v['sample']}, coverage {v['coverage']}")
        print(f"  {v['caveat_en']}")
    else:
        print("\nvalidation  none — run cb.py sample to check this against labels")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="classify items by whether a revision will change the answer",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--tracks", default=None,
                    help="comma-separated item sets to classify, or all "
                         "(default: volatility.tracks in the config)")
    ap.add_argument("--write", action="store_true",
                    help="stamp volatility onto the item files")
    ap.add_argument("--out", metavar="FILE", help="write the report as JSON")
    args = ap.parse_args(argv)
    cfg = resolve_config(args)
    describe(cfg)

    data = Path(cfg["out_dir"])
    # The registered tracks, not a glob over the output directory. Only tracks
    # whose answer is text can be classified -- dapt is scored by perplexity
    # and vlm is keyed on an image -- and which those are is a property of the
    # track set, so it is configurable rather than a literal here.
    vcfg = cfg.get("volatility") or {}
    drop = set(vcfg.get("exclude_tracks", ["1", "3"]))
    known = {k: v for k, v in track_files(cfg).items() if k not in drop}
    if "2" in known:
        known["sft"] = known.pop("2")
    spec = args.tracks or vcfg.get("tracks", "all")
    wanted = [x.strip() for x in str(spec).split(",") if x.strip()] \
        if isinstance(spec, str) else list(spec)
    if wanted == ["all"]:
        wanted = sorted(known)
    picked = []
    for t in wanted:
        hit = [data / known[k] for k in known if k == t or k.startswith(t)]
        hit = [p for p in hit if p.is_file()]
        if not hit:
            LOG.warning("no item file matches %r", t)
        picked.extend(hit)

    tagged_all, per_file = [], {}
    for p in sorted(set(picked)):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if r.get("eval_type")]
        tagged = classify_rows(rows, cfg)
        per_file[p] = tagged
        tagged_all.extend(tagged)

    rep = report(tagged_all)
    _print(rep, data)

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
