#!/usr/bin/env python3
"""
Draw a sample for hand labelling, and score a classifier against it once
the labels come back.

    python cb.py sample volatility -n 200 --sheet labels-volatility.jsonl
    python cb.py sample --score labels-volatility.jsonl

Two of this benchmark's newer parts decide things no rule can check on its own:
whether an item's answer survives the next amendment, and whether a prose
answer says what the key says. Both were built with their own evidence — rule
signals in one case, a separation test in the other — and neither has been
compared against a person. Until it is, their output is a hypothesis.

The sample is stratified rather than random so the rare classes are actually
represented: a uniform draw over items that are 60% volatile and 2% stable
would return a handful of the class the classifier is least sure about.

The file it writes is the work order. Each row carries what a labeller needs
and an empty `label` to fill in; nothing else in the row should be edited,
because `--score` matches on `id` and reads the classifier's own answer from
the same row.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from kcbench.common import add_common_args, log, resolve_config

LOG = log("sample")

TASKS = {
    "volatility": {
        "field": "volatility",
        "classes": ("volatile", "stable", "unknown"),
        "question": "이 문항의 답이 소관 고시·기준의 다음 개정으로 바뀔 수 있습니까?",
        "guide": ("volatile = 개정되면 답이 달라짐 (수치 한도, 기간, 비율 등) / "
                  "stable = 정의·원리·계산방법처럼 개정돼도 유지됨 / "
                  "unsure = 판단 불가"),
        "show": ("question_ko", "answer", "answer_unit", "qualifier", "doc", "clause"),
    },
}


def stratified(rows: List[dict], field: str, n: int, seed: int) -> List[dict]:
    """
    Equal shares per class, then fill from the largest.

    A classifier is judged on the classes it gets wrong, and those are usually
    the small ones. Sampling in proportion to the population would spend the
    labelling budget confirming the majority class.
    """
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[str(r.get(field))].append(r)
    for v in by.values():
        rng.shuffle(v)

    out, share = [], max(1, n // max(len(by), 1))
    for v in by.values():
        out.extend(v[:share])
    pool = [r for v in by.values() for r in v[share:]]
    rng.shuffle(pool)
    out.extend(pool[: max(0, n - len(out))])
    rng.shuffle(out)
    return out[:n]


def agreement(rows: List[dict], field: str) -> Dict[str, Any]:
    """
    Raw agreement and Cohen's kappa against the hand labels.

    Raw agreement alone flatters a classifier on a skewed set — calling
    everything volatile scores 0.60 here without distinguishing anything.
    Kappa subtracts what chance would have produced.
    """
    labelled = [(str(r.get(field)), str(r.get("label")).strip())
                for r in rows if str(r.get("label", "")).strip()]
    labelled = [(a, b) for a, b in labelled if b != "unsure"]
    # "unknown" is the rule declining to decide, not a wrong answer. Scoring it
    # as a miss understates a classifier that abstains on purpose, so accuracy
    # is reported over the decided items and coverage alongside it — the usual
    # pair for a classifier with a reject option.
    pairs = [(a, b) for a, b in labelled if a != "unknown"]
    coverage = len(pairs) / len(labelled) if labelled else 0.0
    if not pairs:
        return {"n": 0}

    n = len(pairs)
    obs = sum(1 for a, b in pairs if a == b) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    exp = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    kappa = (obs - exp) / (1 - exp) if exp < 1 else 0.0

    per = {}
    for cls in sorted(set(b for _, b in pairs)):
        tp = sum(1 for a, b in pairs if a == cls and b == cls)
        fp = sum(1 for a, b in pairs if a == cls and b != cls)
        fn = sum(1 for a, b in pairs if a != cls and b == cls)
        per[cls] = {"n": tp + fn,
                    "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    return {"n": n, "n_labelled": len(labelled), "coverage": round(coverage, 3),
            "agreement": round(obs, 3), "kappa": round(kappa, 3),
            "per_class": per,
            "confusion": dict(collections.Counter(f"{a}->{b}" for a, b in pairs))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="draw a hand-labelling sample, or score a classifier against one",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("task", nargs="?", default="volatility", choices=sorted(TASKS))
    ap.add_argument("-n", type=int, default=200, help="sample size (default 200)")
    ap.add_argument("--sheet", default=None, help="where to write the work order")
    ap.add_argument("--score", metavar="FILE",
                    help="read a completed file and report agreement")
    args = ap.parse_args(argv)
    cfg = resolve_config(args)
    spec = TASKS[args.task]

    if args.score:
        rows = [json.loads(l) for l in
                Path(args.score).read_text(encoding="utf-8").splitlines() if l.strip()]
        rep = agreement(rows, spec["field"])
        if not rep["n"]:
            LOG.error("no labels filled in yet")
            return 1
        print(f"labelled {rep['n_labelled']} of {len(rows)}; "
              f"the rule decided {rep['n']} of those (coverage {rep['coverage']})")
        print(f"  on decided items: agreement {rep['agreement']}   kappa {rep['kappa']}")
        for cls, m in rep["per_class"].items():
            print(f"  {cls:9s} n={m['n']:4d} precision {m['precision']} recall {m['recall']}")
        print("  confusion:", rep["confusion"])
        return 0

    data = Path(cfg["out_dir"])
    rows: List[dict] = []
    for p in sorted(data.glob("*.jsonl")):
        if p.name in ("provenance.jsonl", "review_queue.jsonl", "track1_dapt.jsonl"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if spec["field"] in r:
                    rows.append(r)
    if not rows:
        LOG.error("no items carry %r - run cb.py %s --write first",
                  spec["field"], args.task)
        return 1

    picked = stratified(rows, spec["field"], args.n, cfg.get("seed", 0) or 20260814)
    out = Path(args.sheet or f"labels-{args.task}.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for r in picked:
            work = {"id": r["id"], "label": "",
                    "_question": spec["question"], "_guide": spec["guide"],
                    spec["field"]: r.get(spec["field"]),
                    "rules": r.get(f"{spec['field']}_rules"),
                    **{k: r.get(k) for k in spec["show"] if r.get(k) is not None}}
            fh.write(json.dumps(work, ensure_ascii=False) + "\n")

    dist = collections.Counter(str(r.get(spec["field"])) for r in picked)
    LOG.info("wrote %s", out)
    print(f"{len(picked)} item(s), stratified by {spec['field']}: {dict(dist)}")
    print(f"\nFill in \"label\" on each row with one of "
          f"{', '.join(spec['classes'][:2])} or unsure, then:")
    print(f"  python cb.py sample --score {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
