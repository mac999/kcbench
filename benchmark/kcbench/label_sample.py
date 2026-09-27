#!/usr/bin/env python3
"""
Draw a sample for hand labelling, and score a classifier against it once
the labels come back.

    python cb.py sample volatility -n 200 --sheet labels-volatility.jsonl
    python cb.py sample --auto llama3.3:70b --sheet labels-volatility.jsonl
    python cb.py sample --score labels-volatility.jsonl

Two of this benchmark's newer parts decide things no rule can check on its own:
whether an item's answer survives the next amendment, and whether a prose
answer says what the key says. Both were built with their own evidence — rule
signals in one case, a separation test in the other — and neither has been
compared against a person. Until it is, their output is a hypothesis.

The sample is stratified rather than random so the rare classes are actually
represented: a uniform draw over items that are 60% volatile and 2% stable
would return a handful of the class the classifier is least sure about.

`--auto` fills the labels with a model instead of a person. That is a silver
standard and is recorded as one: the rows carry which model produced them, and
`--score` says so in its report. It is a stopgap for a classifier that would
otherwise go unvalidated, not a substitute for the human pass — a model that
agrees with a rule may only share its blind spot.

A model may not label unless it has passed the competence check first, on
transformations whose answer is fixed by construction. That gate exists
because the obvious candidate failed it: asked whether a sentence covers
itself, one 30B model said no four times in five.

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


# Transformations whose answer is fixed by construction, used to decide
# whether a model may label at all. Three earlier versions of this gate were
# wrong — they used an LLM paraphrase as the positive case and counted the
# judges' correct rejections as errors, and they altered the first number in a
# string, which was usually a list marker rather than the figure. Nothing here
# involves a model or a judgement about meaning.
GATE_MIN = 0.80


def competence(cfg, model: str, n: int = 8) -> Dict[str, Any]:
    """
    Whether `model` may serve as a labeller, and the evidence for it.

    Identity is the case that matters: a model that cannot say a sentence
    covers itself is not strict, it is broken, and no amount of prompt work
    fixes that.
    """
    import json as _json
    import re as _re
    from kcbench.evaluate import generate
    from kcbench.prompts import render

    data = Path(cfg["out_dir"]) / "uc7_requirement.jsonl"
    if not data.is_file():
        raise SystemExit("competence needs uc7_requirement.jsonl; run cb.py requirement")
    rows = [_json.loads(l) for l in data.read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = [r for r in rows if r.get("answer_value") is not None
              and 40 <= len(r["answer"]) <= 160][:n]

    def ask(kind, gold, pred):
        out = generate(cfg, model, render(cfg, f"judge.{kind}.ko", gold=gold, pred=pred)) or ""
        return "yes" in out.lower()[:40]

    cases = []
    for r in picked:
        g, v = r["answer"], r["answer_value"]
        want = str(int(v)) if float(v) == int(v) else str(v)
        cases += [("identity", "covered", g, g, True),
                  ("spacing", "covered", g, _re.sub(r"(\S)\s+(\S)", r"\1 \2", g), True),
                  ("truncation", "covered", g, g[:len(g) // 2], False),
                  ("identity", "supported", g, g, True)]
        if want in g:
            cases.append(("figure", "supported", g,
                          g.replace(want, str(int(float(v)) + 7), 1), False))

    per: Dict[str, List[bool]] = collections.defaultdict(list)
    for name, kind, g, pd, truth in cases:
        per[name].append(ask(kind, g, pd) == truth)
    score = sum(x for v in per.values() for x in v) / max(
        sum(len(v) for v in per.values()), 1)
    return {"model": model, "n": len(picked), "score": round(score, 3),
            "by_case": {k: round(sum(v) / len(v), 2) for k, v in per.items()},
            "passed": score >= GATE_MIN, "threshold": GATE_MIN}


def self_consistency(cfg, model: str, rows: List[dict], spec: Dict[str, Any],
                     n: int = 12) -> float:
    """
    How often the model gives the same label twice for the same item.

    The competence gate tests sentence judging, which is not the task being
    authorised here, and volatility has no ground truth to test against — the
    corpus turned up three items an authority had stated were amended, which
    is not a sample. What can still be established is that the labeller is not
    answering at random: one that contradicts itself cannot corroborate a rule.
    """
    from kcbench.evaluate import generate
    from kcbench.prompts import render

    def once(r):
        out = (generate(cfg, model, render(
            cfg, "volatility.adjudicate.ko",
            context=(r.get("clause") or "") + "\n" + (r.get("doc") or ""),
            question=r.get("question_ko") or "",
            answer=str(r.get("answer")))) or "").lower()
        return next((c for c in spec["classes"] if c != "unknown" and c in out), "unsure")

    same = [once(r) == once(r) for r in rows[:n]]
    return sum(same) / max(len(same), 1)


def autolabel(cfg, model: str, rows: List[dict], spec: Dict[str, Any],
              overwrite: bool) -> int:
    """Fill empty labels with `model`, marking each as a silver label."""
    from kcbench.evaluate import generate
    from kcbench.prompts import render

    classes = [c for c in spec["classes"] if c != "unknown"]
    done = 0
    for i, r in enumerate(rows, 1):
        if str(r.get("label", "")).strip() and not overwrite:
            continue
        out = (generate(cfg, model, render(
            cfg, "volatility.adjudicate.ko",
            context=(r.get("clause") or "") + "\n" + (r.get("doc") or ""),
            question=r.get("question_ko") or "",
            answer=str(r.get("answer")))) or "").lower()
        hit = next((c for c in classes if c in out), None)
        r["label"] = hit or "unsure"
        r["label_basis"] = f"silver:{model}"
        done += 1
        if i % 25 == 0:
            LOG.info("labelled %d/%d", i, len(rows))
    return done


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
    ap.add_argument("--auto", metavar="MODEL",
                    help="fill empty labels with a model (silver standard)")
    ap.add_argument("--overwrite", action="store_true",
                    help="with --auto, relabel rows that already have a label")
    args = ap.parse_args(argv)
    cfg = resolve_config(args)
    spec = TASKS[args.task]

    if args.auto:
        sheet = Path(args.sheet or f"labels-{args.task}.jsonl")
        if not sheet.is_file():
            LOG.error("no work order at %s; draw one first", sheet)
            return 1
        gate = competence(cfg, args.auto)
        print(f"competence of {gate['model']}: {gate['score']} "
              f"({'pass' if gate['passed'] else 'fail'}, threshold {gate['threshold']})")
        for k, v in gate["by_case"].items():
            print(f"  {k:12s} {v}")
        if not gate["passed"]:
            LOG.error("%s may not label: it fails the cases whose answer is "
                      "fixed by construction", args.auto)
            return 1
        rows = [json.loads(l) for l in
                sheet.read_text(encoding="utf-8").splitlines() if l.strip()]
        sc = self_consistency(cfg, args.auto, rows, spec)
        gate["self_consistency"] = round(sc, 3)
        print(f"  self-consistency on this task: {sc:.2f}")
        if sc < 0.8:
            LOG.error("%s answers this task inconsistently (%.2f); its labels "
                      "would corroborate nothing", args.auto, sc)
            return 1
        n = autolabel(cfg, args.auto, rows, spec, args.overwrite)
        sheet.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                         encoding="utf-8")
        gate_path = sheet.with_suffix(".competence.json")
        gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nlabelled {n} row(s) as silver; evidence in {gate_path.name}")
        print(f"  python cb.py sample --score {sheet}")
        return 0

    if args.score:
        rows = [json.loads(l) for l in
                Path(args.score).read_text(encoding="utf-8").splitlines() if l.strip()]
        rep = agreement(rows, spec["field"])
        if not rep["n"]:
            LOG.error("no labels filled in yet")
            return 1
        bases = collections.Counter(r.get("label_basis", "human") for r in rows
                                    if str(r.get("label", "")).strip())
        if any(b.startswith("silver") for b in bases):
            print("labels: " + ", ".join(f"{k} {v}" for k, v in bases.items())
                  + "  — silver labels are a model's opinion, not a standard")
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
