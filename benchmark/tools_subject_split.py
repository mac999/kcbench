#!/usr/bin/env python3
"""
Split a run by whether the item's mined subject is well formed.

    python benchmark/tools_subject_split.py -c benchmark/config_us.json \
        ground_truth_us/runs/us-qwen38b-open.json

Rule-based subject extraction does not port cleanly between languages. Korean
puts a noun phrase immediately before the figure; English puts the subject
before a modal, behind prepositional openers and relative clauses -- "In areas
where ...", "a liquid having a vapor pressure not ...". After boundary-aligned
chunking and relative-clause handling, 28% of mined English subjects are still
fragments.

That is a property of the key set, not of the model, so a score over the
affected tracks mixes two things. This prints them apart. Whether the gap is
real is a question for the pooled figure over several models: on three English
models it was -0.008 (p=0.68) open book and the per-model sign was not stable.

What counts as a fragment is language-specific, so the patterns are settings
under `subject_quality`; with none configured the split is skipped rather than
guessed at.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kcbench.common import log, resolve_config, score_metrics, track_files  # noqa: E402

LOG = log("subject")


def settings(cfg):
    s = (cfg or {}).get("subject_quality") or {}
    subj = s.get("subject_in_question")
    bad = s.get("malformed") or []
    flags = re.I if s.get("ignore_case", True) else 0
    return (re.compile(subj) if subj else None,
            tuple(re.compile(p, flags) for p in bad))


def well_formed(question, subj_re, bad_res):
    """True / False for an item whose question names a subject, else None."""
    m = subj_re.search(question or "") if subj_re else None
    if not m:
        return None
    text = m.group(1) if m.groups() else m.group(0)
    return not any(rx.search(text) for rx in bad_res)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("-c", "--config", required=True)
    a = ap.parse_args()
    cfg = resolve_config(argparse.Namespace(config=a.config))
    subj_re, bad_res = settings(cfg)
    if not subj_re:
        raise SystemExit("subject_quality.subject_in_question is not set in "
                         f"{a.config}; nothing to split on")

    run = a.run if a.run.is_file() else None
    if run is None:
        for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
            if (base / a.run).is_file():
                run = base / a.run
                break
    if run is None:
        raise SystemExit(f"no such run: {a.run}")
    doc = json.loads(run.read_text(encoding="utf-8"))

    # The questions live in the item files, which the config names.
    keys = cfg["out_dir"]
    wanted = {v for k, v in track_files(cfg).items() if k != "1"}
    questions = {}
    for name in sorted(wanted):
        f = keys / name
        if not f.is_file():
            continue
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                questions[r.get("id")] = r.get("question_ko") or r.get("question") or ""

    metrics = score_metrics(cfg)
    print("%s  model=%s  book=%s" % (run.name, doc.get("model"), doc.get("book")))
    print("%-22s %18s %18s" % ("track", "clean subject", "malformed"))
    tot = {True: [0, 0.0], False: [0, 0.0]}
    for tname, body in (doc.get("tracks") or {}).items():
        buckets = {True: [], False: []}
        for d in body.get("detail") or []:
            ok = well_formed(questions.get(d.get("id"), ""), subj_re, bad_res)
            if ok is None:
                continue
            sc = d.get("score") or {}
            v = next((sc[m] for m in metrics if m in sc), None)
            if v is not None:
                buckets[ok].append(float(v))
        if not any(buckets.values()):
            continue
        cells = []
        for ok in (True, False):
            b = buckets[ok]
            cells.append("%3d  %.4f" % (len(b), sum(b) / len(b)) if b else "          -")
            tot[ok][0] += len(b)
            tot[ok][1] += sum(b)
        print("%-22s %18s %18s" % (tname, cells[0], cells[1]))
    if tot[True][0] and tot[False][0]:
        a_, b_ = tot[True][1] / tot[True][0], tot[False][1] / tot[False][0]
        print("%-22s %10d  %.4f %10d  %.4f"
              % ("ALL", tot[True][0], a_, tot[False][0], b_))
        print("\ndifference %+.4f -- read it against the pooled figure over "
              "several models, not this run alone" % (b_ - a_))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
