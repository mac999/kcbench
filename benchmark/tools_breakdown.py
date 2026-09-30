#!/usr/bin/env python3
"""
Score a run along the two axes v0.5.2 made available, separately.

    python benchmark/tools_breakdown.py data-v052/runs/v052-open.json
    python benchmark/tools_breakdown.py v052-open.json v052-open-uc6.json

A second run file overrides the first track by track, which is how a single
track re-scored after a grader fix is read together with the run it belongs
to instead of redoing the seven tracks the fix did not touch.

The generator routes a *document* by title and metadata cues -- does this
instrument announce its own amendments. `cb.py volatility` classifies an *item*
by its content -- is the answer a figure a revision moves. They agree on 89% of
items and Cohen's kappa is 0.02, because four items in five are volatile under
both and raw agreement is mostly base rate. Reporting one as a check on the
other would be wrong; they are separate cuts and are printed as such.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "ai_ready_v052"


def routes() -> tuple[dict, dict]:
    """chunk digest -> route, and document folder -> route."""
    by_digest, by_folder = {}, {}
    for p in GEN.rglob("rag_corpus.jsonl"):
        folder, seen = str(p.parent.relative_to(GEN)), set()
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            route = (r.get("routing") or {}).get("route")
            seen.add(route)
            by_digest[hashlib.sha256((r.get("text") or "").encode()).hexdigest()] = route
        by_folder[folder] = seen.pop() if len(seen) == 1 else "mixed"
    return by_digest, by_folder


def item_axes(data: Path) -> dict:
    """id -> (generator route, rule volatility) for every text item."""
    by_digest, by_folder = routes()
    out = {}
    for f in sorted(data.glob("*.jsonl")):
        if f.name in ("provenance.jsonl", "track1_dapt.jsonl"):
            continue
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            pv = r.get("provenance") or {}
            fold = pv.get("generated_dir")
            route = (by_digest.get(pv.get("chunk_sha256")) or by_folder.get(fold)
                     or ("train" if fold and fold not in by_folder else None))
            out[r.get("id")] = (route, r.get("volatility"))
    return out


# What counts as "the score" depends on the answer type. A sentence item is
# graded on several axes at once and `grounded` is the strict one -- the answer
# states what the clause requires and claims nothing the clause does not
# support -- so it is the headline and the components are reported beside it.
HEADLINE = ("correct", "f1", "grounded")
SENTENCE_PARTS = ("semantic", "figure", "covered", "supported", "grounded")


def headline(score: dict):
    for k in HEADLINE:
        if score.get(k) is not None:
            return float(score[k]), k
    return None, None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def cut(rows, key, order):
    """Mean score and n per level of one axis, levels in a stated order."""
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r["score"])
    return [(lv, len(buckets[lv]), mean(buckets[lv]))
            for lv in order if buckets.get(lv)]


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    runs = []
    for arg in argv[1:]:
        f = Path(arg)
        if not f.is_file():
            for base in (Path.cwd(), ROOT, ROOT / "benchmark"):
                if (base / arg).is_file():
                    f = base / arg
                    break
            else:
                print(f"no such run file: {arg}")
                return 1
        runs.append(f)
    run = runs[0]
    doc = json.loads(run.read_text(encoding="utf-8"))
    for extra in runs[1:]:
        later = json.loads(extra.read_text(encoding="utf-8"))
        for tname, t in (later.get("tracks") or {}).items():
            was = "replaces" if tname in doc.get("tracks", {}) else "adds"
            print(f"# {extra.name} {was} track {tname}")
            doc.setdefault("tracks", {})[tname] = t
    axes = item_axes(run.parent.parent)

    # gold labels, for the baseline a classification track has to beat
    gold_of = {}
    for f in run.parent.parent.glob("*.jsonl"):
        if f.name in ("provenance.jsonl", "track1_dapt.jsonl"):
            continue
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("eval_type") == "verdict":
                gold_of[r.get("id")] = r.get("answer")

    rows = []
    for tname, t in (doc.get("tracks") or {}).items():
        for d in (t.get("detail") or []):
            sc = d.get("score") or {}
            val, which = headline(sc)
            if val is None:
                continue
            route, vol = axes.get(d.get("id"), (None, None))
            rows.append({"track": tname, "score": val, "metric": which,
                         "gold": gold_of.get(d.get("id")),
                         "route": route, "volatility": vol, "raw": sc,
                         "split": d.get("split"), "type": d.get("eval_type")})
    if not rows:
        print("no gradeable detail rows in", run.name)
        return 1

    print(f"{' + '.join(r.name for r in runs)}  model={doc.get('model')}  book={doc.get('book')}  n={len(rows)}")
    print(f"overall  {mean(r['score'] for r in rows):.4f}\n")

    print("by generator route  (a property of the source document)")
    for lv, n, m in cut(rows, "route", ["train", "both", "retrieve", None]):
        print("  %-10s n=%5d  %.4f" % (lv or "unresolved", n, m))

    print("\nby rule volatility  (a property of the item's answer)")
    for lv, n, m in cut(rows, "volatility", ["stable", "unknown", "volatile", None]):
        print("  %-10s n=%5d  %.4f" % (lv or "unclassified", n, m))

    print("\nby answer type")
    for lv, n, m in sorted(cut(rows, "type", sorted({r["type"] for r in rows})),
                           key=lambda x: -x[1]):
        print("  %-10s n=%5d  %.4f" % (lv, n, m))

    # A track whose reply schema offers a verdict the answer key never uses
    # scores every abstention as an error, which reads as below-chance
    # reasoning when it is abstention plus a class bias. Reported the way a
    # classifier with a reject option is: coverage first, then accuracy on
    # what was covered.
    verdicts = [r for r in rows if r["type"] == "verdict"]
    if verdicts:
        no = sum(1 for r in verdicts if r.get("raw", {}).get("no_answer"))
        committed = [r for r in verdicts if not r.get("raw", {}).get("no_answer")]
        print("\nverdict items  (n=%d)" % len(verdicts))
        print("  scored as given          %.4f" % mean(r["score"] for r in verdicts))
        # A balanced two-label track reported against zero reads as a score.
        # The number it has to beat is the constant classifier: answer the
        # commonest label every time. uc6 does not beat it in either book.
        golds = collections.Counter(r.get("gold") for r in verdicts
                                    if r.get("gold") is not None)
        if golds:
            top, n_top = golds.most_common(1)[0]
            print("  majority-class baseline  %.4f  (always %r)"
                  % (n_top / sum(golds.values()), top))
        if committed and len(committed) != len(verdicts):
            print("  coverage                 %.4f  (%d abstained)"
                  % (len(committed) / len(verdicts), no))
            print("  on committed             %.4f" % mean(r["score"] for r in committed))

    sent = [r for r in rows if r["type"] == "sentence"]
    if sent:
        print("\nsentence items, by axis  (n=%d, judges=%.1f, agreement=%.3f)" % (
            len(sent),
            mean(r["raw"].get("n_judges") for r in sent) or 0,
            mean(r["raw"].get("agreement") for r in sent) or 0))
        for k in SENTENCE_PARTS:
            m = mean(r["raw"].get(k) for r in sent)
            if m is not None:
                print("  %-10s %.4f" % (k, m))

    # Both axes correlate with answer type -- uc2 is numeric and mostly
    # `train`, uc6 is verdict and mostly `both`/`retrieve` -- so a difference
    # between levels is not attributable to the axis until it survives inside
    # a single answer type. Printed as a crosstab rather than a summary so the
    # confound is visible instead of averaged away.
    for axis, order in (("route", ["train", "both", "retrieve"]),
                        ("volatility", ["stable", "unknown", "volatile"])):
        print(f"\n{axis} within answer type   (cell: n / mean)")
        types = sorted({r["type"] for r in rows})
        print("  %-13s" % "" + "".join("%16s" % lv for lv in order))
        for ty in types:
            line = "  %-13s" % ty
            for lv in order:
                sub = [r["score"] for r in rows
                       if r["type"] == ty and r[axis] == lv]
                line += "%16s" % (f"{len(sub)} / {mean(sub):.3f}" if sub else "-")
            print(line)

    print("\nby track")
    for tname in sorted({r["track"] for r in rows}):
        sub = [r for r in rows if r["track"] == tname]
        metric = sorted({r["metric"] for r in sub})
        print("  %-18s n=%5d  %.4f  (%s)" % (tname, len(sub),
                                             mean(r["score"] for r in sub),
                                             "/".join(metric)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
