#!/usr/bin/env python3
"""
Fold a reviewer's verdicts back into the benchmark, and keep them.

    python apply_review.py --queue data/review_queue.jsonl
    python apply_review.py --queue reviewed.jsonl --report
    python apply_review.py --queue reviewed.jsonl --apply
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Dict, List

from kcbench.common import (BENCHMARK_NAME, BENCHMARK_VERSION, add_common_args, describe,
                    log, read_jsonl, resolve_config, utc_now, write_json)

LOG = log("review")

VERDICTS = ("ok", "broken", "ambiguous")


def wilson(k: int, n: int) -> tuple[float, float]:
    """
    Wilson interval for a proportion.

    The textbook normal interval puts the lower bound below zero when the count
    is small, which is exactly the regime a 50-item sample sits in.
    """
    if not n:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Apply review verdicts",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--queue", required=True, metavar="FILE",
                    help="the reviewed review_queue.jsonl")
    ap.add_argument("--reviews", metavar="FILE",
                    help="where verdicts are stored (default <out-dir>/reviews.json)")
    ap.add_argument("--apply", action="store_true", help="write the verdict store")
    ap.add_argument("--report", action="store_true", help="print the summary only")
    args = ap.parse_args()

    cfg = resolve_config(args)
    describe(cfg)

    rows = read_jsonl(Path(args.queue))
    judged = [r for r in rows if (r.get("verdict") or "").strip()]
    LOG.info("%d row(s) in the queue, %d with a verdict", len(rows), len(judged))
    if not judged:
        LOG.warning("nothing to apply - fill the verdict column first "
                    "(%s)", " / ".join(VERDICTS))
        return 0

    bad = [r for r in judged if r["verdict"].strip() not in VERDICTS]
    if bad:
        LOG.error("%d row(s) carry an unknown verdict: %s", len(bad),
                  sorted({r["verdict"] for r in bad}))
        return 1

    counts = collections.Counter(r["verdict"].strip() for r in judged)
    for v in VERDICTS:
        LOG.info("  %-10s %4d", v, counts[v])

    sample = [r for r in judged if "random_sample" in (r.get("reasons") or [])]
    triaged = [r for r in judged if "random_sample" not in (r.get("reasons") or [])]
    estimate: Dict[str, Any] = {}
    if sample:
        defects = sum(1 for r in sample if r["verdict"].strip() != "ok")
        lo, hi = wilson(defects, len(sample))
        estimate = {"sampled": len(sample), "defects": defects,
                    "rate": round(defects / len(sample), 4),
                    "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)}
        LOG.info("error rate over the random sample: %.1f%% (95%% CI %.1f-%.1f%%, n=%d)",
                 estimate["rate"] * 100, lo * 100, hi * 100, len(sample))
    else:
        LOG.warning("no random_sample rows reviewed - the set's error rate cannot be "
                    "estimated from triaged items alone, since those were chosen for "
                    "looking wrong")
    if triaged:
        kept = sum(1 for r in triaged if r["verdict"].strip() == "ok")
        LOG.info("of the %d triaged item(s), %d survived review", len(triaged), kept)

    store_path = Path(args.reviews) if args.reviews else cfg["out_dir"] / "reviews.json"
    store: Dict[str, Any] = {"items": {}}
    if store_path.exists():
        store = json.loads(store_path.read_text(encoding="utf-8"))
        store.setdefault("items", {})

    changed = 0
    for r in judged:
        entry = {"verdict": r["verdict"].strip(),
                 "note": (r.get("note") or "").strip(),
                 "reasons": r.get("reasons") or [],
                 "question_ko": r.get("question_ko"),
                 "reviewed_at": utc_now()}
        if (r.get("fixed_answer") or "").strip():
            entry["fixed_answer"] = r["fixed_answer"].strip()
        if store["items"].get(r["id"], {}).get("verdict") != entry["verdict"]:
            changed += 1
        store["items"][r["id"]] = entry

    store.update({"benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
                  "updated_at": utc_now(), "reviewed": len(store["items"]),
                  "error_rate_estimate": estimate or store.get("error_rate_estimate"),
                  "note_en": ("Verdicts are keyed by item id, which hashes the question's "
                              "content. A rebuilt item with unchanged wording keeps its "
                              "verdict; a reworded one becomes unreviewed again."),
                  "note_ko": ("검수 판정은 문항 내용 해시인 item id 로 저장됩니다. 재빌드 후에도 "
                              "문구가 같으면 판정이 유지되고, 문구가 바뀌면 미검수로 돌아갑니다.")})

    if args.report and not args.apply:
        LOG.info("report only - %s not written (%d verdict(s) would change)",
                 store_path.name, changed)
        return 0

    write_json(store_path, store)
    LOG.info("wrote %s - %d item(s) on file, %d changed", store_path, len(store["items"]), changed)
    LOG.info("build_tracks.py will drop items marked broken or ambiguous from now on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
