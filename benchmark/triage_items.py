#!/usr/bin/env python3
"""
Pick the items a human should look at, and why.

    python triage_items.py --runs v3-qwen3-8b--open,p1-glm4-9b--open
    python triage_items.py --runs-glob 'p1-*--open' --out review_queue.jsonl
    python triage_items.py --runs-glob 'p1-*--open' --sample 50
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List

from common import (BENCHMARK_NAME, BENCHMARK_VERSION, add_common_args, describe,
                    log, read_jsonl, resolve_config, utc_now, write_json,
                    write_jsonl)

LOG = log("triage")

NUM_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")
SAMPLE_SEED = 20260814


def load_runs(runs_dir: Path, names: List[str], glob: str | None) -> Dict[str, dict]:
    paths: List[Path] = []
    for n in names:
        p = Path(n) if Path(n).exists() else runs_dir / (n if n.endswith(".json") else n + ".json")
        if p.exists():
            paths.append(p)
        else:
            LOG.warning("run not found, skipping: %s", n)
    if glob:
        paths += sorted(runs_dir.glob(glob if glob.endswith(".json") else glob + ".json"))
    out = {}
    for p in dict.fromkeys(paths):
        run = json.loads(p.read_text(encoding="utf-8"))
        out[run.get("tag", p.stem)] = run
    return out


def item_scores(runs: Dict[str, dict]) -> Dict[str, Dict[str, Any]]:
    """Per item: what each model scored and what it replied."""
    acc: Dict[str, Dict[str, Any]] = {}
    for tag, run in runs.items():
        for body in (run.get("tracks") or {}).values():
            for it in body.get("detail") or []:
                rec = acc.setdefault(it["id"], {"eval_type": it["eval_type"],
                                                "category": it.get("category"),
                                                "scores": {}, "replies": {},
                                                "no_answer": {}})
                s = it["score"]
                rec["scores"][tag] = s.get("correct", s.get("f1", s.get("key_f1", 0.0)))
                rec["replies"][tag] = it.get("sample_reply", "")
                rec["no_answer"][tag] = it.get("no_answer", 0.0)
    return acc


def numbers_in(text: str) -> List[float]:
    out = []
    for m in NUM_RE.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def triage(items: Dict[str, dict], keyed: Dict[str, dict], min_models: int,
           no_answer_threshold: float = 0.5) -> List[dict]:
    queue: List[dict] = []
    for item_id, rec in items.items():
        src = keyed.get(item_id)
        if not src or len(rec["scores"]) < min_models:
            continue
        scores = list(rec["scores"].values())
        reasons: List[str] = []

        if max(scores) == 0.0:
            reasons.append("consensus_wrong")
        if statistics.fmean(rec["no_answer"].values()) >= no_answer_threshold:
            reasons.append("no_answer")

        # Different numbers, each of them present in the passage: the passage
        # supports more than one reading of the question.
        if rec["eval_type"] == "numeric":
            ctx = src.get("context", "")
            given = {n for tag, r in rec["replies"].items() for n in numbers_in(r)[:1]}
            in_ctx = {n for n in given if str(int(n) if n == int(n) else n) in ctx.replace(",", "")}
            if len(in_ctx) > 1:
                reasons.append("disagreement")

        if reasons:
            queue.append({"id": item_id, "reasons": reasons, **_row(src, rec)})
    return queue


def _row(src: dict, rec: dict) -> dict:
    return {
        "eval_type": src.get("eval_type"),
        "category": src.get("category"),
        "question_ko": src.get("question_ko"),
        "answer": src.get("answer"),
        "clause": src.get("clause"),
        "document": (src.get("provenance") or {}).get("document"),
        "dataset_file": (src.get("provenance") or {}).get("dataset_file"),
        "row_index": (src.get("provenance") or {}).get("row_index"),
        "context": src.get("context"),
        "model_scores": rec["scores"],
        "model_replies": {k: (v or "")[:200] for k, v in rec["replies"].items()},
        # Left blank for the reviewer. Filling these in is the whole job:
        "verdict": "", "fixed_answer": "", "note": "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Queue items for human review",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--runs", default="", help="comma-separated run tags to read")
    ap.add_argument("--runs-glob", help="glob over the runs directory, e.g. 'p1-*--open'")
    ap.add_argument("--runs-dir", help="where run files live (default <out-dir>/runs)")
    ap.add_argument("--tracks", default="2,3", help="tracks to triage")
    ap.add_argument("--min-models", type=int,
                    help="ignore items fewer than this many runs cover (default 2)")
    ap.add_argument("--sample", type=int,
                    help="also queue N random unflagged items, so the review can "
                         "estimate an error rate for the whole set")
    ap.add_argument("--out", metavar="FILE", help="default <out-dir>/review_queue.jsonl")
    args = ap.parse_args()

    cfg = resolve_config(args)
    describe(cfg)

    runs_dir = Path(args.runs_dir) if args.runs_dir else cfg["out_dir"] / "runs"
    runs = load_runs(runs_dir, [t.strip() for t in args.runs.split(",") if t.strip()],
                     args.runs_glob)
    if not runs:
        LOG.error("no runs to read - pass --runs or --runs-glob")
        return 1
    LOG.info("reading %d run(s): %s", len(runs), ", ".join(sorted(runs)))

    keyed: Dict[str, dict] = {}
    files = {"2": "track2_sft.jsonl", "3": "track3_vlm.jsonl"}
    for t in [x.strip() for x in args.tracks.split(",") if x.strip()]:
        path = cfg["out_dir"] / files[t]
        if path.exists():
            keyed.update({r["id"]: r for r in read_jsonl(path)})

    tr = cfg.get("triage", {})
    min_models = args.min_models if args.min_models is not None else tr.get("min_models", 2)
    sample_n = args.sample if args.sample is not None else tr.get("sample", 0)
    items = item_scores(runs)
    queue = triage(items, keyed, min_models, tr.get("no_answer_threshold", 0.5))
    flagged = {q["id"] for q in queue}

    if sample_n:
        rest = [i for i in sorted(keyed) if i in items and i not in flagged]
        rng = random.Random(tr.get("sample_seed", SAMPLE_SEED))
        for item_id in rng.sample(rest, min(sample_n, len(rest))):
            queue.append({"id": item_id, "reasons": ["random_sample"],
                          **_row(keyed[item_id], items[item_id])})

    reasons = collections.Counter(r for q in queue for r in q["reasons"])
    LOG.info("%d item(s) queued of %d scored", len(queue), len(items))
    for reason, n in reasons.most_common():
        LOG.info("  %-16s %4d", reason, n)
    if keyed:
        LOG.info("flagged share: %.1f%% of the benchmark", len(flagged) / len(keyed) * 100)

    out = Path(args.out) if args.out else cfg["out_dir"] / "review_queue.jsonl"
    write_jsonl(out, queue)
    write_json(out.with_suffix(".summary.json"), {
        "benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
        "built_at": utc_now(), "runs": sorted(runs), "items_scored": len(items),
        "queued": len(queue), "flagged": len(flagged), "reasons": dict(reasons),
        "how_to_use_en": ("Fill verdict (ok / broken / ambiguous) and fixed_answer on each row, "
                          "then feed the file to apply_review.py. Rows marked random_sample give "
                          "the error rate for the set; the flagged rows do not."),
        "how_to_use_ko": ("각 행의 verdict(ok / broken / ambiguous)와 fixed_answer 를 채운 뒤 "
                          "apply_review.py 에 넘기십시오. random_sample 행이 전체 오류율의 근거이며, "
                          "선별된 행은 전체 오류율을 대표하지 않습니다."),
    })
    LOG.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
