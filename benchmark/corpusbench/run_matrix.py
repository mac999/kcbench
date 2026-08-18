#!/usr/bin/env python3
"""
Score several models and put the numbers side by side.

    python run_matrix.py --models qwen3:8b,qwen3:30b-a3b,glm4:9b
    python run_matrix.py --models-file models.txt --book both --tracks 2
    python run_matrix.py --report matrix.md --skip-existing
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from corpusbench.common import (BENCHMARK_NAME, BENCHMARK_VERSION, add_common_args, describe,
                    log, resolve_config, utc_now, write_json)

LOG = log("matrix")

BOOKS = {"open": [], "closed": ["--closed-book"]}


def tag_for(model: str, book: str, prefix: str) -> str:
    safe = model.replace(":", "-").replace("/", "-")
    return f"{prefix}{safe}--{book}"


def score(cfg, model: str, book: str, tag: str, args, runs_dir: Path) -> dict | None:
    """One evaluate.py run, as a subprocess so a crash cannot take the sweep down."""
    out = runs_dir / f"{tag}.json"
    if args.skip_existing and out.exists():
        LOG.info("%s already scored - reusing %s", model, out.name)
        return json.loads(out.read_text(encoding="utf-8"))

    cmd = [sys.executable, str(Path(__file__).parent / "evaluate.py"),
           "--model", model, "--tag", tag, "--tracks", args.tracks,
           "--lang", args.lang, *BOOKS[book]]
    if args.config:
        cmd += ["--config", args.config]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    if args.repeats:
        cmd += ["--repeats", str(args.repeats)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    LOG.info("%s (%s book)", model, book)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # One unavailable model must not cost the whole sweep: the others are
        # still worth having, and the failure is recorded in the matrix.
        LOG.error("%s failed (exit %d) - %s", model, proc.returncode,
                  (proc.stderr or proc.stdout).strip().splitlines()[-1:] or "no output")
        return None
    LOG.info("   done in %.0f s", time.time() - started)
    return json.loads(out.read_text(encoding="utf-8"))


def summarise(run: dict | None) -> Dict[str, Any]:
    """The few numbers worth putting in a cross-model table."""
    if not run:
        return {"failed": True}
    out: Dict[str, Any] = {"headline": run.get("headline", {})}
    for track, body in (run.get("tracks") or {}).items():
        for kind, metrics in (body.get("by_type") or {}).items():
            key = next((m for m in ("correct", "f1", "key_f1") if m in metrics), None)
            if key:
                out[f"t{track}.{kind}"] = metrics[key]
            if "no_answer" in metrics:
                out[f"t{track}.{kind}.no_answer"] = metrics["no_answer"]
        if body.get("perplexity") is not None:
            out[f"t{track}.perplexity"] = body["perplexity"]
    return out


def by_category(run: dict | None) -> Dict[str, float]:
    if not run:
        return {}
    acc: Dict[str, list] = {}
    for body in (run.get("tracks") or {}).values():
        for item in body.get("detail") or []:
            cat = item.get("category")
            if not cat:
                continue
            s = item["score"]
            acc.setdefault(cat, []).append(s.get("correct", s.get("f1", s.get("key_f1", 0.0))))
    return {k: round(statistics.fmean(v), 4) for k, v in sorted(acc.items())}


def render(matrix: dict) -> str:
    models: List[str] = matrix["models"]
    books: List[str] = matrix["books"]
    rows = matrix["results"]

    metrics = sorted({m for mo in models for b in books
                      for m in (rows.get(mo, {}).get(b) or {}) if m != "headline"})

    out = [f"# {BENCHMARK_NAME} {BENCHMARK_VERSION} — model matrix", "",
           f"- built: {matrix['built_at']}",
           f"- items: {matrix.get('items')}, repeats: {matrix.get('repeats')}, "
           f"language: {matrix.get('lang')}", ""]

    for book in books:
        out += [f"## {book} book", "",
                "| Model | " + " | ".join(metrics) + " |",
                "|---|" + "---:|" * len(metrics)]
        for mo in models:
            cell = rows.get(mo, {}).get(book) or {}
            if cell.get("failed"):
                out.append(f"| `{mo}` | " + " | ".join(["—"] * len(metrics)) + " |")
                continue
            out.append(f"| `{mo}` | " + " | ".join(
                f"{cell[m]:.3f}" if isinstance(cell.get(m), (int, float)) else "—"
                for m in metrics) + " |")
        out.append("")

    if len(books) == 2 and matrix.get("gaps"):
        out += ["## Headroom — open minus closed", "",
                "The part of the score that comes from having the clause in the prompt "
                "rather than from knowing it. This is what fine-tuning on the corpus has "
                "to close, and the largest number here is where to aim first.", "",
                "지문을 주었을 때와 주지 않았을 때의 차이입니다. 파인튜닝이 메워야 할 "
                "공간이며, 이 값이 큰 항목부터 손대는 것이 순서입니다.", "",
                "| Model | " + " | ".join(metrics) + " |",
                "|---|" + "---:|" * len(metrics)]
        for mo in models:
            g = matrix["gaps"].get(mo, {})
            out.append(f"| `{mo}` | " + " | ".join(
                f"{g[m]:+.3f}" if isinstance(g.get(m), (int, float)) else "—"
                for m in metrics) + " |")
        out.append("")

    cats = matrix.get("by_category") or {}
    if cats:
        book = books[-1]
        all_cats = sorted({c for mo in cats for c in (cats[mo].get(book) or {})})
        if all_cats:
            out += [f"## By category — {book} book", "",
                    "| Model | " + " | ".join(all_cats) + " |",
                    "|---|" + "---:|" * len(all_cats)]
            for mo in models:
                row = (cats.get(mo) or {}).get(book) or {}
                out.append(f"| `{mo}` | " + " | ".join(
                    f"{row[c]:.3f}" if c in row else "—" for c in all_cats) + " |")
            out.append("")

    out += ["## Reading this / 해석", "",
            "Absolute values are not comparable to published leaderboards — the items are "
            "rule-mined from held-out Korean regulation, not written and reviewed by "
            "engineers. What is comparable is one model against another on identical "
            "items, and one checkpoint against its own fine-tune.",
            "",
            "Use `compare.py --base <tag> --after <tag>` for a paired significance test "
            "before calling any difference an improvement.",
            "",
            "절대값은 공개 리더보드와 비교할 수 없습니다. 동일 문항에서의 모델 간 비교, "
            "그리고 파인튜닝 전후 비교가 이 표의 용도입니다. 차이를 개선이라고 부르기 전에 "
            "`compare.py` 로 대응 유의성 검정을 거치십시오."]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score several models side by side",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("-m", "--models", help="comma-separated model tags")
    ap.add_argument("--models-file", metavar="FILE",
                    help="one model tag per line; # starts a comment")
    ap.add_argument("--book", choices=["open", "closed", "both"],
                    help="open book gives the clause, closed book does not (default both)")
    ap.add_argument("--tracks", default="2", help="tracks to run (default 2)")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--repeats", type=int, help="samples per item")
    ap.add_argument("--limit", type=int, help="first N items per track")
    ap.add_argument("--prefix", default="", help="prepended to every run tag")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse a run file that is already there")
    ap.add_argument("--report", metavar="FILE", help="markdown output (default <out-dir>/matrix.md)")
    args = ap.parse_args()

    cfg = resolve_config(args)
    describe(cfg)

    models: List[str] = []
    if args.models:
        models += [m.strip() for m in args.models.split(",") if m.strip()]
    if args.models_file:
        for line in Path(args.models_file).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                models.append(line)
    if not models:
        LOG.error("no models given - use --models or --models-file")
        return 1

    book = args.book or cfg["eval"].get("book", "both")
    books = ["open", "closed"] if book == "both" else [book]
    runs_dir = cfg["out_dir"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    cats: Dict[str, Dict[str, Any]] = {}
    for model in models:
        results[model], cats[model] = {}, {}
        for book in books:
            run = score(cfg, model, book, tag_for(model, book, args.prefix), args, runs_dir)
            results[model][book] = summarise(run)
            cats[model][book] = by_category(run)

    gaps: Dict[str, Dict[str, float]] = {}
    if len(books) == 2:
        for model in models:
            o, c = results[model].get("open") or {}, results[model].get("closed") or {}
            gaps[model] = {k: round(o[k] - c[k], 4) for k in o
                           if isinstance(o.get(k), (int, float)) and isinstance(c.get(k), (int, float))}

    items = None
    track_file = cfg["out_dir"] / f"track{args.tracks.split(',')[0]}_sft.jsonl"
    if track_file.exists():
        items = sum(1 for _ in track_file.open(encoding="utf-8"))

    matrix = {"benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
              "built_at": utc_now(), "models": models, "books": books,
              "tracks": args.tracks, "lang": args.lang,
              "repeats": args.repeats or cfg["eval"]["repeats"], "items": items,
              "results": results, "gaps": gaps, "by_category": cats}
    write_json(cfg["out_dir"] / "matrix.json", matrix)

    report = Path(args.report) if args.report else cfg["out_dir"] / "matrix.md"
    report.write_text(render(matrix), encoding="utf-8")
    LOG.info("wrote %s and %s", cfg["out_dir"] / "matrix.json", report)

    for model in models:
        for book in books:
            head = (results[model][book] or {}).get("headline") or {}
            LOG.info("  %-24s %-6s %s", model, book,
                     " ".join(f"{k}={v}" for k, v in head.items()) or "failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
