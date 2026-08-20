#!/usr/bin/env python3
"""
Self-check — a hallucination signal that needs no answer key.

    python cb.py selfcheck -m qwen3:8b --tag base-sc --tracks sft --closed-book
    python cb.py selfcheck -m qwen3:8b --tag base-sc --samples 5 --threshold 0.5

Every other track here compares an answer against a key. That only works where
a key exists, which rules out free-form output and any question nobody has
written an answer for -- which is most of what a deployed agent is asked.

The reference-free alternative, SelfCheckGPT (Manakul et al., EMNLP 2023):
sample the same question several times and see whether the model tells the same
story twice. A fact the weights hold comes back the same way; something invented
on the spot comes back different. Consistency is the signal, and no ground truth
is involved in producing it.

The answer key is still used, when the track has one, but only afterwards and
only to check the detector: `separation` reports how much higher the
inconsistency runs on answers that were in fact wrong. A detector that does not
separate is not detecting anything.
"""
from __future__ import annotations

import argparse
import collections
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from kcbench.common import (BENCHMARK_NAME, BENCHMARK_VERSION, TRACKS_HELP,
                            add_common_args, describe, items_digest, log,
                            normalise, read_jsonl, resolve_config,
                            resolve_tracks, track_label, wilson, write_json)
from kcbench.calibration import answer_key, graded
from kcbench.evaluate import (Checkpoint, TRACK_FILES, THINK_RE, build_prompt,
                              generate, run_meta)

LOG = log("selfcheck")

WORD_RE = re.compile(r"[0-9A-Za-z가-힣]+")
EXTRACTIVE = ("numeric", "label")


def body(reply: str) -> str:
    """The answer without the reasoning. Two samples reason differently on the
    way to the same answer, and comparing the reasoning measures nothing."""
    out = THINK_RE.sub("", reply)
    if "<think>" in out:
        out = out.split("<think>", 1)[0]
    return out.strip()


def tokens(text: str) -> List[str]:
    return [t for t in WORD_RE.findall(normalise(text).lower()) if len(t) > 1]


def similarity(primary: str, sample: str, item: dict) -> float:
    """
    How much of the answer under test the sample agrees with.

    Extractive items have one commitment each, so agreement is exact or nothing.
    Prose is scored by containment rather than Jaccard: a sample that says the
    same thing at greater length is agreeing, not disagreeing, and Jaccard would
    penalise it for the extra words.
    """
    if item.get("eval_type") in EXTRACTIVE:
        a, b = answer_key(primary, item), answer_key(sample, item)
        return float(a is not None and a == b)
    ta, tb = set(tokens(primary)), set(tokens(sample))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def check_item(cfg, model: str, item: dict, prompt: str, samples: int,
               temperature: float, tol: float) -> Dict[str, Any]:
    # the answer under test, decoded the way the benchmark decodes it
    primary = generate(cfg, model, prompt)
    sims = []
    for _ in range(samples):
        s = generate(cfg, model, prompt, None, temperature=temperature)
        sims.append(similarity(body(primary), body(s), item))
    consistency = statistics.fmean(sims) if sims else 0.0
    rec = {"id": item["id"], "eval_type": item.get("eval_type"),
           "category": item.get("category"),
           "consistency": round(consistency, 4),
           "inconsistency": round(1.0 - consistency, 4),
           "sample_reply": primary[:400]}
    # only for validating the detector, never for producing its score
    if item.get("eval_type") in EXTRACTIVE:
        rec["correct"] = graded(primary, item, tol)
    return rec


def summarise(points: List[dict], threshold: float) -> Dict[str, Any]:
    if not points:
        return {"n": 0}
    inc = [p["inconsistency"] for p in points]
    flagged = [p for p in points if p["inconsistency"] >= threshold]
    out: Dict[str, Any] = {
        "n": len(points),
        "mean_inconsistency": round(statistics.fmean(inc), 4),
        "median_inconsistency": round(statistics.median(inc), 4),
        "flagged": len(flagged),
        "flagged_rate": round(len(flagged) / len(points), 4),
        "flagged_rate_ci95": [round(x, 4) for x in wilson(len(flagged), len(points))],
        "threshold": threshold,
    }
    scored = [p for p in points if "correct" in p]
    if scored:
        wrong = [p["inconsistency"] for p in scored if p["correct"] < 0.5]
        right = [p["inconsistency"] for p in scored if p["correct"] >= 0.5]
        out["validation"] = {
            "n_scored": len(scored),
            "mean_inconsistency_when_wrong": round(statistics.fmean(wrong), 4) if wrong else None,
            "mean_inconsistency_when_right": round(statistics.fmean(right), 4) if right else None,
            # the number that says whether the signal is worth anything: how far
            # apart the two populations sit. At or below zero it is not a detector
            "separation": (round(statistics.fmean(wrong) - statistics.fmean(right), 4)
                           if wrong and right else None),
            # over graded items only: an ungraded item is not a right answer,
            # and counting it as one biased this number downward
            "flag_precision": (round(sum(1 for p in fs if p["correct"] < 0.5) / len(fs), 4)
                               if (fs := [p for p in flagged if "correct" in p]) else None),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reference-free hallucination signal from sampling consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("-m", "--model", required=True, help="Ollama model tag")
    ap.add_argument("-t", "--tag", help="name for this run (default: the model tag)")
    ap.add_argument("--tracks", default="sft", metavar="LIST",
                    help=f"tracks to check. {TRACKS_HELP}")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--samples", type=int, help="stochastic samples per item")
    ap.add_argument("--temperature", type=float, help="sampling temperature")
    ap.add_argument("--threshold", type=float,
                    help="inconsistency at or above which an answer is flagged")
    ap.add_argument("--limit", type=int, help="first N items per track")
    ap.add_argument("--think", choices=["on", "off"],
                    help="server-side reasoning toggle; 'off' matches an "
                         "enable_thinking=False fine-tune")
    ap.add_argument("--closed-book", action="store_true")
    ap.add_argument("--ollama-url", help="override the Ollama endpoint")
    ap.add_argument("--runs-dir", help="where run files go (default <out-dir>/runs)")
    args = ap.parse_args()

    cfg = resolve_config(args)
    if args.ollama_url:
        cfg["eval"]["ollama_base_url"] = args.ollama_url
    if args.think:
        cfg["eval"]["think"] = args.think == "on"
    sc = cfg.setdefault("selfcheck", {})
    for key in ("samples", "temperature", "threshold"):
        if getattr(args, key, None) is not None:
            sc[key] = getattr(args, key)
    describe(cfg)

    if sc["temperature"] <= 0:
        LOG.error("selfcheck needs a temperature above 0: at 0 every sample is "
                  "identical and every answer looks perfectly consistent")
        return 2

    tag = args.tag or args.model.replace(":", "-").replace("/", "-")
    runs_dir = Path(args.runs_dir) if args.runs_dir else cfg["out_dir"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tol = cfg["eval"]["numeric_tolerance"]

    files = dict(TRACK_FILES)
    for k, v in (cfg.get("usecases") or {}).items():
        if not k.startswith("_") and isinstance(v, dict) and v.get("enabled", True):
            files[k] = v.get("track_file", f"{k}.jsonl")

    result = {"benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
              "tag": tag, "model": args.model, "kind": "selfcheck",
              "lang": args.lang, "limit": args.limit,
              "book": "closed" if args.closed_book else "open",
              "samples": sc["samples"], "temperature": sc["temperature"],
              "threshold": sc["threshold"], "tracks": {}}

    started = time.time()
    for t in resolve_tracks(args.tracks):
        path = cfg["out_dir"] / files.get(t, f"{t}.jsonl")
        if not path.exists():
            LOG.warning("track %s not built (%s) - skipping", track_label(t), path.name)
            continue
        rows = read_jsonl(path)
        rows = rows[:args.limit] if args.limit else rows
        LOG.info("track %s: %d item(s)", track_label(t), len(rows))
        params = {"model": args.model, "lang": args.lang, "limit": args.limit,
                  "closed_book": bool(args.closed_book), "samples": sc["samples"],
                  "temperature": sc["temperature"], "items": len(rows)}
        ckpt = Checkpoint(runs_dir / ".ckpt" / f"{tag}-sc-{t}.jsonl", params)
        resumed = ckpt.load()
        if resumed:
            LOG.info("track %s: resuming, %d item(s) already checked",
                     track_label(t), resumed)
        ckpt.open()
        points = [ckpt.done[r["id"]] for r in rows if r["id"] in ckpt.done]
        try:
            for i, item in enumerate(rows, 1):
                if item["id"] in ckpt.done:
                    continue
                try:
                    rec = check_item(cfg, args.model, item,
                                     build_prompt(item, args.lang, args.closed_book),
                                     sc["samples"], sc["temperature"], tol)
                except Exception as exc:
                    LOG.warning("generate failed on %s (%s)", item["id"], exc)
                    continue
                points.append(rec)
                ckpt.add(rec)
                if i % 25 == 0:
                    LOG.info("%s: %d/%d", track_label(t), i, len(rows))
        finally:
            ckpt.close()
        ckpt.path.unlink(missing_ok=True)
        summary = summarise(points, sc["threshold"])
        summary["items_digest"] = items_digest(rows)
        summary["detail"] = points
        result["tracks"][t] = summary

    result["elapsed_sec"] = round(time.time() - started, 1)
    result["meta"] = run_meta(cfg)
    result["headline"] = {f"{track_label(t)}_inconsistency": v["mean_inconsistency"]
                          for t, v in result["tracks"].items() if v.get("n")}
    out = write_json(runs_dir / f"{tag}.json", result)
    LOG.info("wrote %s", out)
    for t, v in result["tracks"].items():
        if not v.get("n"):
            continue
        val = v.get("validation") or {}
        LOG.info("  %-18s mean inconsistency %.4f  flagged %.3f  separation %s",
                 track_label(t), v["mean_inconsistency"], v["flagged_rate"],
                 val.get("separation"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
