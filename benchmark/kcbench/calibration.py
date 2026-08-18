#!/usr/bin/env python3
"""
Calibration — is the model's confidence worth anything?

    python cb.py ece -m qwen3:8b --tag base --tracks sft --closed-book
    python cb.py ece -m qwen3:8b --tag base --method verbalized --samples 1
    python cb.py ece -m my-ft:v1 --tag ft --tracks sft,uc2_rebar_spec --bins 5

A score says how often the model is right. It does not say whether the model
knows when it is wrong, and for an agent whose answer feeds a safety decision
that is the more dangerous failure: not being wrong, but being wrong and sure.

Expected Calibration Error bins predictions by stated confidence and asks, in
each bin, whether the confidence matched the accuracy. Guo et al. (2017),
"On calibration of modern neural networks", is the reference this follows, and
ISO/IEC JTC1 SC42 TS 25223 is where the vocabulary comes from.

Only items with a yes/no grader take part: numeric and label. A set-F1 answer
is partially right, and there is no accepted way to bin a partial credit against
a confidence.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kcbench.common import (BENCHMARK_NAME, BENCHMARK_VERSION, TRACKS_HELP,
                            add_common_args, describe, items_digest, log,
                            normalise, read_jsonl, resolve_config,
                            resolve_tracks, track_label, wilson, write_json)
from kcbench.evaluate import (NUM_RE, Checkpoint, TRACK_FILES, answered,
                              build_prompt, generate, grade_label,
                              grade_numeric, run_meta)

LOG = log("ece")

CONF_RE = re.compile(r"(?:confidence|확신도|신뢰도)\D{0,12}?([0-9]{1,3})\s*%?", re.I)
GRADEABLE = ("numeric", "label")


# what the model committed to, for the vote

def answer_key(reply: str, item: dict) -> str | None:
    """
    The answer a reply commits to, normalised so two samples that said the same
    thing different ways count as agreeing. None when it committed to nothing.
    """
    if not answered(reply):
        return None
    if item["eval_type"] == "numeric":
        m = NUM_RE.search(reply)
        if not m:
            return None
        try:
            return f"{float(m.group(0).replace(',', '')):.6g}"
        except ValueError:
            return None
    low = normalise(reply).lower()
    hits = [(low.find(w), w) for w in sorted(item["label_vocab"], key=len, reverse=True)
            if w in low]
    if not hits:
        return None
    pos, got = min(hits)
    for p, w in hits:
        if p == pos and len(w) > len(got):
            got = w
    return got


def graded(reply: str, item: dict, tol: float) -> float:
    if item["eval_type"] == "numeric":
        return float(grade_numeric(reply, item, tol))
    return grade_label(reply, item)["correct"]


# the two ways to get a confidence out of a model that does not report one

VERBALISED_KO = ("\n\n답 다음 줄에 그 답이 맞을 확률을 "
                 "'확신도: NN%' 형식으로 0에서 100 사이 정수로 쓰시오.")
VERBALISED_EN = ("\n\nOn the line after the answer, state the probability that "
                 "the answer is correct as 'confidence: NN%', an integer from 0 to 100.")


def confidence_by_sampling(cfg, model: str, item: dict, prompt: str,
                           samples: int, temperature: float,
                           tol: float) -> Tuple[float, float, str, int]:
    """
    Self-consistency. Sample the same question `samples` times and let the
    answers vote: confidence is the modal answer's share, the prediction is that
    modal answer. Black-box, needs no cooperation from the model, and it is what
    SelfCheckGPT-style detection rests on.

    Note the floor: with K samples the lowest reachable confidence is 1/K, so
    the bins below that stay empty by construction.
    """
    replies = []
    for _ in range(samples):
        replies.append(generate(cfg, model, prompt, None, temperature=temperature))
    keys = [answer_key(r, item) for r in replies]
    votes = collections.Counter(k for k in keys if k is not None)
    if not votes:
        return 0.0, 0.0, "", sum(1 for k in keys if k is None)
    top, n_top = votes.most_common(1)[0]
    conf = n_top / len(replies)
    winner = replies[keys.index(top)]
    return conf, graded(winner, item, tol), winner, sum(1 for k in keys if k is None)


def confidence_by_asking(cfg, model: str, item: dict, prompt: str, lang: str,
                         tol: float) -> Tuple[float, float, str, int]:
    """
    Ask the model. Cheap -- one generation -- and the number is the model's own
    claim rather than an estimate derived from its behaviour, which is both the
    appeal and the weakness: stated confidence is known to be coarse and to
    cluster on round numbers.
    """
    suffix = VERBALISED_KO if lang == "ko" else VERBALISED_EN
    reply = generate(cfg, model, prompt + suffix, None)
    m = CONF_RE.search(reply)
    if not m:
        return -1.0, graded(reply, item, tol), reply, 0
    conf = max(0.0, min(100.0, float(m.group(1)))) / 100.0
    return conf, graded(reply, item, tol), reply, 0


# the metric

def reliability(points: List[dict], bins: int) -> Dict[str, Any]:
    """
    Equal-width bins over [0, 1], the standard construction. Returns the table
    a reliability diagram is drawn from, plus ECE, MCE and the Brier score.
    """
    n = len(points)
    if not n:
        return {"n": 0}
    rows = []
    ece = mce = 0.0
    signed = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        # the last bin owns its upper edge, so confidence 1.0 is counted
        group = [p for p in points
                 if (lo <= p["confidence"] < hi) or (i == bins - 1 and p["confidence"] == 1.0)]
        if not group:
            continue
        conf = statistics.fmean(p["confidence"] for p in group)
        acc = statistics.fmean(p["correct"] for p in group)
        w = len(group) / n
        gap = acc - conf
        ece += w * abs(gap)
        signed += w * gap
        mce = max(mce, abs(gap))
        k = round(acc * len(group))
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(group),
                     "confidence": round(conf, 4), "accuracy": round(acc, 4),
                     "gap": round(gap, 4), "weight": round(w, 4),
                     "contribution": round(w * abs(gap), 4),
                     "accuracy_ci95": [round(x, 4) for x in wilson(k, len(group))]})
    brier = statistics.fmean((p["confidence"] - p["correct"]) ** 2 for p in points)
    acc = statistics.fmean(p["correct"] for p in points)
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "accuracy_ci95": [round(x, 4) for x in wilson(round(acc * n), n)],
        "mean_confidence": round(statistics.fmean(p["confidence"] for p in points), 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier": round(brier, 4),
        "signed_gap": round(signed, 4),
        # the direction is the actionable half: a model that is under-confident
        # wastes good answers, one that is over-confident is the safety problem
        "direction": ("overconfident" if signed < -0.01 else
                      "underconfident" if signed > 0.01 else "calibrated"),
        "bins": rows,
    }


def score_track(cfg, model: str, rows: List[dict], args, ckpt: Checkpoint | None) -> dict:
    tol = cfg["eval"]["numeric_tolerance"]
    cal = cfg["calibration"]
    rows = [r for r in rows if r.get("eval_type") in GRADEABLE]
    rows = rows[:args.limit] if args.limit else rows
    points = [ckpt.done[r["id"]] for r in rows if ckpt and r["id"] in ckpt.done] if ckpt else []
    skipped = 0

    for i, item in enumerate(rows, 1):
        if ckpt and item["id"] in ckpt.done:
            continue
        prompt = build_prompt(item, args.lang, args.closed_book)
        try:
            if args.method == "verbalized":
                conf, correct, reply, blank = confidence_by_asking(
                    cfg, model, item, prompt, args.lang, tol)
            else:
                conf, correct, reply, blank = confidence_by_sampling(
                    cfg, model, item, prompt, cal["samples"], cal["temperature"], tol)
        except Exception as exc:
            LOG.warning("generate failed on %s (%s)", item["id"], exc)
            continue
        if conf < 0:
            # verbalized only: no confidence in the reply. Counting it as any
            # particular number would be inventing data.
            skipped += 1
            continue
        rec = {"id": item["id"], "eval_type": item["eval_type"],
               "category": item.get("category"), "confidence": round(conf, 4),
               "correct": correct, "no_answer_samples": blank,
               "sample_reply": reply[:400]}
        points.append(rec)
        if ckpt and reply:
            ckpt.add(rec)
        if i % 25 == 0:
            LOG.info("%d/%d", i, len(rows))

    out = reliability(points, cal["bins"])
    out["skipped_no_confidence"] = skipped
    out["detail"] = points
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Expected Calibration Error over a track",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("-m", "--model", required=True, help="Ollama model tag")
    ap.add_argument("-t", "--tag", help="name for this run (default: the model tag)")
    ap.add_argument("--tracks", default="sft", metavar="LIST",
                    help=f"tracks to calibrate on. {TRACKS_HELP}. Only numeric and "
                         "label items take part")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--method", choices=["self_consistency", "verbalized"],
                    help="where the confidence comes from (default from config)")
    ap.add_argument("--samples", type=int,
                    help="samples per item for self_consistency (default from config)")
    ap.add_argument("--temperature", type=float,
                    help="sampling temperature for self_consistency (default from config)")
    ap.add_argument("--bins", type=int, help="confidence bins (default from config)")
    ap.add_argument("--limit", type=int, help="first N items per track, for a smoke test")
    ap.add_argument("--closed-book", action="store_true")
    ap.add_argument("--ollama-url", help="override the Ollama endpoint")
    ap.add_argument("--runs-dir", help="where run files go (default <out-dir>/runs)")
    args = ap.parse_args()

    cfg = resolve_config(args)
    if args.ollama_url:
        cfg["eval"]["ollama_base_url"] = args.ollama_url
    cal = cfg.setdefault("calibration", {})
    for key in ("method", "samples", "temperature", "bins"):
        if getattr(args, key, None) is not None:
            cal[key] = getattr(args, key)
    args.method = cal["method"]
    describe(cfg)

    if args.method == "self_consistency" and cal["temperature"] <= 0:
        LOG.error("self_consistency needs a temperature above 0: at 0 every sample is "
                  "identical, so every confidence is 1.0 and the ECE is meaningless")
        return 2

    tag = args.tag or args.model.replace(":", "-").replace("/", "-")
    runs_dir = Path(args.runs_dir) if args.runs_dir else cfg["out_dir"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    files = dict(TRACK_FILES)
    for k, v in (cfg.get("usecases") or {}).items():
        if not k.startswith("_") and isinstance(v, dict) and v.get("enabled", True):
            files[k] = v.get("track_file", f"{k}.jsonl")

    result = {"benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
              "tag": tag, "model": args.model, "kind": "calibration",
              "lang": args.lang, "limit": args.limit,
              "book": "closed" if args.closed_book else "open",
              "method": args.method, "samples": cal["samples"],
              "temperature": cal["temperature"], "bins": cal["bins"],
              "tracks": {}}

    started = time.time()
    for t in resolve_tracks(args.tracks):
        path = cfg["out_dir"] / files.get(t, f"{t}.jsonl")
        if not path.exists():
            LOG.warning("track %s not built (%s) - skipping", track_label(t), path.name)
            continue
        rows = read_jsonl(path)
        LOG.info("track %s: %d item(s)", track_label(t), len(rows))
        params = {"model": args.model, "lang": args.lang, "limit": args.limit,
                  "closed_book": bool(args.closed_book), "method": args.method,
                  "samples": cal["samples"], "temperature": cal["temperature"],
                  "items": len(rows)}
        ckpt = Checkpoint(runs_dir / ".ckpt" / f"{tag}-ece-{t}.jsonl", params)
        resumed = ckpt.load()
        if resumed:
            LOG.info("track %s: resuming, %d item(s) already scored",
                     track_label(t), resumed)
        ckpt.open()
        try:
            result["tracks"][t] = score_track(cfg, args.model, rows, args, ckpt)
        finally:
            ckpt.close()
        ckpt.path.unlink(missing_ok=True)
        result["tracks"][t]["items_digest"] = items_digest(
            [r for r in rows if r.get("eval_type") in GRADEABLE])

    result["elapsed_sec"] = round(time.time() - started, 1)
    result["meta"] = run_meta(cfg)
    result["headline"] = {f"{track_label(t)}_ece": v["ece"]
                          for t, v in result["tracks"].items() if v.get("n")}
    out = write_json(runs_dir / f"{tag}.json", result)
    LOG.info("wrote %s", out)
    for t, v in result["tracks"].items():
        if not v.get("n"):
            continue
        LOG.info("  %-18s ECE %.4f  MCE %.4f  Brier %.4f  acc %.4f  %s",
                 track_label(t), v["ece"], v["mce"], v["brier"], v["accuracy"],
                 v["direction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
