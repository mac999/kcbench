#!/usr/bin/env python3
"""
Score one model against the benchmark.

    python evaluate.py --model qwen3:8b --tag base
    python evaluate.py --model my-ft:latest --tag sft-v1 --tracks 2,3
    python evaluate.py --model qwen3:8b --lang en --repeats 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from kcbench.common import (TRACKS_HELP, add_common_args, describe, log, normalise,
                            read_jsonl, resolve_tracks, track_label,
                    resolve_config, write_json)

LOG = log("eval")

NUM_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")
IFC_RE = re.compile(r"\bIfc[A-Za-z]+\b")


# model access

def generate(cfg, model: str, prompt: str, images: List[str] | None = None) -> str:
    # No "think" flag. Setting it false on a thinking model does not stop the
    payload: Dict[str, Any] = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": cfg["eval"]["temperature"],
                    "num_predict": cfg["eval"]["num_predict"],
                    "num_ctx": cfg["eval"]["num_ctx"]},
    }
    if images:
        payload["images"] = images
    r = requests.post(f"{cfg['eval']['ollama_base_url']}/api/generate",
                      json=payload, timeout=cfg["eval"]["request_timeout"])
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _score_prompt(cfg, model: str, text: str, timeout: float | None = None) -> dict:
    """
    Ask the server to score `text` itself and hand back its logprobs.

    num_predict must stay at 1, not 0: Ollama accepts 0 and then never answers,
    which is how a track 1 run turns into three requests hanging for minutes and
    a box deep in swap. One token is thrown away; the prompt is what matters.
    """
    r = requests.post(f"{cfg['eval']['ollama_base_url']}/api/generate",
                      json={"model": model, "prompt": text, "stream": False, "raw": True,
                            "logprobs": True, "top_logprobs": 1,
                            "options": {"num_predict": 1, "temperature": cfg["eval"]["temperature"],
                                        "num_ctx": cfg["eval"]["num_ctx"]}},
                      timeout=timeout or cfg["eval"]["request_timeout"])
    r.raise_for_status()
    return r.json()


def prompt_logprobs_available(cfg, model: str) -> bool:
    """
    Whether the backend returns a logprob per *prompt* token.

    Perplexity over held-out text needs the likelihood the model assigns to that
    text. Ollama (0.32) returns logprobs only for tokens it generated, which is
    the likelihood of its own continuation — a different quantity, and scoring
    it would silently answer a question nobody asked. Probed once rather than
    per item, because the failure is a property of the server, not the chunk.
    """
    try:
        body = _score_prompt(cfg, model, "건축물의 높이는 12미터 이상이어야 한다.", timeout=120)
    except Exception as exc:
        LOG.warning("prompt scoring probe failed (%s)", exc)
        return False
    if body.get("prompt_logprobs"):
        return True
    LOG.warning("%s returns no prompt_logprobs - %s scores generated tokens only",
                cfg["eval"]["ollama_base_url"], "this backend")
    return False


def perplexity(cfg, model: str, text: str) -> float | None:
    """Perplexity of held-out text: exp of the mean negative log-likelihood."""
    body = _score_prompt(cfg, model, text)
    lps = [t.get("logprob") for t in body.get("prompt_logprobs") or [] if t.get("logprob") is not None]
    if not lps:
        return None
    return math.exp(-statistics.fmean(lps))


# grading

def grade_numeric(reply: str, item: dict, tol: float) -> bool:
    """First number in the reply, compared with relative tolerance."""
    m = NUM_RE.search(reply)
    if not m:
        return False
    try:
        got = float(m.group(0).replace(",", ""))
    except ValueError:
        return False
    want = item["answer_value"]
    return abs(got - want) <= abs(want) * tol + 1e-9


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def answered(reply: str) -> bool:
    """
    Did the model actually answer, as opposed to returning nothing?

    A reasoning model that emits `<think></think>` and stops has answered
    nothing, but the string is not empty, so a plain emptiness test counts it as
    an answer and the run reports a knowledge failure where there was a
    generation failure. Strip the reasoning block before deciding. An unclosed
    `<think>` means the token budget ran out mid-thought, which is the same
    thing: no answer arrived.
    """
    body = THINK_RE.sub("", reply)
    if "<think>" in body:
        body = body.split("<think>", 1)[0]
    return bool(body.strip())


def _lines(reply: str) -> List[str]:
    out = []
    for raw in reply.splitlines():
        s = re.sub(r"^\s*(?:[-*•]|\d{1,2}[.)]|[가나다라마바사아자차][.)])\s*", "", raw).strip()
        if s:
            out.append(s)
    return out


def grade_nameset(reply: str, item: dict) -> Dict[str, float]:
    """
    Set precision, recall, F1 — a partial answer should score partially.

    Compared in normalised form. Regulation is copied out of PDFs with
    full-width punctuation and inconsistent spacing, so a raw string comparison
    scores a correct entry wrong for a trailing full stop and ends up measuring
    transcription.
    """
    want_raw = [str(a).strip() for a in item["answer"]]
    is_ifc = any(w.startswith("Ifc") for w in want_raw)
    # For IFC questions, take the class names wherever they sit in the sentence;
    # for regulation lists, take the lines.
    got_raw = IFC_RE.findall(reply) if is_ifc else _lines(reply)
    if item.get("match_mode") == "fuzzy":
        return _nameset_fuzzy(got_raw, want_raw)
    want = {w if is_ifc else normalise(w) for w in want_raw}
    got = {g if is_ifc else normalise(g) for g in got_raw if g.strip()}
    if not got:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit = len(want & got)
    p = hit / len(got)
    r = hit / len(want) if want else 0.0
    return {"precision": p, "recall": r, "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}


_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def _nameset_fuzzy(got_raw: List[str], want_raw: List[str]) -> Dict[str, float]:
    """
    Token-coverage matching for items whose answers are prose measures.

    An incident item's answer is a clause-length sentence; models abbreviate it
    or prefix the clause number, and both are right answers a person would
    accept. A predicted line counts as a hit when it covers most of one gold
    item's content words. Greedy one-to-one, so a line cannot claim two golds.
    """
    want_toks = [set(_WORD_RE.findall(normalise(w))) for w in want_raw]
    got_toks = [set(_WORD_RE.findall(normalise(g))) for g in got_raw if g.strip()]
    if not got_toks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    claimed: set = set()
    hit = 0
    for g in got_toks:
        best, best_cov = None, 0.0
        for i, w in enumerate(want_toks):
            if i in claimed or not w:
                continue
            cov = len(g & w) / len(w)
            if cov > best_cov:
                best, best_cov = i, cov
        if best is not None and best_cov >= 0.6:
            claimed.add(best)
            hit += 1
    p = hit / len(got_toks)
    r = hit / len(want_toks) if want_toks else 0.0
    return {"precision": p, "recall": r, "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}


ABSTAIN_RE = re.compile(r"자료\s*없음|확인할\s*수\s*없|답할\s*수\s*없|알\s*수\s*없|no\s+data", re.I)


def grade_label(reply: str, item: dict) -> Dict[str, float]:
    """
    First vocabulary word in the reply is the model's verdict.

    Positional, not membership, because the vocabulary overlaps itself:
    'partial_match' contains 'match', and a reply naming several labels has to
    be read as its first commitment rather than scored on whichever happens to
    match the key.
    """
    low = reply.lower()
    hits = [(low.find(w), w) for w in sorted(item["label_vocab"], key=len, reverse=True)
            if w in low]
    if not hits:
        return {"correct": 0.0}
    pos, got = min(hits)
    # A longer label starting at the same offset wins: 'partial_match' over 'match'.
    for p, w in hits:
        if p == pos and len(w) > len(got):
            got = w
    return {"correct": float(got == item["answer"])}


def grade_faithfulness(reply: str, item: dict, tol: float) -> Dict[str, float]:
    """
    Swapped context: correct means abstaining. Matched context: correct means
    answering, and answering right. Abstention is tracked separately so the
    aggregate can show a model that abstains on everything, which the accuracy
    alone would half-reward.
    """
    abstained = bool(ABSTAIN_RE.search(reply))
    if item.get("context_matches"):
        correct = (not abstained) and grade_numeric(reply, item, tol)
    else:
        correct = abstained
    return {"correct": float(correct), "abstained": float(abstained)}


def grade_mapping(reply: str, item: dict, tol: float) -> Dict[str, float]:
    """Key coverage and per-key value accuracy, reported separately."""
    want: Dict[str, Any] = item["answer"]
    got: Dict[str, float] = {}
    for line in reply.splitlines():
        m = re.match(r"\s*[-*•]?\s*([A-Za-z가-힣][\w가-힣 ]*?)\s*[:=]\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", line)
        if m:
            got[m.group(1).strip()] = float(m.group(2).replace(",", ""))
    if not got:
        return {"key_f1": 0.0, "value_accuracy": 0.0}
    hit = set(want) & set(got)
    p, r = len(hit) / len(got), len(hit) / len(want)
    exact = sum(1 for k in hit if abs(got[k] - float(want[k])) <= abs(float(want[k])) * tol + 1e-9)
    return {"key_f1": 0.0 if p + r == 0 else 2 * p * r / (p + r),
            "value_accuracy": exact / len(want) if want else 0.0}


# prompts

def build_prompt(item: dict, lang: str, closed_book: bool = False) -> str:
    """
    The prompt as the model sees it.

    With the passage attached, the task is extraction: the answer is on the page
    and a competent reader finds it whatever it has been trained on. That makes
    the open-book score a poor instrument for "did domain training help" — there
    is little for training to add. Closed book asks the same keyed question with
    the passage withheld, which is the form where domain knowledge is the only
    way to answer. Both are scored against the same answer key.
    """
    q = item.get(f"question_{lang}") or item.get("question_ko") or item.get("question", "")
    instr = item.get(f"instruction_{lang}") or item.get("instruction_ko", "")
    ctx = None if closed_book else item.get("context")
    # An incident scenario is part of the question, not the passage: it stays
    # in the prompt even closed book, where the clause is what is withheld.
    scenario = item.get("scenario_ko")
    if scenario:
        q = f"{scenario}\n\n{q}"
    head = "다음 조문을 읽고 질문에 답하시오." if lang == "ko" else "Read the passage and answer the question."
    if ctx:
        return f"{head}\n\n[{'조문' if lang == 'ko' else 'Passage'}]\n{ctx}\n\n[{'질문' if lang == 'ko' else 'Question'}]\n{q}\n\n{instr}"

    # Without the passage the question needs to say which document it is about,
    doc, clause = item.get("doc"), item.get("clause")
    if doc:
        where = f"「{doc}」" + (f" {clause}" if clause else "")
        q = (f"{where}에 따르면, {q}" if lang == "ko"
             else f"According to {where}: {q}")
    return f"{q}\n\n{instr}"


def b64_image(path: Path) -> str:
    import base64
    return base64.b64encode(path.read_bytes()).decode()


# run

def run_track1(cfg, model: str, rows: List[dict], limit: int | None) -> dict:
    rows = rows[:limit] if limit else rows
    if not prompt_logprobs_available(cfg, model):
        LOG.warning("dapt skipped: this backend cannot score held-out text. "
                    "Measure perplexity with a runner that returns prompt logprobs "
                    "(vLLM, llama-perplexity) against the same track1_dapt.jsonl.")
        return {"items": len(rows), "scored": 0, "perplexity": None,
                "skipped": "backend returns no prompt logprobs"}
    vals = []
    for i, r in enumerate(rows, 1):
        try:
            pp = perplexity(cfg, model, r["text"])
        except Exception as exc:
            LOG.warning("perplexity failed on %s (%s)", r["id"], exc)
            pp = None
        if pp and math.isfinite(pp):
            vals.append(pp)
        if i % 50 == 0:
            LOG.info("dapt: %d/%d", i, len(rows))
    if not vals:
        LOG.warning("dapt produced no perplexities - the server may not return prompt logprobs")
        return {"items": len(rows), "scored": 0, "perplexity": None}
    return {"items": len(rows), "scored": len(vals),
            "perplexity": round(statistics.fmean(vals), 4),
            "perplexity_median": round(statistics.median(vals), 4)}


def item_lang(item: dict, lang: str, mix: Dict[str, float]) -> str:
    """
    The prompt language for one item.

    "mix" draws per item from the configured ratio, keyed on the item id rather
    than a random stream: the same item is asked in the same language in every
    run, so a before/after comparison never mixes a language change into a
    training effect.
    """
    if lang != "mix":
        return lang
    h = int(hashlib.sha1(item["id"].encode("utf-8")).hexdigest()[:8], 16) % 100
    return "ko" if h < int(round(100 * mix.get("ko", 0.8))) else "en"


class Checkpoint:
    """
    Per-item journal so a run that dies part way through a track picks up where
    it stopped instead of starting over. Only items that got at least one
    non-empty reply are journalled: a record produced by a server that had gone
    away would otherwise survive the restart and be scored as a wrong answer,
    which is the failure this exists to prevent.
    """

    def __init__(self, path: Path, params: dict):
        self.path = path
        self.params = params
        self.done: Dict[str, dict] = {}
        self.fh = None

    def load(self) -> int:
        if not self.path.exists():
            return 0
        head, records = None, {}
        with self.path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # a torn trailing line is what a killed process leaves
                    LOG.warning("checkpoint %s: dropping malformed line %d",
                                self.path.name, n)
                    continue
                if "_params" in obj:
                    head = obj["_params"]
                elif obj.get("id"):
                    records[obj["id"]] = obj
        if head != self.params:
            stale = self.path.with_suffix(".stale")
            LOG.warning("checkpoint %s was written under different settings; "
                        "moving it to %s and starting over", self.path.name, stale.name)
            self.path.replace(stale)
            return 0
        self.done = records
        return len(records)

    def open(self) -> None:
        fresh = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")
        if fresh:
            self._write({"_params": self.params})

    def _write(self, obj: dict) -> None:
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def add(self, rec: dict) -> None:
        if self.fh:
            self._write(rec)

    def close(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None


def run_qa(cfg, model: str, rows: List[dict], lang: str, repeats: int,
           limit: int | None, image_root: Path | None,
           closed_book: bool = False, ckpt: Checkpoint | None = None) -> dict:
    rows = rows[:limit] if limit else rows
    tol = cfg["eval"]["numeric_tolerance"]
    mix = cfg["eval"].get("lang_mix") or {"ko": 0.8, "en": 0.2}
    # A dead inference server fails every call, and the empty replies grade as
    # wrong: the track finishes and writes a score that is really the blank-reply
    # rate. Consecutive failures are the signal -- an occasional timeout resets
    # the count, a server that went away never does.
    max_fails = cfg["eval"].get("max_consecutive_failures", 20)
    # A server that answers 200 with an empty body trips none of the above:
    # every call "succeeds" and every answer grades as wrong. Blank replies in
    # a row are the second signal for the same dead-server case.
    max_empty = cfg["eval"].get("max_consecutive_empty", 25)
    fails = 0
    empties = 0
    per_item = [ckpt.done[r["id"]] for r in rows
                if ckpt and r["id"] in ckpt.done] if ckpt else []

    for i, item in enumerate(rows, 1):
        if ckpt and item["id"] in ckpt.done:
            continue
        images = None
        # "images" is a list in prompt order (UC3 sends render then photo);
        # "image" is the older single-photo field from track 3.
        wanted = item.get("images") or ([item["image"]] if item.get("image") else [])
        if image_root and wanted:
            paths = [image_root / w for w in wanted]
            missing = [p for p in paths if not p.exists()]
            if missing:
                LOG.warning("image missing, skipping %s: %s", item["id"], missing[0])
                continue
            images = [b64_image(p) for p in paths]

        asked = item_lang(item, lang, mix)
        scores, replies = [], []
        for _ in range(repeats):
            try:
                reply = generate(cfg, model, build_prompt(item, asked, closed_book), images)
            except Exception as exc:
                LOG.warning("generate failed on %s (%s)", item["id"], exc)
                reply = ""
                fails += 1
                if max_fails and fails >= max_fails:
                    raise RuntimeError(
                        f"{fails} consecutive generate failures at item {i}/{len(rows)}; "
                        f"aborting so a partial track is not scored") from exc
            else:
                fails = 0
            replies.append(reply)
            kind = item["eval_type"]
            if kind == "numeric":
                scores.append({"correct": float(grade_numeric(reply, item, tol))})
            elif kind == "nameset":
                scores.append(grade_nameset(reply, item))
            elif kind == "label":
                scores.append(grade_label(reply, item))
            elif kind == "faithfulness":
                scores.append(grade_faithfulness(reply, item, tol))
            else:
                scores.append(grade_mapping(reply, item, tol))

        keys = scores[0].keys()
        # Subgroup keys travel with the score so the aggregate can split a
        # use-case track by what matters: contaminated vs held-out items,
        # swapped vs control contexts, and which label task was asked.
        extra = {k: item[k] for k in ("split", "context_matches", "label_task", "usecase")
                 if k in item}
        rec = {
            "id": item["id"], "eval_type": item["eval_type"],
            "category": item.get("category"), "prompt_lang": asked, **extra,
            "score": {k: round(statistics.fmean(s[k] for s in scores), 4) for k in keys},
            # An empty reply grades as wrong but is not a wrong answer — it is
            # a missing one, and the two need telling apart when a run goes bad.
            "no_answer": round(statistics.fmean(float(not answered(r)) for r in replies), 4),
            "sample_reply": replies[0][:400],
        }
        per_item.append(rec)
        if any(answered(r) for r in replies):
            empties = 0
            if ckpt:
                ckpt.add(rec)
        else:
            empties += 1
            if max_empty and empties >= max_empty:
                raise RuntimeError(
                    f"{empties} consecutive blank replies at item {i}/{len(rows)}; "
                    f"aborting so a partial track is not scored")
        if i % 25 == 0:
            LOG.info("%s: %d/%d", rows[0].get("track", "qa"), i, len(rows))

    return {"items": len(per_item), "by_type": _aggregate(per_item), "detail": per_item}


def _mean_scores(group: List[dict]) -> dict:
    metrics = sorted({k for p in group for k in p["score"]})
    out: Dict[str, Any] = {"n": len(group)}
    for m in metrics:
        out[m] = round(statistics.fmean(p["score"].get(m, 0.0) for p in group), 4)
    out["no_answer"] = round(statistics.fmean(p["no_answer"] for p in group), 4)
    return out


def _aggregate(per_item: List[dict]) -> dict:
    out: Dict[str, dict] = {}
    for kind in sorted({p["eval_type"] for p in per_item}):
        group = [p for p in per_item if p["eval_type"] == kind]
        out[kind] = _mean_scores(group)
        # The splits that decide what a number means. A faithfulness accuracy
        # without the swapped/control split hides an always-abstain model, and a
        # use-case score without the contamination split is not reportable.
        for field, prefix in (("split", "split"), ("context_matches", "context"),
                              ("label_task", "task"), ("prompt_lang", "lang")):
            vals = sorted({str(p[field]) for p in group if field in p})
            if len(vals) > 1:
                out[kind][f"by_{prefix}"] = {
                    v: _mean_scores([p for p in group if str(p.get(field)) == v])
                    for v in vals}
    return out


def headline(result: dict) -> Dict[str, float]:
    """One number per track, for the comparison table."""
    h: Dict[str, float] = {}
    t1 = result["tracks"].get("1")
    if t1 and t1.get("perplexity"):
        h["track1_perplexity"] = t1["perplexity"]
    named = {"2": "track2_sft", "3": "track3_vlm"}
    for track, t in result["tracks"].items():
        if track == "1" or not t or not t.get("by_type"):
            continue
        vals = []
        for kind, m in t["by_type"].items():
            vals.append(m.get("correct", m.get("f1", m.get("key_f1", 0.0))))
        h[f"{named.get(track, track)}_score"] = round(statistics.fmean(vals), 4)
    return h


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score a model against the benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("-m", "--model", required=True, help="Ollama model tag to evaluate")
    ap.add_argument("-t", "--tag", help="name for this run (default: the model tag)")
    ap.add_argument("--tracks", default="1,2,3", metavar="LIST",
                    help=f"tracks to run. {TRACKS_HELP}; 'probe' scores the "
                         "training-side probe set, 'uc' every use-case track")
    ap.add_argument("--lang", choices=["ko", "en", "mix"], default="ko",
                    help="prompt language; 'mix' draws per item from eval.lang_mix "
                         "(deterministic by item id), the answer key is the same either way")
    ap.add_argument("--repeats", type=int, help="samples per item (default 3)")
    ap.add_argument("--limit", type=int, help="first N items per track, for a smoke test")
    ap.add_argument("--closed-book", action="store_true",
                    help="withhold the passage, so the item tests domain knowledge rather "
                         "than reading comprehension")
    ap.add_argument("--ollama-url", help="override the Ollama endpoint")
    ap.add_argument("--runs-dir", help="where run files go (default <out-dir>/runs)")
    args = ap.parse_args()

    cfg = resolve_config(args)
    if args.ollama_url:
        cfg["eval"]["ollama_base_url"] = args.ollama_url
    if args.repeats:
        cfg["eval"]["repeats"] = args.repeats
    describe(cfg)

    tag = args.tag or args.model.replace(":", "-").replace("/", "-")
    runs_dir = Path(args.runs_dir) if args.runs_dir else cfg["out_dir"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    files = {"1": "track1_dapt.jsonl", "2": "track2_sft.jsonl", "3": "track3_vlm.jsonl",
             "probe": "probe_trained.jsonl"}
    # Use-case tracks come from the config registry, so evaluate.py needs no
    # change when a use case is added: --tracks uc1_safety, or "uc" for all.
    usecases = {k: v for k, v in (cfg.get("usecases") or {}).items()
                if not k.startswith("_") and isinstance(v, dict) and v.get("enabled", True)}
    for k, v in usecases.items():
        files[k] = v.get("track_file", f"{k}.jsonl")
    wanted = resolve_tracks(args.tracks)
    if "uc" in wanted:
        wanted = [t for t in wanted if t != "uc"] + list(usecases)
    result = {"tag": tag, "model": args.model, "lang": args.lang,
              "repeats": cfg["eval"]["repeats"], "limit": args.limit,
              "book": "closed" if args.closed_book else "open",
              "benchmark_dir": str(cfg["out_dir"]), "tracks": {}}

    started = time.time()
    for t in wanted:
        path = cfg["out_dir"] / files[t]
        if not path.exists():
            LOG.warning("track %s not built (%s) - skipping", track_label(t), path.name)
            continue
        rows = read_jsonl(path)
        LOG.info("track %s: %d item(s)", track_label(t), len(rows))
        if t == "1":
            result["tracks"]["1"] = run_track1(cfg, args.model, rows, args.limit)
        else:
            has_images = any(r.get("image") or r.get("images") for r in rows)
            ckpt = Checkpoint(runs_dir / ".ckpt" / f"{tag}-{t}.jsonl",
                              {"model": args.model, "lang": args.lang,
                               "repeats": cfg["eval"]["repeats"], "limit": args.limit,
                               "closed_book": bool(args.closed_book),
                               "items": len(rows)})
            resumed = ckpt.load()
            if resumed:
                LOG.info("track %s: resuming, %d of %d item(s) already scored",
                         track_label(t), resumed, len(rows))
            ckpt.open()
            try:
                result["tracks"][t] = run_qa(
                    cfg, args.model, rows, args.lang, cfg["eval"]["repeats"], args.limit,
                    cfg["generated_dir"] if has_images else None, args.closed_book, ckpt)
            finally:
                ckpt.close()
            # kept until the track scores, so an abort leaves something to resume from
            ckpt.path.unlink(missing_ok=True)

    result["elapsed_sec"] = round(time.time() - started, 1)
    result["headline"] = headline(result)
    out = write_json(runs_dir / f"{tag}.json", result)
    LOG.info("wrote %s", out)
    for k, v in result["headline"].items():
        LOG.info("  %-26s %s", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
