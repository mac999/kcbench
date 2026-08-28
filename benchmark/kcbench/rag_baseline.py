#!/usr/bin/env python3
"""
RAG baseline — score the retrieval a deployment would actually have.

    python cb.py rag -m qwen3:8b --tag base-rag --tracks sft
    python cb.py rag -m qwen3:8b --tag base-rag --top-k 5 --think off

The benchmark's two book settings are the ends of a scale, not the whole of it.
Closed book is a model answering from memory alone; open book hands it the exact
clause the item was mined from, which is what a *perfect* retriever would do.
A deployed system sits between them, and how far between is a property of the
retriever, not of the fine-tune.

This scores that middle. The held-out chunks are embedded once and searched per
item; the top-k are concatenated as the passage in place of the gold clause.
Everything downstream — prompt shape, graders, tolerances — is identical to an
open-book run, so the three numbers are directly comparable.

Retrieval quality is reported alongside accuracy: `recall_at_k` is the share of
items whose gold chunk was actually retrieved. An accuracy drop with recall near
1.0 means the model failed; a drop with low recall means the retriever did.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from kcbench.common import (BENCHMARK_NAME, BENCHMARK_VERSION, TRACKS_HELP,
                            add_common_args, describe, items_digest, log,
                            read_jsonl, resolve_config, resolve_tracks,
                            track_label, write_json)
from kcbench.evaluate import (TRACK_FILES, Checkpoint, _aggregate, answered,
                              generate, grade_label, grade_nameset,
                              grade_numeric, run_meta)

LOG = log("rag")


def embed(cfg, model: str, texts: List[str], batch: int = 32) -> List[List[float]]:
    out: List[List[float]] = []
    url = f"{cfg['eval']['ollama_base_url']}/api/embed"
    for i in range(0, len(texts), batch):
        r = requests.post(url, json={"model": model, "input": texts[i:i + batch]},
                          timeout=cfg["eval"]["request_timeout"])
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
        if (i // batch) % 20 == 0:
            LOG.info("embedded %d/%d", min(i + batch, len(texts)), len(texts))
    return out


def normalise_rows(vecs: List[List[float]]) -> List[List[float]]:
    out = []
    for v in vecs:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def top_k(qv: List[float], mat: List[List[float]], k: int) -> List[int]:
    scored = [(sum(a * b for a, b in zip(qv, row)), i) for i, row in enumerate(mat)]
    scored.sort(reverse=True)
    return [i for _, i in scored[:k]]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score with retrieved context instead of the gold clause",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-t", "--tag")
    ap.add_argument("--tracks", default="sft", metavar="LIST", help=f"{TRACKS_HELP}")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    ap.add_argument("--think", choices=["on", "off"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ollama-url")
    ap.add_argument("--runs-dir")
    args = ap.parse_args()

    cfg = resolve_config(args)
    if args.ollama_url:
        cfg["eval"]["ollama_base_url"] = args.ollama_url
    if args.think:
        cfg["eval"]["think"] = args.think == "on"
    describe(cfg)

    tag = args.tag or f"{args.model.replace(':', '-')}-rag"
    runs_dir = Path(args.runs_dir) if args.runs_dir else cfg["out_dir"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tol = cfg["eval"]["numeric_tolerance"]

    # the corpus a deployment would search: the held-out chunks
    corpus_rows = read_jsonl(cfg["out_dir"] / "track1_dapt.jsonl")
    texts = [r.get("text", "") for r in corpus_rows]
    LOG.info("corpus %d chunk(s), embedding with %s", len(texts), args.embed_model)
    corpus_keys = [((r.get("provenance") or {}).get("dataset_file"),
                    (r.get("provenance") or {}).get("row_index")) for r in corpus_rows]
    mat = normalise_rows(embed(cfg, args.embed_model, texts))

    result = {"benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
              "tag": tag, "model": args.model, "kind": "rag",
              "embed_model": args.embed_model, "top_k": args.top_k,
              "lang": args.lang, "limit": args.limit, "book": "retrieved",
              "think": cfg["eval"].get("think"), "tracks": {}}

    started = time.time()
    for t in resolve_tracks(args.tracks):
        path = cfg["out_dir"] / TRACK_FILES.get(t, f"{t}.jsonl")
        if not path.exists():
            LOG.warning("track %s not built - skipping", track_label(t))
            continue
        rows = read_jsonl(path)
        rows = rows[:args.limit] if args.limit else rows
        LOG.info("track %s: %d item(s)", track_label(t), len(rows))

        qvs = normalise_rows(embed(cfg, args.embed_model,
                                   [r.get(f"question_{args.lang}") or r.get("question_ko", "")
                                    for r in rows]))
        per_item, hits = [], 0
        for i, (item, qv) in enumerate(zip(rows, qvs), 1):
            idx = top_k(qv, mat, args.top_k)
            passage = "\n\n".join(texts[j] for j in idx)
            # The item's context is an excerpt cut from a corpus chunk, so the
            # strings do not match; provenance does. A retrieval counts as a hit
            # when it returns the chunk the item was mined from, identified by
            # the generated file and row it came from.
            pv = item.get("provenance") or {}
            gold = (pv.get("dataset_file"), pv.get("row_index"))
            if gold[0] is not None and gold in {corpus_keys[j] for j in idx}:
                hits += 1
            q = item.get(f"question_{args.lang}") or item.get("question_ko", "")
            instr = item.get(f"instruction_{args.lang}") or item.get("instruction_ko", "")
            prompt = (f"다음 조문을 읽고 질문에 답하시오.\n\n[조문]\n{passage}\n\n"
                      f"[질문]\n{q}\n\n{instr}")
            try:
                reply = generate(cfg, args.model, prompt)
            except Exception as exc:
                LOG.warning("generate failed on %s (%s)", item["id"], exc)
                reply = ""
            kind = item["eval_type"]
            if kind == "numeric":
                sc = {"correct": float(grade_numeric(reply, item, tol))}
            elif kind == "nameset":
                sc = grade_nameset(reply, item)
            else:
                sc = grade_label(reply, item)
            per_item.append({"id": item["id"], "eval_type": kind,
                             "category": item.get("category"), "prompt_lang": args.lang,
                             "score": {k: round(v, 4) for k, v in sc.items()},
                             "no_answer": round(float(not answered(reply)), 4),
                             "retrieved": idx, "sample_reply": reply[:400]})
            if i % 25 == 0:
                LOG.info("%s: %d/%d", track_label(t), i, len(rows))

        agg = _aggregate(per_item)
        result["tracks"][t] = {"items": len(per_item), "by_type": agg,
                               "recall_at_k": round(hits / max(len(rows), 1), 4),
                               "items_digest": items_digest(rows), "detail": per_item}

    result["elapsed_sec"] = round(time.time() - started, 1)
    result["meta"] = run_meta(cfg)
    result["headline"] = {f"{track_label(t)}_rag_score":
                          round(sum(m.get("correct", m.get("f1", 0.0))
                                    for m in v["by_type"].values()) / max(len(v["by_type"]), 1), 4)
                          for t, v in result["tracks"].items()}
    out = write_json(runs_dir / f"{tag}.json", result)
    LOG.info("wrote %s", out)
    for t, v in result["tracks"].items():
        LOG.info("  %-10s recall@%d %.3f", track_label(t), args.top_k, v["recall_at_k"])
        for k, m in v["by_type"].items():
            LOG.info("     %-9s %s", k, {a: b for a, b in m.items() if a in ("n", "correct", "f1")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
