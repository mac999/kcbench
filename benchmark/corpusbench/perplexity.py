#!/usr/bin/env python3
"""
Track 1 — perplexity over held-out text, computed locally.

    python perplexity.py --model Qwen/Qwen3-8B --tag base
    python perplexity.py --model ./out/qwen3-8b-dapt --tag dapt-v1
    python perplexity.py --model Qwen/Qwen3-0.6B --tag smoke --limit 20
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import List

from corpusbench.common import (BENCHMARK_NAME, BENCHMARK_VERSION, add_common_args, describe,
                    log, read_jsonl, resolve_config, utc_now, write_json)

LOG = log("ppl")


def load_model(name: str, dtype: str, device: str):
    import torch                                    # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: PLC0415

    torch_dtype = {"auto": "auto", "float16": torch.float16,
                   "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    LOG.info("loading %s (%s, %s)", name, dtype, device)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch_dtype, device_map=device, trust_remote_code=True)
    model.eval()
    return tok, model


def chunk_nll(tok, model, text: str, max_len: int) -> tuple[float, int] | None:
    """
    Summed negative log-likelihood over one chunk, and its token count.

    Returned as a sum rather than a mean so the corpus figure can be weighted by
    length. Averaging per-chunk perplexities instead lets a two-token fragment
    count as much as a full clause.
    """
    import torch                                    # noqa: PLC0415

    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids
    if ids.shape[1] < 2:                            # nothing to predict
        return None
    ids = ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits
    # Shift: position i predicts token i+1.
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    target = ids[:, 1:]
    picked = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(-picked.sum().item()), int(target.numel())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Perplexity on held-out text",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("-m", "--model", required=True,
                    help="HF model id or a local checkpoint directory")
    ap.add_argument("-t", "--tag", help="name for this run (default: the model id)")
    ap.add_argument("--limit", type=int, help="first N chunks, for a smoke test")
    ap.add_argument("--max-length", type=int,
                    help="tokens per chunk before truncation (default 2048)")
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device", help="device_map for transformers")
    ap.add_argument("--runs-dir", help="where run files go (default <out-dir>/runs)")
    args = ap.parse_args()

    cfg = resolve_config(args)
    pp = cfg["perplexity"]
    max_length = args.max_length or pp["max_length"]
    dtype = args.dtype or pp["dtype"]
    device = args.device or pp["device"]
    describe(cfg)

    path = cfg["out_dir"] / "track1_dapt.jsonl"
    if not path.exists():
        LOG.error("track 1 not built - run build_tracks.py first (%s)", path)
        return 1
    rows: List[dict] = read_jsonl(path)
    if args.limit:
        rows = rows[:args.limit]
    LOG.info("track 1: %d chunk(s)", len(rows))

    tok, model = load_model(args.model, dtype, device)

    total_nll, total_tokens, skipped = 0.0, 0, 0
    per_chunk: List[float] = []
    by_category: dict[str, list] = {}
    started = time.time()

    for i, row in enumerate(rows, 1):
        got = chunk_nll(tok, model, row["text"], max_length)
        if not got:
            skipped += 1
            continue
        nll, n = got
        total_nll += nll
        total_tokens += n
        ppl = math.exp(nll / n)
        per_chunk.append(ppl)
        by_category.setdefault(row.get("category", "?"), []).append(ppl)
        if i % pp["log_every"] == 0:
            LOG.info("  %d/%d - running perplexity %.3f", i, len(rows),
                     math.exp(total_nll / total_tokens))

    if not total_tokens:
        LOG.error("nothing scored")
        return 1

    # Token-weighted is the figure to compare across checkpoints; the median of
    corpus_ppl = math.exp(total_nll / total_tokens)
    result = {
        "benchmark": BENCHMARK_NAME, "version": BENCHMARK_VERSION,
        "tag": args.tag or args.model.replace("/", "-"),
        "model": args.model, "scored_at": utc_now(),
        "track": "1", "chunks": len(per_chunk), "skipped": skipped,
        "tokens": total_tokens, "max_length": max_length, "dtype": dtype,
        "tracks": {"1": {
            "items": len(rows), "scored": len(per_chunk),
            "perplexity": round(corpus_ppl, 4),
            "perplexity_median": round(statistics.median(per_chunk), 4),
            "perplexity_mean_per_chunk": round(statistics.fmean(per_chunk), 4),
            "by_category": {k: round(statistics.fmean(v), 4)
                            for k, v in sorted(by_category.items())},
        }},
        "elapsed_sec": round(time.time() - started, 1),
    }
    result["headline"] = {"track1_perplexity": result["tracks"]["1"]["perplexity"]}

    runs_dir = Path(args.runs_dir) if args.runs_dir else cfg["out_dir"] / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out = write_json(runs_dir / f"{result['tag']}.json", result)
    LOG.info("wrote %s", out)
    LOG.info("  perplexity (token-weighted) %.4f over %d token(s)", corpus_ppl, total_tokens)
    LOG.info("  perplexity (median chunk)   %.4f", result["tracks"]["1"]["perplexity_median"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
