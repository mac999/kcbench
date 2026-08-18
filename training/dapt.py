#!/usr/bin/env python3
"""
Domain-adaptive pre-training on the held-out-free corpus.

    python dapt.py --smoke
    python dapt.py --model Qwen/Qwen3-8B --out out/qwen3-8b-dapt
    python dapt.py --epochs 2 --rank 128 --batch 4

Reads benchmark/data/train/train_dapt.jsonl, which is the corpus with every
benchmark document removed. Training on ai_ready_full directly would put the
evaluation text in front of the model and make every later score meaningless.

LoRA rather than a full update: an 8B full fine-tune needs the optimiser state
for every weight and gives catastrophic forgetting a much larger surface, and
the benchmark measures both what was learned and what was lost. Rank is the knob
that matters here, because injecting facts asks more of the adapter than
matching a style does.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT / "benchmark/data/train/train_dapt.jsonl"


class Chunks(Dataset):
    def __init__(self, path: Path, tok, max_len: int, limit: int | None = None):
        self.rows = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                text = json.loads(line).get("text")
                if text:
                    self.rows.append(text)
                if limit and len(self.rows) >= limit:
                    break
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        enc = self.tok(self.rows[i], truncation=True, max_length=self.max_len)
        return {"input_ids": enc["input_ids"]}


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    ids, mask, labels = [], [], []
    for b in batch:
        x = b["input_ids"]
        pad = width - len(x)
        ids.append(x + [pad_id] * pad)
        mask.append([1] * len(x) + [0] * pad)
        # -100 on padding so it contributes no loss
        labels.append(x + [-100] * pad)
    return {"input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
            "labels": torch.tensor(labels)}


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA DAPT on the construction corpus")
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-8B")
    ap.add_argument("-d", "--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("-o", "--out", type=Path, default=PROJECT / "training/out/qwen3-8b-dapt")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--checkpointing", action="store_true",
                    help="trade compute for memory; unnecessary on a box with "
                         "headroom and it costs about a third of the throughput")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--limit", type=int, help="use only the first N chunks")
    ap.add_argument("--smoke", action="store_true", help="200 chunks, one short run")
    ap.add_argument("--resume-from", type=Path, metavar="DIR",
                    help="adapter checkpoint (out/.../step-N) to continue from. "
                         "Optimiser moments are not saved and the reshuffled epoch "
                         "does not replay bit-exactly, so a resume is approximate: "
                         "the schedule continues where it was, the first steps act "
                         "as a short re-warmup, and a few chunks are seen twice or "
                         "skipped. Good enough for DAPT; not for a paper.")
    args = ap.parse_args()

    if args.smoke:
        args.limit, args.epochs, args.save_every = 200, 1.0, 10_000

    torch.manual_seed(20260814)
    print(f"model  {args.model}")
    print(f"data   {args.data}")
    print(f"out    {args.out}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.config.use_cache = False
    if args.checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    resumed = 0
    if args.resume_from:
        m = re.search(r"step-(\d+)", str(args.resume_from))
        if not m:
            raise SystemExit(f"cannot read a step number from {args.resume_from}")
        resumed = int(m.group(1))
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        print(f"resume from {args.resume_from} at optimiser step {resumed}")
    else:
        lora = LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
            bias="none", task_type="CAUSAL_LM",
            # Every linear in the block, not just the attention projections: the
            # feed-forward is where a transformer keeps most of what it knows, and
            # this run is trying to add knowledge rather than restyle output.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"lora   r={args.rank} alpha={args.alpha}  trainable {trainable/1e6:.1f}M "
          f"of {total/1e9:.2f}B ({trainable/total*100:.2f}%)")

    data = Chunks(args.data, tok, args.max_len, args.limit)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps_per_epoch = max(1, len(loader) // args.accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(f"data   {len(data):,} chunks, {len(loader):,} batches, "
          f"{total_steps:,} optimiser step(s)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * args.warmup), total_steps)
    for _ in range(resumed):
        sched.step()

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    model.train()
    step, seen_tokens, running, started = resumed, 0, 0.0, time.time()
    stop = False
    # Skip the share of the epoch the interrupted run already consumed. The
    # shuffle does not replay bit-exactly (the RNG has advanced differently),
    # so this resumes the epoch's position, not its precise batch sequence.
    to_skip = resumed * args.accum

    for epoch in range(math.ceil(args.epochs)):
        if stop:
            break
        for i, batch in enumerate(loader):
            if i < to_skip:
                continue
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / args.accum).backward()
            running += out.loss.item()
            seen_tokens += int(batch["attention_mask"].sum().item())

            if (i + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                loss = running / (args.log_every * args.accum)
                elapsed = time.time() - started
                rec = {"step": step, "total_steps": total_steps, "epoch": epoch,
                       "loss": round(loss, 4), "ppl": round(math.exp(min(loss, 20)), 3),
                       "lr": round(sched.get_last_lr()[0], 8),
                       "tokens": seen_tokens, "tok_per_s": round(seen_tokens / elapsed, 1),
                       "elapsed_min": round(elapsed / 60, 1)}
                print(json.dumps(rec), flush=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                running = 0.0

            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(args.out / f"step-{step}")
                print(f"saved {args.out / f'step-{step}'}", flush=True)

            if step >= total_steps:
                stop = True
                break
        to_skip = 0  # only the first epoch was partially consumed

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "run.json").write_text(json.dumps({
        "model": args.model, "data": str(args.data),
        "epochs": args.epochs, "batch": args.batch, "accum": args.accum,
        "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
        "max_len": args.max_len, "steps": step, "tokens": seen_tokens,
        "elapsed_min": round((time.time() - started) / 60, 1),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done, {step} step(s), {seen_tokens:,} token(s), "
          f"{(time.time() - started) / 60:.1f} min -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
