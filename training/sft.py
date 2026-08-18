#!/usr/bin/env python3
"""
Supervised fine-tuning on the instruction pairs.

    python sft.py --smoke
    python sft.py --base out/qwen3-8b-dapt --out out/qwen3-8b-sft
    python sft.py --model Qwen/Qwen3-8B --out out/qwen3-8b-sft-only

Reads benchmark/data/train/train_sft.jsonl. `input` and `output` are nested
objects there, not strings, so both are flattened into the chat template before
tokenising.

Loss is computed on the answer only. Training on the prompt as well teaches the
model to generate questions, which is not the task and dilutes the gradient that
is.

--base continues from a DAPT adapter, which is the intended order: DAPT teaches
the wording, SFT teaches the answer shape. Passing --model instead trains SFT on
the bare base, which is worth running once as a control if DAPT's contribution
is in question.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT / "benchmark/data/train/train_sft.jsonl"


def as_text(value) -> str:
    """Flatten the nested input/output objects into something a prompt can hold."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("answer", "context", "text"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key]
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class Pairs(Dataset):
    def __init__(self, path: Path, tok, max_len: int, limit: int | None = None):
        self.samples = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                instruction = (r.get("instruction") or "").strip()
                answer = as_text(r.get("output")).strip()
                if not instruction or not answer:
                    continue
                context = as_text(r.get("input")).strip()
                user = f"{instruction}\n\n[조문]\n{context}" if context else instruction
                self.samples.append((user, answer))
                if limit and len(self.samples) >= limit:
                    break
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        user, answer = self.samples[i]
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = self.tok(prompt, add_special_tokens=False).input_ids
        a_ids = self.tok(answer + self.tok.eos_token, add_special_tokens=False).input_ids
        ids = (p_ids + a_ids)[:self.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:self.max_len]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    ids, mask, labels = [], [], []
    for b in batch:
        x, y = b["input_ids"], b["labels"]
        pad = width - len(x)
        ids.append(x + [pad_id] * pad)
        mask.append([1] * len(x) + [0] * pad)
        labels.append(y + [-100] * pad)
    return {"input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
            "labels": torch.tensor(labels)}


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA SFT on the construction instruction pairs")
    ap.add_argument("-m", "--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--base", type=Path, help="DAPT adapter to continue from")
    ap.add_argument("-d", "--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("-o", "--out", type=Path, default=PROJECT / "training/out/qwen3-8b-sft")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.limit, args.epochs, args.save_every = 200, 1.0, 0

    torch.manual_seed(20260814)
    print(f"model  {args.model}")
    print(f"base   {args.base or 'none, training the bare base'}")
    print(f"data   {args.data}")
    print(f"out    {args.out}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if args.base:
        model = PeftModel.from_pretrained(model, args.base, is_trainable=True)
        print("continuing the DAPT adapter")
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {trainable/1e6:.1f}M")

    data = Pairs(args.data, tok, args.max_len, args.limit)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps_per_epoch = max(1, len(loader) // args.accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(f"data   {len(data):,} pair(s), {total_steps:,} optimiser step(s)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * args.warmup), total_steps)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    model.train()
    step, seen, running, started = 0, 0, 0.0, time.time()
    stop = False

    for epoch in range(math.ceil(args.epochs)):
        if stop:
            break
        for i, batch in enumerate(loader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / args.accum).backward()
            running += out.loss.item()
            seen += int(batch["attention_mask"].sum().item())

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
                       "lr": round(sched.get_last_lr()[0], 8), "tokens": seen,
                       "tok_per_s": round(seen / elapsed, 1),
                       "elapsed_min": round(elapsed / 60, 1)}
                print(json.dumps(rec), flush=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                running = 0.0

            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(args.out / f"step-{step}")

            if step >= total_steps:
                stop = True
                break

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "run.json").write_text(json.dumps({
        "model": args.model, "base_adapter": str(args.base) if args.base else None,
        "data": str(args.data), "epochs": args.epochs, "batch": args.batch,
        "accum": args.accum, "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
        "max_len": args.max_len, "steps": step, "tokens": seen,
        "elapsed_min": round((time.time() - started) / 60, 1),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done, {step} step(s), {seen:,} token(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
