#!/usr/bin/env python3
"""
Supervised fine-tuning on the instruction pairs.

    python sft.py --smoke
    python sft.py --base out/qwen3-8b-dapt --out out/qwen3-8b-sft
    python sft.py --model Qwen/Qwen3-8B --out out/qwen3-8b-sft-only

Reads ground_truth/train/train_sft.jsonl. `input` and `output` are nested
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
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT / "ground_truth/train/train_sft.jsonl"


def as_text(value, json_fallback: bool = True) -> str:
    """Flatten the nested input/output objects into something a prompt can hold."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("answer", "context", "text"):
            if isinstance(value.get(key), str) and value[key].strip():
                return value[key]
        # An input whose context is "" used to come back as the JSON of its
        # metadata, and 45% of the v3 pairs trained on a [조문] block holding
        # {"metadata": {...}}. Without the fallback an empty context stays empty.
        return json.dumps(value, ensure_ascii=False) if json_fallback else ""
    return str(value)


def as_target(output, mode: str) -> str:
    """The completion the model learns: the answer text, or the full output object."""
    if mode == "json" and isinstance(output, dict):
        return json.dumps({"answer": output.get("answer", ""),
                           "evidence": output.get("evidence", []),
                           "final_label": output.get("final_label", "")},
                          ensure_ascii=False)
    return as_text(output)


# How the user turn is laid out. "legacy" is what every arm before v4.5 was
# trained on; "eval-ko"/"eval-en" reproduce the benchmark's own qa.open and
# qa.closed prompts (kcbench/prompts.py) so training and scoring share a format.
PROMPT_FORMATS = {
    "legacy": ("{instruction}\n\n{label}\n{context}", "{instruction}"),
    "eval-ko": ("다음 조문을 읽고 질문에 답하시오.\n\n[조문]\n{context}\n\n[질문]\n{instruction}",
                "다음 질문에 답하시오.\n\n[질문]\n{instruction}"),
    "eval-en": ("Read the passage and answer the question.\n\n[Passage]\n{context}\n\n"
                "[Question]\n{instruction}",
                "Answer the question.\n\n[Question]\n{instruction}"),
}


class Pairs(Dataset):
    def __init__(self, path: Path, tok, max_len: int, limit: int | None = None,
                 context_label: str = "[조문]", prompt_format: str = "legacy",
                 json_fallback: bool = True, target: str = "text"):
        self.samples = []
        with_ctx, without_ctx = PROMPT_FORMATS[prompt_format]
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                instruction = (r.get("instruction") or "").strip()
                answer = as_target(r.get("output"), target).strip()
                if not instruction or not answer:
                    continue
                context = as_text(r.get("input"), json_fallback).strip()
                user = (with_ctx if context else without_ctx).format(
                    instruction=instruction, label=context_label, context=context)
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
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--context-label", default="[조문]",
                    help="line that introduces the passage in the user turn; an English "
                         "build trains with [Clause]")
    ap.add_argument("--prompt-format", choices=sorted(PROMPT_FORMATS), default="legacy",
                    help="user-turn layout; eval-ko/eval-en match the benchmark prompts")
    ap.add_argument("--no-json-fallback", action="store_true",
                    help="an input with no context string trains with an empty context "
                         "instead of the JSON of its metadata")
    ap.add_argument("--target", choices=["text", "json"], default="text",
                    help="learn the answer text, or the {answer, evidence, final_label} object")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", type=Path, metavar="DIR",
                    help="adapter checkpoint (out/.../step-N) to continue from. "
                         "Optimiser moments are not saved and the reshuffled epoch "
                         "does not replay bit-exactly, so a resume is approximate: "
                         "the schedule continues where it was and a few pairs are "
                         "seen twice or skipped.")
    args = ap.parse_args()

    if args.smoke:
        args.limit, args.epochs, args.save_every = 200, 1.0, 0

    torch.manual_seed(args.seed)
    print(f"model  {args.model}")
    print(f"base   {args.base or 'none, training the bare base'}")
    print(f"data   {args.data}")
    print(f"out    {args.out}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    resumed = 0
    if args.resume_from:
        m = re.search(r"step-(\d+)", str(args.resume_from))
        if not m:
            raise SystemExit(f"cannot read a step number from {args.resume_from}")
        resumed = int(m.group(1))
        # the checkpoint already carries the DAPT adapter it was trained on top of
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        print(f"resume from {args.resume_from} at optimiser step {resumed}")
    elif args.base:
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

    data = Pairs(args.data, tok, args.max_len, args.limit, args.context_label,
                 args.prompt_format, not args.no_json_fallback, args.target)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps_per_epoch = max(1, len(loader) // args.accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(f"data   {len(data):,} pair(s), {total_steps:,} optimiser step(s)")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * args.warmup), total_steps)
    if resumed:
        # the schedule used to restart at step 0 on a resume, re-warming and
        # then cooling from the top; put it where the interrupted run was
        for _ in range(resumed):
            sched.step()
        state = args.resume_from / "optimizer.pt"
        if state.exists():
            opt.load_state_dict(torch.load(state, map_location="cuda"))
            print("optimiser state restored")

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    model.train()
    step, seen, running, started = resumed, 0, 0.0, time.time()
    stop = False
    # Skip the share of the epoch the interrupted run already consumed. The
    # shuffle does not replay bit-exactly, so this resumes the position, not
    # the precise batch sequence.
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
                torch.save(opt.state_dict(), args.out / f"step-{step}" / "optimizer.pt")
                print(f"saved {args.out / f'step-{step}'}", flush=True)

            if step >= total_steps:
                stop = True
                break
        to_skip = 0  # only the first epoch was partially consumed

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "run.json").write_text(json.dumps({
        "model": args.model, "base_adapter": str(args.base) if args.base else None,
        "data": str(args.data), "epochs": args.epochs, "batch": args.batch,
        "accum": args.accum, "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
        "max_len": args.max_len, "steps": step, "tokens": seen,
        "prompt_format": args.prompt_format, "json_fallback": not args.no_json_fallback,
        "target": args.target,
        "elapsed_min": round((time.time() - started) / 60, 1),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done, {step} step(s), {seen:,} token(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
