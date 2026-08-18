#!/usr/bin/env python3
"""
Fold a LoRA adapter into the base weights.

    python merge.py --adapter out/qwen3-8b-dapt --out out/qwen3-8b-dapt-merged

Track 1 reads adapters directly, so this is only needed for the answer tracks,
which go through Ollama and want a single set of weights. The merged directory
is a normal HF checkpoint and is what a GGUF conversion takes as input.

Writes roughly 16 GB in bf16. The adapter it came from stays where it is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    ap.add_argument("-a", "--adapter", type=Path, required=True)
    ap.add_argument("-m", "--model", help="base model; defaults to the adapter's record of it")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = ap.parse_args()

    base = args.model
    if not base:
        cfg = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        base = cfg.get("base_model_name_or_path")
        if not base:
            raise SystemExit("adapter_config.json names no base model, pass --model")
    print(f"base    {base}")
    print(f"adapter {args.adapter}")

    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, device_map="cpu", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok_src = args.adapter if (args.adapter / "tokenizer_config.json").exists() else base
    AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True).save_pretrained(args.out)
    print(f"merged -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
