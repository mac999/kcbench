# training

Fine-tuning pipeline for the Korean construction corpus. Kept separate from
`../benchmark/` on purpose: the benchmark is the measuring instrument, and an
instrument that shares code with the thing it measures stops being one.

```
training/
  dapt.py              domain-adaptive pre-training (stage 1)
  sft.py               supervised fine-tuning (stage 2)
  merge.py             fold the adapter into the base weights
  out/                 checkpoints, adapters, logs
  README.md
```

## What this trains, and in what order

Two stages, because they teach different things and the benchmark measures them
separately.

| Stage | Script | Data | Teaches | Benchmark signal |
|---|---|---|---|---|
| 1. DAPT | `dapt.py` | `train_dapt.jsonl`, 26,767 raw chunks | the wording and structure of Korean construction regulation | track 1 perplexity should fall |
| 2. SFT | `sft.py` | `train_sft.jsonl`, 15,666 instruction pairs | answering questions in the expected shape | probe and track 2 should rise |

Run DAPT first and check track 1 before spending time on SFT. Perplexity moves
long before answer accuracy does, so a DAPT run that leaves it unchanged has not
trained, and no amount of SFT afterwards will fix that. Diagnosing it at stage 1
costs one measurement; diagnosing it after stage 2 costs both runs.

## Input

Both scripts read from `../benchmark/data/train/`, which
`benchmark/make_train_split.py` writes. **Never train from `ai_ready_full/`
directly.** That directory contains generated output for the benchmark documents
too, and training on it puts the evaluation text in front of the model, after
which every score this repository produces is measuring recall of the training
set.

The split is not a simple document exclusion. The corpus collects the same
regulation more than once — under names differing by a suffix, as
amendment-and-original pairs, as sibling standards sharing whole clauses — so the
split also drops any row whose text hashes to a reserved chunk. That caught 816
rows a name-based exclusion left behind.

Check before every run:

```bash
cd ../benchmark && python make_train_split.py --check
```

### Record shapes

`train_dapt.jsonl` — raw text for language modelling. Only `text` is read; the
rest travels for provenance.

```json
{"id": "dapt_000022",
 "text": "상수도설계기준\n제1조(목적) 이 고시는 ...",
 "doc_id": "137. 개정전문 (상수도설계기준 일부개정고시안 20250911)",
 "source_org": "기후에너지환경부장관", "source_date": "2025",
 "section_path": "제3조", "source_file": "01_design_standards_kds/.../dapt_training_data.jsonl"}
```

Bodies average 665 characters, roughly 460 tokens, and are capped at 901. One
epoch is about 12.3M tokens.

`train_sft.jsonl` — instruction pairs. `input` and `output` are nested objects,
not strings, so `sft.py` flattens them into a chat template.

```json
{"id": "sft_000064", "task_type": "regulation_qa",
 "instruction": "상수도설계기준의 시행일자는 언제인가요?",
 "input": {"context": "이 고시는 발령한 날부터 시행한다.", "metadata": {...}},
 "output": {"answer": "2025년 10월 1일", "evidence": [{"doc_id": "...", "section": "..."}]}}
```

## Method

**LoRA, not a full update.** A full 8B fine-tune needs optimiser state for every
weight and gives catastrophic forgetting a much larger surface. The benchmark
measures what was lost as well as what was gained — open-book accuracy is the
regression check — and an adapter keeps that risk small and reversible.

**All linear layers, not just attention.** `target_modules` covers `q,k,v,o` and
`gate,up,down`. A transformer keeps most of what it knows in the feed-forward
block, and this corpus is being taught facts rather than a style, so restricting
the adapter to attention projections would be adapting the wrong part.

**Rank is the knob that matters.** Injecting knowledge asks more of an adapter
than matching a tone does. The default is r=64, alpha=128. If probe accuracy does
not move after a clean DAPT run, raise the rank before changing anything else.

**bf16 with gradient checkpointing.** The 8B weights are 16 GB, activations are
traded for compute, and the optimiser only carries the adapter. No quantised
base: `bitsandbytes` has no ARM64 wheel for this machine, and a bf16 base avoids
the accuracy question entirely on 130 GB of unified memory.

## Settings

| Flag | Default | What it changes |
|---|---|---|
| `-m, --model` | `Qwen/Qwen3-8B` | HF id or a local checkpoint directory |
| `-d, --data` | `../benchmark/data/train/train_dapt.jsonl` | input file |
| `-o, --out` | `out/qwen3-8b-dapt` | where adapters and logs are written |
| `--epochs` | `1.0` | fractional values are allowed and stop mid-epoch |
| `--batch` | `4` | sequences per forward pass |
| `--accum` | `8` | gradient accumulation, so the effective batch is 32 |
| `--lr` | `1e-4` | AdamW, cosine schedule, 3% warmup |
| `--rank` / `--alpha` | `64` / `128` | LoRA capacity; raise both together |
| `--dropout` | `0.05` | LoRA dropout |
| `--max-len` | `1024` | tokens per sequence; chunks rarely exceed 700 |
| `--warmup` | `0.03` | share of steps spent warming up |
| `--log-every` | `20` | optimiser steps between log lines |
| `--save-every` | `500` | intermediate checkpoints, 0 to disable |
| `--limit` | — | use only the first N rows |
| `--smoke` | — | 200 rows, for checking the pipeline runs at all |

Raise `--batch` before `--accum` while memory allows; they multiply to the same
effective batch, but a larger forward pass is faster per token.

## Output

```
out/qwen3-8b-dapt/
  adapter_model.safetensors    the LoRA weights
  adapter_config.json          rank, alpha, target modules
  tokenizer files
  train_log.jsonl              one record per logged step
  run.json                     the settings this run actually used
  step-500/, step-1000/, ...   intermediate checkpoints
```

`train_log.jsonl` carries loss, its exponential, learning rate, tokens seen and
throughput:

```json
{"step": 20, "total_steps": 836, "epoch": 0, "loss": 1.8234, "ppl": 6.193,
 "lr": 9.87e-05, "tokens": 294912, "tok_per_s": 1843.2, "elapsed_min": 2.7}
```

Training loss and benchmark perplexity are not the same number — the loss here is
over training text the model is actively fitting, and track 1 is over held-out
text it has never seen. Watch the loss to confirm the run is healthy; use track 1
to decide whether it worked.

## Evaluating what came out

The adapter has to be merged before Ollama can serve it, but perplexity reads HF
checkpoints directly, so track 1 needs no merge:

```bash
cd ../benchmark
python perplexity.py -m ../training/out/qwen3-8b-dapt --tag dapt-v1
python compare.py --base v3-ppl-qwen3-8b --after dapt-v1
```

For the answer tracks, merge and serve:

```bash
cd ../training
python merge.py --adapter out/qwen3-8b-dapt --out out/qwen3-8b-dapt-merged
# convert to GGUF and register with Ollama, then
cd ../benchmark
python evaluate.py -m qwen3-dapt:v1 --tag dapt-probe --tracks probe --closed-book
python evaluate.py -m qwen3-dapt:v1 --tag dapt-t2 --tracks 2 --closed-book
python evaluate.py -m qwen3-dapt:v1 --tag dapt-open --tracks 2
```

## Baselines to beat

`qwen3:8b` before any training, from `benchmark/CHANGELOG.md`:

| Metric | Base | Minimum | Meaningful |
|---|---:|---:|---:|
| track 1 perplexity | 7.589 | 7.0 | 6.0 |
| probe numeric, closed book | 0.169 | 0.30 | 0.50 |
| track 2 numeric, closed book | 0.166 | 0.22 | 0.30 |
| track 2 numeric, open book | 0.953 | hold above 0.90 | |

The two closed-book targets differ because the sets differ. 83% of probe answers
appear in the training data; only 25% of track 2 answers do, because track 2 is
mined from documents the split withheld. Probe asks whether the model absorbed
what it was taught; track 2 asks whether it generalises to regulation it never
saw. Expecting track 2 to reach probe's numbers is expecting the model to recall
text that was never in front of it.

Open book is the regression check, not a result. It sits near its ceiling on any
competent model — 3B scores 0.849 and 30B scores 0.932 on the same items — so it
cannot show improvement, but it will show damage.

## Reading the outcome

| track 1 | probe | track 2 | Reading |
|---|---|---|---|
| falls | rises | rises | worked |
| falls | rises | flat | absorbed the corpus, does not generalise |
| falls | flat | flat | adapting style, not acquiring facts: raise the rank |
| flat | flat | flat | the run did not train: check lr, data, adapter targets |
| any | any | any, open book falls | forgetting: lower lr or epochs |
