# Reproduction snapshot, November 2026 manuscript

This folder fixes the state from which the manuscript submitted in November
2026 was built: the commit, the settings files, the data releases, the answer
keys, the scoring runs and the trained models behind every table and figure.
The benchmark keeps changing after the submission; `check.sh` tells whether
the tree still matches, and `run.sh` reruns the experiments from the frozen
settings copied here.

```
check.sh       commits, settings files and hashed inputs against MANIFEST.txt
run.sh         data | keys | score | dapt | train
config/        frozen copies of the benchmark, generator and training settings
env/           interpreter and package versions, Ollama model digests
MANIFEST.txt   commits, tool versions, sha256 of 126 inputs and outputs
```

## State recorded

| | |
|---|---|
| kcbench commit | `117cfcb` (2026-11-16) |
| gen_aec_syn_data commit | `64580bf` (2026-10-20) |
| generator | qwen3:30b-a3b through Ollama, temperature 0.5, JSON mode, 8,192 context (`config/generator/`) |
| data releases | v0.5.2 (`train_data_v052/`, 2026-09-27), v0.5.3 (`train_data_v053/`, 2026-10-18), 29 CFR 1926 (`train_data_us/`, 2026-10-06) |
| answer keys | `ground_truth_v052/` (2026-09-27), `ground_truth_v053/` (2026-10-20, holdout seed 20260814), `ground_truth_us/` (2026-10-06) |
| scoring | greedy, one sample, thinking off; judge panel exaone3.5:7.8b, glm4:9b, gemma2:9b (`config_kr_ft.json`, `config_kr_v053_ft.json`) |
| training | Qwen3-8B, LoRA r 64 alpha 128; SFT lr 5e-5, batch 2 x accum 16, max-len 3072, 2 epochs; DAPT lr 1e-4, batch 8 x accum 4, 1 epoch (`config/training/exp_v053cmp.json`) |
| hardware | NVIDIA GB10, driver 580.159.03; torch 2.13.0+cu130, transformers 5.15.1, peft 0.20.0; Ollama 0.32.1; llama.cpp `01818e4` |

## Where each result comes from

| Manuscript | Source files | Stage |
|---|---|---|
| Table 1 (nine text tracks, 7,002 items) | `ground_truth_v053/*.jsonl` | keys |
| Fig. 3 (open vs closed book, base model) | `ground_truth_v053/runs/v053cmp-base-{open,closed}.json` | score |
| Section 4.1 (7,542 provenance records) | `ground_truth_v053/provenance.jsonl`, `contamination_report.json` | keys |
| Table 2, Fig. 5 (sign reversal, 810 items) | `ground_truth_v052/runs/{v052-open,uc6v2-open-qwen8b,ft-dapt-open}.json`, `compare_*_vs_ft-dapt-open.json` | score, dapt |
| Table 3, Fig. 6 (235-item review) | `ground_truth_v053/final_review/review_queue_sample235.expert_final.jsonl` | - |
| Table 4 (arms, rows, steps) | `training/out/v053cmp-*/run.json` | train |
| Fig. 7, Sections 5.2-5.3 | `ground_truth_v053/runs/v053cmp-*-open.json`, `compare_v053cmp-*.json` | train |
| Section 5.4, Fig. 8 (abstention, seeds) | `v053cmp-{v052,n2202}{,-s2,-s3}-open.json`, `v053cmp-*-closed.json`, `ground_truth_v053/train*/train_star*.jsonl` | train |
| Section 6 (English build) | `ground_truth_us/runs/us-qwen38b-{open,closed}.json`, `ground_truth_v052/runs/v052-{open,closed}.json` | score |

Figures 1 and 2 are diagrams, Fig. 4 is a screenshot of the review tab
(`cb.py webview -c config_us.json`). Corpus and generation counts (1,947
documents, 104,088 generated rows, 12,725 verified SFT rows, acceptance
38.4% to 91.5%) come from the generator logs and
`ground_truth_v053/train/manifest.json`; the English corpus figures from
`ground_truth_us/export/manifest.json`.

## Running it

```
./check.sh                 seconds; exit 1 when the tree has drifted
./run.sh data              download the released v0.5.2 and v0.5.3 data
./run.sh keys [--all]      rebuild the v0.5.3 key, splits and subsets (--all: v0.5.2 and English keys too)
./run.sh score             qwen3:8b on the three keys, hours on a GB10
./run.sh dapt              DAPT model (reuses qwen3-dapt:v1f when registered) and its scores
./run.sh train [--plan]    the ten training arms, about three GPU-days
```

Scoring needs Ollama with qwen3:8b and the three judges. Training uses the
interpreter in `KCBENCH_TRAIN_PY` (default `/home/tom/venvs/cu130/bin/python`,
packages in `env/pip_freeze_training_cu130.txt`) and llama.cpp at
`LLAMA_CPP` for the GGUF conversion. Every scoring tag and training step is
skipped when its output exists; delete the run file to redo one.

`run.sh data --regenerate` and `--us` rerun the generator. They produce
different rows, since the generator samples at temperature 0.5; the released
data is what every later stage reads.

## What reproduces exactly and what does not

- Answer keys, splits and derived subsets rebuild bit for bit from the released
  data for the fixed seed (checked 2026-11-22 against the hashes).
- Scoring a registered model again reproduces its run file up to Ollama's
  numerical determinism; greedy decoding returned identical replies across
  repeats on this machine, but another Ollama or llama.cpp build can move
  individual items.
- Retraining an arm gives a model of the same recipe, not the same weights.
  Section 5.4 shows that the equal-size arm depends on the shuffle seed, so a
  retrained set should be read against the spread in Fig. 8, not against single
  values. The adapter of the original DAPT run is no longer on disk; the Ollama
  model is the surviving artefact and `env/ollama_model_blobs.txt` ties the
  published scores to its digest.
- The LLM generation and the expert review are not re-runnable; their outputs
  are inputs here and are hashed in `MANIFEST.txt`.

## Local files

Several inputs are gitignored and exist only on the experiment machine or on
request: the per-item run files under `ground_truth_*/runs/`, the training
splits and derived subsets, `training/out/`, the review queue and its finished
version, the generated data releases, and the manuscript sources. `check.sh`
reports them as missing rather than failing.
