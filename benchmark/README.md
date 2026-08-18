# kcbench

A held-out benchmark for measuring whether fine-tuning on a corpus actually
improved a model. The question it answers is *"did training help, and where?"* —
base checkpoint versus fine-tuned checkpoint, on identical frozen items. It is
not a ranking against other people's models. The items are mined from held-out
documents by rule and verified against the source, not written and reviewed by
engineers, so an absolute score means nothing next to a published leaderboard.
The delta does.

The machinery is domain-agnostic — it takes a directory of chunked documents and
mines factual questions out of them — but it was built for, and ships configured
for, Korean construction standards (KDS/KCS), safety regulation and IFC building
models. That is where the name comes from: **K**orean **c**onstruction. See
[Adapting it to another domain](../README.md#adapting-it-to-another-domain) for
what to change if your corpus is something else.

This file documents the build stages, the tracks, and every setting that changes
what gets built. The top-level [README](../README.md) covers what the tool is
for and how to install it.

**Language.** Evaluation prompts default to Korean. The subject is Korean
construction law, and translating clause terminology changes the question
itself. Every item also carries an English prompt (`question_en`), so
`--lang en` scores the same answer key in English. Numeric items additionally
carry `answer_en` with units in English notation.

---

## Pipeline

One entry point, `cb.py`. Every command below is also a module under
`kcbench/`, runnable directly as `python -m kcbench.evaluate` if you
prefer.

Build (`cb.py build` runs these in order):

```
cb.py holdout      data/holdout.json              reserve the evaluation documents
cb.py tracks       data/track{1,2,3}*.jsonl       mine items, record provenance
cb.py probe        data/probe_trained.jsonl       mine the training-side probe set
cb.py usecases     data/uc*.jsonl                 use-case tracks from the config registry
cb.py split        data/train/                    training split, holdout excluded
cb.py verify       data/provenance.jsonl          provenance and contamination checks
cb.py export       data/export/                   manifest, dataset card, harness tasks
```

Score and compare:

```
cb.py eval         data/runs/<tag>.json           score tracks 2, 3 and the use cases
cb.py ppl          data/runs/<tag>.json           track 1 perplexity
cb.py matrix       data/matrix.{json,md}          score several models at once
cb.py compare      data/runs/compare_*.json       before/after, with a significance test
```

Review (a person judges; the tooling only selects and applies):

```
cb.py triage       data/review_queue.jsonl        pick what a reviewer should read
cb.py review       data/reviews.json              apply verdicts, kept across rebuilds
```

### Quick start

```bash
# 1. Build everything, and fail the build if any item turns out contaminated
python cb.py build -i ../ai_ready_full --strict

# 2. Baseline the checkpoint you are about to fine-tune
python cb.py ppl  -m Qwen/Qwen3-8B --tag base-ppl
python cb.py eval -m qwen3:8b --tag base --closed-book --tracks 2

# 3. Fine-tune on data/train/, never on the source dataset, then score again
python cb.py ppl  -m ./out/qwen3-8b-dapt --tag ft-ppl
python cb.py eval -m qwen3-ft:v1 --tag ft --closed-book --tracks 2

# 4. The number you actually wanted, with a paired significance test
python cb.py compare --base base --after ft --markdown report.md
```

A scoring pass over a full track takes hours. `run_resumable.sh` wraps it in a
supervisor that retries and resumes:

```bash
./run_resumable.sh qwen3-ft:v1 ft-probe:probe ft-t2:2
```

Every scored item is journalled to `data/runs/.ckpt/` as it completes, so a run
that dies picks up where it stopped rather than starting over. The journal is
deleted once the track writes its score.

### Generalisation and acquisition

Two sets, two questions. Track 2 is mined from held-out documents, so 75 percent
of its answers appear nowhere in the training data and cannot be learned — it
measures generalisation to regulation the model never saw. `probe_trained.jsonl`
is mined from the training side, where 83 percent of answers are present, and
measures whether training put the corpus into the weights at all.

| | track 2 | probe |
|---|---|---|
| Drawn from | held-out documents | trained-on documents |
| Answer present in training data | 25% | 83% |
| Measures | generalisation | acquisition |
| Reporting | benchmark score | diagnostic only, never quote as a score |

Read them together. Probe up and track 2 flat means the model memorised the
corpus without generalising; both flat means the training did not take at all.
Probe rows carry `split: "train"` and `contamination: "intentional"` so the two
cannot be mixed up.

```bash
python cb.py eval -m qwen3:8b --tag base-probe --tracks probe --closed-book
```

### Open book, closed book

`--closed-book` withholds the passage and names the document and article
instead. The distinction decides what the score means:

| | open book | closed book |
|---|---|---|
| Prompt | includes the clause | document title and article number only |
| Measures | reading comprehension | domain knowledge |
| qwen3:8b baseline | numeric 0.953 | numeric 0.166 |
| Use it for | regression check after training | **measuring the fine-tune** |

Open book is near its ceiling on any competent model — 3B scores 0.849 and 30B
scores 0.932, so it cannot separate a fine-tune from its base. Use it to check
that training did not cost the model its reading comprehension, and read the
closed-book number as the result. `cb.py matrix --book both` reports the gap
between them, which is the headroom fine-tuning has to work with.

### CLI

Every command takes the same path flags, and each one overrides `config.json`:

| Flag | Meaning |
|---|---|
| `-c, --config FILE` | settings file (default `./config.json`) |
| `-i, --generated-dir DIR` | chunked training data to evaluate, e.g. `../ai_ready_full` |
| `--corpus-dir DIR` | source PDFs and IFC models |
| `--metadata-dir DIR` | collector catalogues |
| `--pipeline-dir DIR` | checkout supplying the chunker, so chunking matches |
| `-o, --out-dir DIR` | where artefacts are written |
| `--seed N` | holdout seed |
| `-v, --verbose` | debug logging |

Resolution order is **defaults → config.json → command line**, so keeping one
config per dataset variant and redirecting a single run with `-i` both work:

```bash
python cb.py build --config configs/v3.json -i /data/ai_ready_v3 -o /data/bench_v3
python cb.py build --skip-holdout --tracks 2,3      # re-mine without re-splitting
python cb.py eval  --config configs/v3.json -m qwen3:8b --tracks 2,3 --limit 20
```

Stages can be skipped individually (`--skip-split`, `--skip-verify`,
`--skip-export`), which is what you want when re-running one part of a build.

`--limit` runs the first N items per track — use it to smoke-test the harness
before committing to a full scoring pass.

### Settings that change what gets built

All live in `config.json`, and every one is overridden by the matching command
line flag. Keep one config per dataset variant rather than editing this one.

What to change when the dataset grows:

| Key | Effect |
|---|---|
| `holdout.chunk_fraction` | share of chunks reserved. 0.20 yields ~395 items and withholds 16.3% |
| `track2_sft.target_items` | ceiling on the item count |
| `track2_sft.max_items_per_doc` | buys items without withholding more training data, up to ~12 |
| `probe.target_items` | size of the training-side probe set |

What to change when the model or the machine changes:

| Key | Effect |
|---|---|
| `eval.num_ctx` | context per request. Unset, the server sizes the KV cache from the model maximum |
| `eval.num_predict` | generation budget. A reasoning model needs 4096; a non-reasoning one is fine at 512 |
| `eval.temperature` | sampling temperature, 0 for scoring |
| `eval.repeats` | samples per item. 1 is correct at temperature 0, where decoding is deterministic |
| `eval.max_consecutive_failures`, `eval.max_consecutive_empty` | when to abort rather than score a dead server |
| `eval.ollama_base_url`, `eval.request_timeout` | where the server is and how long to wait |
| `eval.models`, `eval.book` | defaults for `cb.py matrix` |
| `perplexity.max_length` | tokens per chunk. Lower it if track 1 runs out of memory |
| `perplexity.dtype`, `perplexity.device` | `float16` and a different `device_map` for a smaller card |

What to change when the question is statistical:

| Key | Effect |
|---|---|
| `compare.alpha` | significance threshold for McNemar and the bootstrap interval |
| `compare.bootstrap_rounds`, `compare.bootstrap_seed` | resampling depth and reproducibility |
| `triage.min_models` | how many runs an item needs before disagreement means anything |
| `triage.sample` | unflagged items drawn so review can state an error rate |

What to change when item quality is the problem:

| Key | Effect |
|---|---|
| `track2_sft.numeric_uniqueness` | `strict` drops thresholds whose subjects overlap |
| `track2_sft.nameset_require_lead_in` | demands the sentence that introduces a list |
| `track2_sft.min_subject_len`, `max_subject_len` | bounds on the question subject |
| `holdout.ifc_exclude_categories` | categories never used for evaluation |
| `holdout.ifc_min_element_types` | drops IFC models too trivial to ask about |
| `track3_vlm.mapping_max_total_elements` | no counting questions above this size |

Changing `holdout.seed`, `holdout.chunk_fraction` or any `track2_sft` mining rule
produces a different item set, and scores across the change are not comparable.
Re-baseline after any of them.

---

## The three tracks

### Track 1 — DAPT, scored by perplexity

Held-out regulation text, no labels. Perplexity is the earliest signal that
domain-adaptive pre-training did anything: it moves long before answer accuracy
does, and if it does not move, nothing downstream will.

Lower is better. `cb.py compare` inverts it before calling a change an
improvement.

Perplexity is the exponential of the mean negative log-likelihood the model
assigns to the text it is reading — how many equally likely options it was
effectively choosing between at each token. It needs a log-probability for every
token of the *input*, which an inference server will not give you: Ollama returns
logprobs only for tokens it generated, which is the likelihood of the model's own
continuation and a different quantity. `cb.py eval` probes once and reports
`null` rather than answering a question nobody asked.

`cb.py ppl` measures it properly, running the weights through transformers:

```bash
pip install transformers accelerate       # torch is already required
python cb.py ppl -m Qwen/Qwen3-8B --tag base-ppl
python cb.py ppl -m ./out/qwen3-8b-dapt --tag ft-ppl
```

It takes an HF checkpoint rather than an Ollama tag, which is the form a
fine-tuned model is in before it is converted for serving. Reports the
token-weighted figure (the one to compare), the per-chunk median, and a
per-category breakdown. Qwen3-8B scores 7.589 over 5,381 chunks; a DAPT run that
leaves that unmoved has not trained, whatever the downstream numbers say.

### Track 2 — SFT, verifiable facts from held-out regulation

Questions whose answer is a string the source document contains. Two shapes:

- **numeric** — a stipulated threshold: `연면적은 몇 ㎡ 이상이어야 하는가?` → `2,000 ㎡`.
  Only numbers carrying a qualifier (이상 / 이하 / 미만 / 초과 / 이내) are mined,
  because that qualifier is what marks a figure as a *requirement* rather than a
  date, a clause number or a page reference.
- **nameset** — the entries of an enumerated list, scored as set precision /
  recall / F1 so a partial answer scores partially.

Admission is strict, and the reasons are counted in `data/rejections.json`:

| Rule | Why |
|---|---|
| subject must be a clean noun phrase | a fragment like `있는 유량측정장치를 펌프 정격토출량` is not a question |
| no unbalanced brackets, no swallowed measurement, no second clause | each produced unanswerable v1 items |
| overlapping subjects sharing unit and qualifier are dropped | `펌프 정격토출량` 150% and `…를 펌프 정격토출량` 175% cannot be told apart |
| enumerations need the sentence that introduces them | otherwise the question is built from whatever line the chunk starts with |
| a list running to the chunk edge is rejected | it was cut, so the answer key is incomplete |
| repealed entries (`<삭제 …>`) are rejected | they are not items |

Where no lead-in sentence exists, the governing clause heading (`제9조(전원 등)`)
identifies the list instead — the passage is given to the model, so the heading
names exactly one list within it.

### Track 3 — VLM, IFC elements against the file itself

The model names the structural elements in a synthesised site photo, and
`bim_elements.json` records exactly which elements went into that render.

- **nameset** — which IFC types are present (F1)
- **mapping** — type → count (key F1 and value accuracy, reported separately)

Two limits are enforced rather than documented and ignored. Parser regression
fixtures (`06_reference_and_test` — files named `wrong-geometry`, `segfault`,
`infiniteLoop`) are excluded: they are 57 of the corpus's 93 models, they hold
one or two elements each, and a proportional split fills the holdout with them.
And counting is only asked of models small enough to count from a photograph.

That leaves a small track. It is a directional signal, not a ranking, and the
corpus is the reason: after exclusions only 12 models qualify.

---

## Use-case tracks

The agent this model is being trained for has five use cases: safety checks
(UC1), rebar inspection against the specification (UC2), site photo versus BIM
(UC3), regulation-violation investigation over provided context (UC4), and
incident analysis from a method statement and checklist (UC5). The target is
80% output accuracy for the fine-tuned model behind RAG, which makes **open
book the operative measurement** — retrieval hands the model the clause, and
the track asks whether it can use it.

`cb.py usecases` builds one file per use case from the `usecases` registry in
config.json. Each entry names a builder and its parameters, so a future use case
is a config entry plus at most one builder function:

| UC | File | Builder | Items | Answer key |
|---|---|---|---:|---|
| uc1_safety | `uc1_safety_qa.jsonl` | `doc_filtered_qa` | 157 | figures and lists from safety documents, reusing the track 2 miner |
| uc2_rebar_spec | `uc2_spec_threshold.jsonl` | `doc_filtered_qa` | 150 | numeric limits from design standards and specifications |
| uc3_bim_site | `uc3_cross_image.jsonl` | `vlm_labels` | 39 | the judgement label attached at generation, `unknown` excluded |
| uc4_faithfulness | `uc4_faithfulness.jsonl` | `context_swap` | 160 | track 2 items whose clause was swapped for an unrelated one |
| uc5_incident | `uc5_incident.jsonl` | `missing_measures` | 118 | the measures withheld from a clause's own enumeration |

Three of these need explanation.

**uc4 — faithfulness to the passage.** Retrieval fails in every deployed RAG
system, and the closed-book runs showed what happens next: zero abstentions and
invented numbers. Half the items carry a context swapped for an unrelated
passage (verified not to contain the keyed answer); the item is correct only
when the model answers `자료 없음` (no information). The other half are
unmodified controls, so a model that abstains on everything scores 50%, not
100%. This is the track to watch for agent safety.

**uc5 — incident analysis.** The corpus holds no real incident reports, so the
scenario is synthetic: the item lists the measures an investigation confirmed on
site and asks which of the clause's required measures are missing. The ground
truth is the withheld subset of the clause's own enumeration. Grading is
token-coverage matching (`match_mode: fuzzy`) rather than exact lines, because
the answers are prose and a correct abbreviation should score.

**uc3 — label caveat.** The ground truth is the label the generation pipeline
attached, an annotation rather than a measurement like the IFC catalogue, and
the comparison labels skew heavily to `mismatch`. Items whose label is
`unknown` are excluded (`exclude_labels`): they grade honest judgement as
wrong, and removing them moved the same model's score 0.375 → 0.590. Read
scores with the `by_task` split, and treat the track as weaker evidence than
tracks 2 and 3 until the labels have been human-reviewed.

**Contamination marking.** Safety work standards and most specification
documents sit on the training side of the split, so UC1/UC2/UC5 items drawn from
them carry `split: "train"` and `contamination: "intentional"`, same as the
probe set. The aggregate reports a `by_split` breakdown; only holdout-side items
may be read as closed-book evidence of generalisation. Open book — the RAG
measurement — is fair on both.

Score them with `cb.py eval --tracks uc` for all of them, or name one:
`--tracks uc4_faithfulness`. A newly registered use case is picked up without a
code change.

---

## Provenance and contamination

Every item records where it came from, and the claim is checkable:

```jsonc
"provenance": {
  "dataset": "ai_ready_full",
  "dataset_file": "01_design_standards_kds/개정전문 (…)/dapt_training_data.jsonl",
  "row_index": 2,
  "char_start": 559, "char_end": 564,
  "chunk_sha256": "cbc5d495f973e817…",
  "document": "…", "category": "…", "source_name": "….pdf"
}
```

`cb.py verify` re-checks four things per item and writes `provenance.jsonl`
plus `contamination_report.json`:

| Check | What it proves |
|---|---|
| `source_exists` | the generated file named in the item is present |
| `source_matches` | the row at that index still hashes to the recorded digest |
| `in_holdout` | the document is on the reserved list |
| `absent_from_train` | no training row carries that text, **by digest** |

The last one is by digest over every text field of the training split, not by
document name — and that is not a formality. This corpus collects the same
regulation more than once: under names differing by a suffix (`…_20240724` and
`…_20240724__1`), as amendment-and-original pairs, and as sibling standards
sharing whole clauses. Excluding held-out documents by name left **759 of their
chunks in the training data** under a sibling's name. `cb.py split` drops those
rows too and reports them separately.

Run `python cb.py split --check` before any training run. Train from
`data/train/`, never from the source dataset directly. `cb.py holdout` also
mirrors the reserved list to `BENCHMARK_HOLDOUT.json` at the project root, so a
corpus regeneration can skip those files.

**Do not change the seed** once you have scored a base model. A different seed
reserves different documents, and two runs on different items are not comparable.

## Review

Deciding whether a question is answerable and correctly keyed is a person's job.
Deciding *which* questions are worth that person's hour is not, and that is the
part the tooling does. Reviewing 395 items is a day; reviewing the 30-odd that
several models fail in agreement is twenty minutes and catches most of what a
full pass would.

```bash
python cb.py matrix --models qwen3:8b,glm4:9b,qwen3:14b --book open
python cb.py triage --runs-glob '*--open' --sample 30
# fill verdict (ok / broken / ambiguous) and fixed_answer in review_queue.jsonl
python cb.py review --queue data/review_queue.jsonl --apply
```

`cb.py triage` queues three kinds of suspect, plus a random sample:

| Signal | What it usually means |
|---|---|
| `consensus_wrong` | every model missed it with the answer on the page — the question is the problem |
| `disagreement` | models answered different numbers, all present in the passage — more than one reading |
| `no_answer` | models returned nothing — malformed item or a prompt that provokes refusal |
| `random_sample` | not suspect; drawn so the review can state an error rate for the whole set |

The sample rows are what the error rate is computed from. Triaged rows were
selected *because* they looked wrong, so a rate measured on them reports a defect
level several times the real one — `cb.py review` keeps the two apart and reports
a Wilson interval.

Verdicts land in `reviews.json` and the track builder reads it on every later
build, dropping anything marked broken or ambiguous. Item ids hash the question's
content, so a verdict survives a rebuild for as long as the question is worded
the same way; a reworded question correctly becomes unreviewed again. Review does
not have to be redone every time the corpus grows.

## Packaging

`cb.py export` writes `data/export/`:

- `manifest.json` — sha256 and line count per artefact, item counts per track and
  per category, the config the build used, and the contamination summary. Two
  builds are identical if their manifests are.
- `DATASET_CARD.md` — Hugging Face card layout: what it measures, structure,
  provenance, limitations, licensing.
- `lm_eval/*.yaml` — lm-evaluation-harness task files. Its `exact_match` is
  stricter than the grader here, so use `cb.py eval` for the reported number.

## Grading

Typed, not text-scraped. Scraping numbers out of prose rewards verbosity: a long
answer containing the right digits somewhere is not a correct answer. Numbers are
parsed and compared with relative tolerance (`numeric_tolerance`, default 2%);
namesets compare as sets after Unicode and punctuation normalisation; mappings
key by key. Each item is sampled `repeats` times.

Two harness settings are not optional:

- **`num_ctx`** — without it the server sizes the KV cache from the model's
  maximum context, which on qwen3 is 131k per sequence and 40 GB of KV cache on a
  partial offload.
- **`num_predict`** — a reasoning model spends most of its budget thinking. At
  1024 the budget ran out mid-thought on a quarter of items and returned an empty
  answer, which grades as wrong and reads as a knowledge failure. Runs record
  `no_answer` per type so this is visible rather than inferred.

### Precedent for each grading type

Every grading type here has a precedent in a published benchmark. The point is
not novelty; it is that a reviewer can recognise what is being computed.

| Type | Grading | Precedent |
|---|---|---|
| numeric | first figure, 2% relative tolerance | DROP, FinQA and other numeric-extraction QA |
| nameset | set precision / recall / F1, partial credit | SQuAD and DROP token-F1 over multiple spans |
| nameset (fuzzy) | 60% content-word coverage, greedy 1:1 | same intent as DROP's partial matching. uc5 only: demanding exact strings of a prose answer measures transcription, not knowledge |
| label | match within a closed vocabulary | MMLU and KMMLU multiple-choice accuracy. uc3, after `unknown` is excluded |
| faithfulness | abstention plus paired controls | SQuAD 2.0 (unanswerable and answerable at roughly 1:1), RGB's negative rejection |
| cognitive_level tag | not scored, carried per item | AECBench's five levels (recall, understand, reason, calculate, apply) |

SQuAD 2.0 is the reason uc4 is built the way it is. An always-abstain baseline
scores 48.9 F1 there, so without controls an abstention rate masquerades as a
score. uc4 mixes 50% unmodified controls for the same reason, which caps a
blanket abstainer at 0.5.

RGB (AAAI'24) separates four RAG abilities: noise robustness, negative
rejection, information integration, counterfactual robustness. uc4 targets the
first two — an unrelated clause is 100% noise. The other two are not measured
yet.

### Track size, and what it can decide

Per-task item counts are in the same range as published benchmarks. Multiple
choice scales cheaply, which is why KMMLU runs to hundreds per subject;
extraction-and-verification sets like FinanceBench (150 items) sit in the tens
to low hundreds.

| Benchmark | Total | Tasks | Per task |
|---|---:|---:|---:|
| AECBench (Chinese AEC) | ~4,800 | 23 | ~200 |
| KMMLU | 35,030 | 45 subjects | ~780 (multiple choice) |
| **kcbench** | ~1,300 | track 2 + probe + 6 use-case tracks | 39-400 |

Size decides what a track can settle. Taking the half-width of a 95% binomial
interval near p = 0.8, which is the accuracy target these tracks were built to
test:

| Track | Type | Items | CI half-width | Can it decide 80%? |
|---|---|---:|---:|---|
| uc1_safety | numeric | 85 | ±5-8pp | yes |
| uc1_safety | nameset | 72 | ±9pp | marginal |
| uc2_rebar_spec | numeric | 150 | ±4-5pp | yes |
| uc3_bim_site | label | 39 | ±15pp | directional only, and the labels are annotations |
| uc4_faithfulness | abstain/control | 160 | ±5pp | yes |
| uc5_incident | nameset | 118 | ±7pp | yes |

## Grading, continued

Do not set `think: false` on a thinking model to avoid this: it does not stop the
model reasoning, only stops the server separating the reasoning out, so the chain
of thought lands in `response` and the answer never arrives.

## Output

`cb.py eval` writes `data/runs/<tag>.json` with per-item scores, per-type
aggregates including `no_answer`, and a headline number per track. `cb.py
compare` reads two of those and prints the deltas plus a per-category breakdown.

Long runs journal to `data/runs/.ckpt/<tag>-<track>.jsonl` while they are in
flight. Two guards stop a dead inference server from being scored as a bad
model: a run of failed calls aborts the track, and so does a run of blank
replies, which is what a server that answers but no longer generates looks like.
Neither writes a score file, and neither poisons the journal — blank replies are
never journalled, so a resumed run retries them.

## Known limits

- Items are rule-mined, not expert-reviewed. Good enough for a before/after
  delta; not sufficient for a public leaderboard without review (AECBench and
  KMMLU both spent most of their effort exactly there). `cb.py triage` narrows
  what a reviewer has to read to roughly a tenth of the set, and `cb.py review`
  keeps the verdicts across rebuilds, but the judgement is a person's.
- Track 2 numeric questions inherit whatever ambiguity the source clause has.
- Track 3 is bounded by how few real building models the corpus holds.
- Track 1 needs an HF checkpoint and a GPU; it cannot be scored through Ollama.

Design decisions are recorded with the measurement that motivated them; see
"Precedent for each grading type" above for how the grading was checked against
published work.
