# Designing the agent around these results

The benchmark answers one question: *did training help, and where?* This
document answers the next one. Given what the four-recipe campaign in
[README.md](README.md#worked-example-a-korean-construction-corpus) measured,
how should the retrieval-augmented construction agent actually be built?

Every claim here either traces to a number in that campaign or carries a
citation. Where something is inference rather than measurement, it says so.

---

## What these conclusions depend on

Read this first. Nothing below is a general result about fine-tuning, retrieval
or agent design. Every conclusion is conditioned on four things, and changing
any of them can change the answer.

**The corpus — its scale and how it was made.** Stage 1 saw 3.7 million tokens
(26,767 chunks from 865 documents) in a single pass; stage 2 trained on 15,666
instruction pairs. Continued pre-training that demonstrably adds domain
knowledge is normally reported at billions of tokens with several passes per
fact, so this is roughly three orders of magnitude below that. The pairs are
also synthetic, generated for retrieval-augmented use: 95% carry the source
passage in the prompt and answer in a median of 56 characters, which teaches
extraction and brevity rather than recall. **A larger or differently generated
corpus could plausibly move the closed-book recall that four recipes here could
not.** The augmentation turns rearranged the same 3.7M tokens; none of them
added information the corpus did not already hold.

**The adapter — its parameter budget.** Stage 2 used LoRA at rank 64: 174.6M
trainable parameters, about 2.1% of an 8B model. Knowledge injection is
believed to scale with adapter capacity, so a larger adapter, a different
tuning method, or full fine-tuning may not behave as this campaign did.

This is not a hypothetical. **Closed-book knowledge injection has been achieved
at a corpus scale comparable to this one**, and the published cases differ from
this campaign in three specific ways at once — all of them things it did not do:

| | This campaign | [Knowledge-Instruct (2025)](https://arxiv.org/html/2504.05571v1) |
|---|---|---|
| Tuning | LoRA rank 64 (2.1% of weights) | **full SFT** |
| Restatements per fact | 2 | **5**, plus 25 instruction templates |
| Preparation | rule-mined from clauses | entity and fact extraction, contextualisation, deduplication, then paraphrase |
| Result | closed-book flat, 0.147 → 0.156 | **0% → 81.8%** on new facts; 13.2% → 76.8% on long-tail facts, Llama-3.1-8B |

Two further conditions are worth naming because they are cheap to get wrong.
Paraphrase counts are reported to help until roughly three restatements per
fact and to plateau after; this campaign used two, below that point. And
[capacity saturation at rank 64 has been observed directly](https://arxiv.org/html/2607.21861v1)
— recall collapsing from 71% to 50% as documents were added, restored by
doubling to rank 128 — though the same line of work found that
[raising the rank without raising the learning rate with it can make things
worse](https://aclanthology.org/2025.findings-ijcnlp.58.pdf), and that
[higher ranks buy memorisation without necessarily buying accuracy](https://arxiv.org/pdf/2502.05087).

So the honest statement of this campaign's central negative result is narrower
than "fine-tuning does not add knowledge here". It is: **at rank 64, with two
restatements per fact, on clause-mined pairs, four recipes moved closed-book
recall by nothing.** The published successes changed all three of those
variables together, which is also why this campaign cannot say which one
mattered.

**The GPU — one memory pool, shared.** The machine has 128 GB of *unified*
memory: the CPU, the GPU, the page cache and every process draw on the same
pool, so there is no separate VRAM to protect a training job. Training this 8B
model has crashed on it five times, always while the benchmark was scoring on
the same box — which turns out to matter for what can be concluded, and is
worked through under [the memory ceiling](#the-memory-ceiling-and-what-it-does-not-prove).
The short version: the crashes are best explained by concurrency and allocator
fragmentation rather than adapter size, so the larger-adapter experiment is
available and has simply not been run yet.

**The base model.** Qwen3-8B. In particular, its untrained abstention rate of
0.950 is what makes "do not train what the model already does" the right advice
here. A base model that abstained poorly would justify the opposite conclusion.

The latency arithmetic later on has the same character: it describes one class
of machine, named there, and does not transfer unchanged to a datacentre GPU.

---

## Terms used here

The campaign's vocabulary is compressed. Spelled out:

**Supporting passage.** The stretch of a regulation document where the answer to
a question is actually written. An item is built from one, and the scoring mode
decides whether the model gets to see it.

**v1 through v4.** Four fine-tunes. The base model, the stage 1 adapter, the
hyperparameters and the evaluation items were all held fixed; **only the stage 2
training set changed.** So a score difference is attributable to the training
data and nothing else.

| Version | What was added to the training set | Pairs |
|---|---|---:|
| `v1` | nothing — the generated question-and-answer pairs as they came | 15,666 |
| `v2` | questions with the supporting passage deleted, so the model must answer from memory (~5,900); questions demanding a complete list (~890) | 22,249 |
| `v3` | each memory question restated several ways (~7,600); **refusal practice** (1,920) | 32,080 |
| `v4` | identical to v3 except the refusal practice is made harder | 32,005 |

**Refusal practice.** A question is deliberately paired with a passage that does
not answer it, and the correct answer becomes *"the given text does not support
an answer."* In v3 the mismatched passage comes from an **unrelated document**,
which is easy to spot because the vocabulary is obviously off-topic. In v4 it
comes from the **same document**, so terminology and style match and the model
cannot decide by topic alone. v4 is the harder version of the same lesson.

**Abstention rate.** The share of unsupported questions the model correctly
declined to answer instead of inventing something. **Higher is better.** It is
never read alone — the track pairs unsupported questions 1:1 with properly
supported ones, because a model that refuses everything would score 1.0 on
abstention while being useless. Both numbers have to be high.

**top-k, and recall@k.** When retrieval is attached, the system searches the
document store per question and pastes the *k* most similar fragments into the
prompt. top-3 pastes three, top-10 pastes ten. **recall@k** is the share of
questions where the fragment that actually contains the answer was among those
*k*. Pasting more fragments raises the chance the right one is included, which
is why both recall and accuracy rise from top-3 to top-10.

---

## The three findings that constrain the design

**1. Fine-tuning changed how the model handles text it is given; it never
changed what the model knows.** Four data recipes produced closed-book recall of
0.153, 0.128, 0.156 and 0.144 against an untrained baseline of 0.147, with no
comparison reaching significance. Meanwhile calibration error fell 0.641 → 0.281
and open-book reading stayed intact. The conclusion is not that fine-tuning is
worthless — it is that **fine-tuning is the wrong instrument for knowledge
injection here, and the right one for output quality.**

**2. Retrieval quality dominates the fine-tuning budget.** On the same items,
moving from a badly chosen embedding model to a good one is worth roughly 0.35
accuracy. Four rounds of fine-tuning moved recall by nothing.

**3. Every attempt to train refusal made refusal worse.** Abstention fell
monotonically as refusal training was added and sharpened: 0.950 untrained,
0.925 at v1, 0.750 at v2 and v3, 0.688 at v4.

The first two point at where to spend effort. The third is the one that changes
the architecture, so it is worth understanding before designing around it.

---

## Why the abstention collapse happened

### What was measured

The first significant drop is at **v2**, 0.925 → 0.750 (p = 0.0005) — and v2
contains no refusal training at all. What v2 added was ~5,900 questions with the
supporting passage stripped out, whose lesson is *answer even when no passage is
given*. That is the direct opposite of *decline when the passage does not
support an answer*, and the second capability broke immediately after the first
was trained in.

v3 and v4 then added refusal practice specifically to repair it. Neither did.
v4, the harder version, made it worse and took collateral damage with it:
accuracy on **properly supported** passages fell 0.988 → 0.938, and calibration
error rose 0.281 → 0.349.

That collateral damage is the diagnostic. A model that had learned *refuse when
unsupported* would refuse more precisely and leave supported questions alone. A
model that had merely become less willing to commit would lose ground on both —
which is the observed pattern. The effect reads as **interference, not
instruction**.

### What is inferred

Four explanations, none of them individually proven:

**There was no deficit to fill.** The untrained model already abstains correctly
95% of the time. Training a capability the model already has offers nothing to
gain and plenty to disturb, and a monotonic decline is what that looks like.

**The dominant signal in the training set runs the other way.** Of ~32,000
pairs, the refusal examples are 1,920 — about 6%. The other 94% all teach *a
question has an answer, produce it*. A minority rule does not survive contact
with a majority rule; the reflex that gets reinforced is "answer anyway."

**v4's difficulty increase may have backfired specifically.** Drawing the
mismatched passage from the same document means the vocabulary overlaps. But in
the overwhelming majority of the training set, a passage from the same document
*does* contain the answer. The model therefore received far more evidence for
"overlapping vocabulary means the answer is here" than against it, and the
counterexamples were swamped.

**Domain tuning may erode general alignment.** Refusal behaviour is installed by
the model vendor's general alignment training. Continued training on a narrow
domain weakening it is a commonly reported effect.

### One thing to rule out first

Abstention is graded by matching refusal phrasing in the reply. A fine-tuned
model that learned to decline in *different words* than the grader recognises
would score as if it had stopped refusing. This benchmark has already produced
three faults of exactly this kind — a serving-template mismatch, a
reasoning-parse mismatch, and a grader format bias — so **read a sample of the
v4 replies before accepting the number.** If the wording changed, the finding is
a grader bug; if it did not, everything above stands.

---

## Solution 1 — separate generating from verifying

The campaign tried to put two jobs in one set of weights: *summarise and reason
over construction regulation*, and *decline when the evidence does not support
an answer*. They pulled against each other, and the second lost. The
architectural answer is to stop asking one model to do both.

Splitting a generator from a verifier is an established pattern, not an
improvisation: the generator produces an answer, and a separate component judges
only whether that answer is supported by the retrieved passages. Production
guardrail stacks are layered this way rather than concentrating every judgement
in the largest model — one published reference stack pairs a flow-control layer
(NVIDIA NeMo Guardrails) with an 8B classifier (Llama Guard 3) and an
86M-parameter first-pass filter (Prompt Guard). That is cited as an existence
proof of the shape, not as the stack to adopt; equivalent layered designs exist
from other vendors and in open source.

The important detail is that **the second model does not need to be large.** Its
task is narrow and its output is a verdict, not prose. Three published models
illustrate the size range — again as examples of a category, not
recommendations. Others in the same class exist, the field moves quickly, and
the right choice is whichever scores best on *your* held-out data:

| Verifier | Size | Reported result |
|---|---:|---|
| Patronus Lynx | 8B | +24.5% over GPT-3.5 on HaluBench |
| MiniCheck-FT5 | 770M | GPT-4-level accuracy at ~400× lower cost |
| Vectara HHEM-2.1-Open | 110M | <600 MB RAM; 2k-token input in ~1.5 s on ordinary CPU |

The figures are the publishers' own, measured on English benchmarks. Treat them
as evidence that small verifiers are viable, not as scores that will reproduce
on Korean regulation — see option **c** below.

Four ways to implement the split, cheapest first. Work down the list only as far
as the accuracy requirement demands.

### a. Toggle the adapter — start here

Stage 2 was LoRA. Detaching the adapter returns the original general-purpose
model, **and that model already abstains at 0.950** — the best score any
configuration in the campaign achieved. So the "second model" already exists: it
is the same weights with the adapter off. Attach it for domain generation,
detach it for the supported-or-not judgement. The extra memory cost is the
adapter itself, a few hundred megabytes.

vLLM supports holding multiple adapters and switching per request. The caveat is
throughput: a reported case shows up to a 50% drop in maximum throughput with a
LoRA adapter active versus the base model, with degradation of roughly 24–47%
depending on adapter rank. **Measure this on the target hardware before
committing** — it is the one number that could make this option unattractive.

### b. Decide with rules before involving a model

Two checks that cost nothing because they add no inference call:

- **Retrieval-score threshold.** If the best retrieved fragment scores below a
  floor, decline without asking the model at all. This is the cheapest possible
  abstention and it is exactly right for the failure mode that matters — no
  relevant passage was found.
- **Mandatory citation.** Require the answer to quote a span from the retrieved
  passage, and treat an answer without a verifiable quote as a refusal. Turns
  faithfulness into a string check.

Both fit the plan to prepare retrieval per use case, because a per-use-case
threshold can be tuned against that use case's own held-out questions.

### c. Add a small dedicated verifier

If rules are not enough, a 110M–770M verifier runs on CPU and never touches the
GPU serving the generator. **The blocker is language:** these models are
English-centred, and this project has already measured what English-centred
models do to Korean regulation — the retrieval sweep found they return the right
passage roughly 2% of the time at top-3 against 24% for multilingual ones.
Validate any such verifier on a Korean held-out set before trusting it. Assume
nothing transfers.

### d. A dedicated 8B verifier, last

An 8B verifier such as Lynx is the highest-accuracy option and the most
expensive. On this hardware two 8B models compete for the same memory bandwidth,
so it is worth reserving for the specific use cases where the cost of an
unsupported answer justifies it — safety lookups, for instance — rather than
applying it to every request.

---

## Solution 2 — spend the budget on retrieval first

The measured ladder, base model, numeric accuracy on the held-out set:

| Condition | recall@k | accuracy |
|---|---:|---:|
| no retrieval (closed book) | — | 0.147 |
| retrieval with an English-centred embedder, top-10 | 0.041 | 0.144 |
| retrieval with `bge-m3`, top-3 | 0.240 | 0.397 |
| retrieval with `bge-m3`, top-10 | 0.400 | 0.481 |
| retrieval with `snowflake-arctic-embed2`, top-10 | 0.428 | 0.478 |
| perfect retrieval (the passage handed over) | 1.000 | 0.944 |

This ordering is not peculiar to this corpus. A controlled comparison of the two
approaches found [retrieval outperforming fine-tuning consistently, for both
knowledge the model had seen in training and knowledge it had
not](https://arxiv.org/abs/2312.05934) — so "retrieval first" is a recommendation
this campaign's numbers agree with rather than one they establish alone.

Three things follow.

**The open-book figure is a ceiling, not a forecast.** 0.944 is what the system
scores when handed exactly the right passage. It describes the headroom
available to retrieval, not the accuracy of a deployed system. The honest
current number for an end-to-end system on this corpus is 0.481.

**That ceiling is worth chasing, because the gap is retrieval's, not the
model's.** At top-10 the retriever finds the right passage 40% of the time and
the system answers 48% of questions — the model is already doing slightly better
than its retriever by working from neighbouring passages. Raising recall raises
accuracy almost directly.

**Retrieval is being built separately, and should be.** Reranking, and retrieval
tuned per use case rather than one global index, are both expected to push
recall well above the 0.40 measured here — a target above 0.5 is reasonable, and
each 0.1 of recall has been worth roughly 0.1 of accuracy in the measured range.
That is a better return than any of the four training recipes produced.

**Validate the embedding model before anything downstream of it.** The most
dangerous result in the whole sweep: retrieval with `nomic-embed-text` scores
0.144, and no retrieval at all scores 0.147. The pipeline runs, passages are
returned, the model answers — and nothing surfaces the fact that the passages
were unrelated. Only the score shows it. Within a competent multilingual family
the choice barely matters (0.481 vs 0.478); across families it is the difference
between a working system and a broken one.

---

## Solution 3 — do not train what the base model already does

The campaign's clearest negative result is that refusal training made refusal
worse at every dose. The corresponding rule is narrow and firm: **capabilities
the base model already performs well are not fine-tuning targets.** Abstention
at 0.950 is not a gap; it is an asset to be preserved, which argues for keeping
it in the un-adapted model (Solution 1a) rather than trying to reproduce it in
the adapted one.

**On deployment choice.** For a retrieval-augmented agent — where the passage is
supplied and the model's job is to use it or decline it — **v1 is the checkpoint
to deploy.** It has the highest abstention of any fine-tune (0.925 against the
base's 0.950), halved calibration error, intact reading, and it is the cheapest
of the four to produce. v2 through v4 spent training budget on recall they never
gained and paid for it in honesty they already had.

**On adapter capacity.** The campaign suggests the remaining levers for
knowledge injection are on the pre-training side: paraphrase-augmented
pre-training text, more passes over each fact, or more adapter capacity. All
three remain open. The third looked blocked by hardware and is not — see the
next section.

**If refusal training is attempted anyway**, two changes follow from the
analysis:

- Raise its share well above 6%, or accept that the majority signal will
  overwrite it.
- Do not ship it in the same training set as passage-removed questions. One
  teaches *answer without evidence* and the other teaches *decline without
  evidence*; the campaign shows what happens when both are present.

---

---

## Solution 4 — full fine-tuning, if recall is the goal

Only relevant if closed-book recall is a requirement. If the deployment supplies
the clause at inference, Solution 2 is the cheaper answer and this section does
not apply.

**What is known to work.** Closed-book injection has succeeded at a corpus scale
comparable to this one, and the reported recipe changes three things together:
full SFT rather than a low-rank adapter, five restatements per fact rather than
one or two, and fact extraction before paraphrasing rather than mining clauses
as they stand. On an 8B model that took new facts from 0% to 81.8% and
long-tail facts from 13.2% to 76.8%
([Knowledge-Instruct](https://arxiv.org/html/2504.05571v1)).

**What it costs on this class of machine.** An 8B model wants roughly 122 GB
before activations — 15 GB of weights, 15 GB of gradients, 91 GB of AdamW
moments and the fp32 master copy — against 128 GB shared with the OS, the page
cache and anything else running. An 8-bit optimiser brings it to about 76 GB and
gradient checkpointing covers the activations, so it is reachable, but only with
both turned on and nothing else scoring at the same time.

| | For | Against |
|---|---|---|
| Capacity | the only method with published closed-book success at this corpus scale | LoRA at rank 128 is untested here and much cheaper to try |
| Memory | fits with an 8-bit optimiser and checkpointing | does not fit as normally configured; no headroom for concurrent work |
| Time | one run answers the question | ~20 h per run on this hardware, and the recipe changes three variables at once, so a failure does not say which |
| Risk | — | full-weight training can degrade behaviour the base model already had, which this campaign saw repeatedly with a *much* smaller intervention |

**Is it worth it against better retrieval?** Unknown, and worth saying plainly:
the literature has full SFT succeeding at injection and, separately,
[retrieval outperforming fine-tuning for the same
purpose](https://arxiv.org/abs/2312.05934), but the two have not been priced
against each other at equal effort on a corpus like this. What this campaign can
report is which was cheaper *here*: retrieval moved accuracy from 0.147 to 0.481
in an afternoon, and four training rounds moved it nothing. Try retrieval first
for that reason — not because full fine-tuning is known to lose.

**Order of attempts**, cheapest first: five restatements per fact, then rank 128
with the learning rate raised alongside it, then full SFT. Each answers whether
the next is necessary.

## The memory ceiling, and what it does not prove

A machine of this class can run out of memory while fine-tuning an 8B model,
depending on how the training run is configured and what else is resident. It
did here, five times — and each time the benchmark was scoring on the same box.
That circumstance is what decides how much the crashes prove, which is worth
being careful about, because the pessimistic reading would close off the most
promising experiment still open.

**Why the memory is easy to exhaust.** The 128 GB is unified: CPU, GPU, page
cache and every process share one pool, so anything else resident subtracts
directly from what training can allocate. Rough steady-state figures for
Qwen3-8B (36 layers, hidden 4096, intermediate 12288, vocabulary 151,936),
in the configuration this campaign used:

| Process | Consumes | Approx. |
|---|---|---:|
| Training | bf16 weights | 16.4 GB |
| | LoRA parameters, gradients, AdamW state (rank 64) | ~3 GB |
| | logits at the LM head, with the float32 upcast for the loss | ~6 GB |
| | activations retained for backward, 36 layers, **no gradient checkpointing** | ~16 GB |
| `cb.py ppl` | a *second* full bf16 copy of the model | 16.4 GB |
| | `log_softmax` over a 2048-token chunk, upcast to float32 | ~3 GB spike |
| `cb.py eval` | Ollama's Q4 weights plus an 8,192-token KV cache | ~6 GB |
| | **total** | **~67 GB** |

Sixty-seven gigabytes against 128 nominally fits, and none of the individual
figures is alarming. What consumes the remainder is less visible: the operating
system and a page cache that may have just been handed a 16 GB checkpoint to
write; allocator fragmentation over a long run, which unified memory is
particularly prone to; the two logit spikes above, which are transient and can
coincide; and the fact that on a unified pool an over-allocation can end with
the kernel killing the process rather than raising a catchable CUDA error.

**What this does not show.** Every crashed run was at the default rank 64.
**Adapter size was never the variable being tested.** Training on its own
accounts for roughly 41 GB of the 128, and raising the rank is cheap in
comparison: rank 128 roughly doubles the LoRA parameters to ~349M, adding about
3 GB, and rank 256 adds about 6 GB. Given the box to itself, a substantially
larger adapter fits with room left over.

So "more adapter capacity" is **not** foreclosed by this hardware. The
experiment simply has not been run cleanly, and the project's own training notes
already prescribe it: *if probe accuracy does not move after a clean DAPT run,
raise the rank before changing anything else.* The campaign's central negative
result — that no data recipe moved closed-book recall — should therefore be read
as untested at larger adapter sizes, not as settled.

**Three changes make that test runnable**, in order of effect:

1. **Do not score while training.** Serialise the two. This is a scheduling
   change and it recovers roughly 22 GB.
2. **Turn on gradient checkpointing.** It is opt-in in this pipeline and the
   campaign's runs did not use it, so they carried ~16 GB of activations they
   did not have to. It costs about a third of the throughput and it is the
   single largest saving available.
3. **Let the allocator grow in place.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   costs nothing and takes fragmentation out of the picture on long runs.

With those, the headroom for a rank sweep is ample, and the knowledge-injection
question can be answered rather than assumed. The literature above suggests what
to change first if the goal is recall rather than a clean ablation: raise the
restatements per fact to five, raise the rank to 128 with the learning rate
adjusted alongside it, and — if those do not move it — try full fine-tuning,
which is what the published successes used.

Full fine-tuning is the other lever the literature points at, and it has its own
memory arithmetic — see [Solution 4](#solution-4--full-fine-tuning-if-recall-is-the-goal).

## Does the split cost too much time?

On the hardware this campaign ran on, no — provided the verifier is configured
to emit a verdict rather than an explanation.

That hardware is an **NVIDIA DGX Spark**, a desktop machine built on the
**NVIDIA GB10 Grace Blackwell Superchip**: a 20-core Arm CPU and a Blackwell GPU
sharing **128 GB of LPDDR5X unified memory at roughly 273 GB/s**. The bandwidth
figure is the one that matters here, and it is well below a datacentre GPU's.
Published measurements for Llama 3.1 8B on this platform give about **7,991
tokens/s reading the prompt and 20.5 tokens/s generating** at batch 1, rising to
368 tokens/s generation at batch 32.

Everything in this section is therefore specific to that machine. It is the
environment the benchmark happened to run on, not a recommended target. On a
datacentre GPU generation is far faster, which shrinks the denominator and makes
the verifier's share of end-to-end latency correspondingly larger — the
conclusion that verification is nearly free holds most strongly on
bandwidth-limited hardware like this one. Re-derive the arithmetic for whatever
you deploy on; the method below is what transfers, not the numbers.

The asymmetry is the whole argument. **Generation is slow; reading is fast.** A
verifier reads a long passage and an answer, then emits a few tokens — it is
almost entirely prompt-reading work:

| Step | Work | Rough time |
|---|---|---|
| Generator writes a ~200-character answer | generation-bound | ~10 s |
| Verifier reads ~2,000 tokens, emits a verdict | reading-bound | ~0.3 s |

The verification stage costs a few percent of end-to-end latency. Memory is not
a constraint either: two 8B models in bf16 occupy roughly 32 GB of the 128 GB
available, and the adapter-toggle approach needs only one copy.

The failure mode to avoid is making the verifier a second *generative* model
that writes out its reasoning — that doubles the slow half of the pipeline.
**Constrain the verifier to a short verdict.**

One further caveat from practice: streaming output and post-hoc verification are
in tension, since nothing can be verified until the answer is complete. And
guardrails are not perfect — some correct answers will be blocked. Both are
design tradeoffs to make deliberately, not surprises to discover in production.

---

## What to validate before committing

| # | To check | Why it matters |
|---|---|---|
| 1 | Read a sample of v4's actual replies | If refusal wording merely changed, the abstention finding is a grader bug — check before building around it |
| 2 | Measure vLLM throughput with the adapter attached | Reported degradation is 24–50%; this decides whether adapter-toggling is viable |
| 3 | Score any small verifier on Korean held-out items | These models are English-centred; the embedding sweep already showed what that does here |
| 4 | Set the retrieval-score threshold per use case | The cheapest abstention mechanism, and it needs each use case's own data to calibrate |
| 5 | Re-measure end-to-end once reranking lands | 0.481 is the current honest number; the whole retrieval argument rests on moving it |
| 6 | Run a rank sweep with nothing else on the machine, gradient checkpointing on | The knowledge-injection result is untested above rank 64; the five OOM crashes were concurrency, not capacity |

---

## Sources

- [Lynx: State-of-the-Art Open Source Hallucination Detection Model](https://www.patronus.ai/blog/lynx-state-of-the-art-open-source-hallucination-detection-model) — Patronus AI
- Liu et al., [*MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents*](https://aclanthology.org/2024.emnlp-main.499/), EMNLP 2024
- [HHEM 2.1: A Better Hallucination Detection Model](https://www.vectara.com/blog/hhem-2-1-a-better-hallucination-detection-model) — Vectara
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) — NVIDIA, for the GB10 Grace Blackwell specification
- [NVIDIA DGX Spark In-Depth Review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) — LMSYS, for the throughput figures
- [Throughput and latency degradation with a LoRA adapter](https://github.com/vllm-project/vllm/issues/10062) — vLLM issue #10062
- [Measuring the Effectiveness and Performance of AI Guardrails](https://developer.nvidia.com/blog/measuring-the-effectiveness-and-performance-of-ai-guardrails-in-generative-ai-applications/) — NVIDIA, on the layered guardrail stack
- [*Knowledge-Instruct: Effective Continual Pre-training from Limited Data using Instructions*](https://arxiv.org/html/2504.05571v1) — closed-book injection succeeding at a comparable corpus scale, via full SFT and five restatements per fact
- [*Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs*](https://arxiv.org/abs/2312.05934) — retrieval outperforming fine-tuning for both existing and new knowledge, independently of this campaign
- [*Data Quality over Capacity: Internalizing Documents into LoRA Adapters for Closed-Book QA*](https://arxiv.org/html/2607.21861v1) — rank-64 capacity saturation, restored at rank 128
- [*How Much is Too Much? Exploring LoRA Rank Trade-offs*](https://aclanthology.org/2025.findings-ijcnlp.58.pdf) — why raising rank without the learning rate regresses
- [*Mitigating Unintended Memorization with LoRA*](https://arxiv.org/pdf/2502.05087) — higher rank increases memorisation without guaranteeing accuracy
- All benchmark figures: [README.md](README.md#worked-example-a-korean-construction-corpus)
