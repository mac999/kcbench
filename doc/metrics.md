# Metric reference — definitions and sources

Every metric [kcbench](../README.md) reports: what it is, how it is graded, and
where the definition comes from.

A run produces numbers in four layers: how an answer is graded, whether the
score can be trusted, what the score cannot see, and the before/after comparison
that is the actual result. Each layer is reported separately — nothing here is
averaged into a single figure.

## 1. Answer types and grading rules

Every item declares an `eval_type`, and that decides how its reply is graded.
The types are not comparable with each other, so a track carrying two of them
reports two numbers rather than their mean.

| Type | Score | Range | Read as |
|---|---|---|---|
| `numeric` | accuracy | 0–1 | share of stated thresholds recalled |
| `nameset` | precision / recall / **F1** | 0–1 | how much of a list was recovered |
| `label` | accuracy | 0–1 | share of classifications correct |
| `faithfulness` | accuracy + **abstention rate** | 0–1 | does it refuse when unsupported |
| `verdict` | accuracy, **grounded** | 0–1 | does it judge compliance off the right clause |

Where a reply carries a JSON object naming `answer`, `answerable`, `verdict`,
`value` or `evidence`, the field is graded rather than the whole string. That is
what `eval.structured_answer` switches, and it is on by default: grading the raw
text reads whichever number comes first, which on a reasoning model is a figure
quoted inside the thought rather than the answer — the mechanism behind the
0.284 that reran at 0.950 under [Item validity](retrieval-and-validity.md#item-validity--what-is-measurable-without-expert-review).
A reply with no object is passed through untouched, so models that answer in
prose score exactly as they did before.
| `mapping` | key F1 + value accuracy | 0–1 | named the right things, counted them right |
| perplexity | perplexity | 1–∞, **lower is better** | fit to unseen text |

Three more numbers are not grading types — they are asked of the model about
its own answers, and defined in section 3 below:

| Metric | Score | Range | Read as |
|---|---|---|---|
| **ECE** (`cb.py ece`) | calibration error | 0–1, **lower is better** | does it know when it is right |
| Brier (beside ECE) | squared error of confidence | 0–1, lower is better | accuracy and calibration in one number |
| inconsistency (`cb.py selfcheck`) | 1 − sampling agreement | 0–1 | does it tell the same story twice, no answer key needed |

**`numeric` — the first number in the reply, within 2% relative tolerance.**
*"What is the minimum thickness?"* → `40mm`. An item is right or wrong and the
track reports the share right. This is the bulk of the benchmark: 320 of the 395
`sft` items and 320 of the 400 probe items.

**`nameset` — set F1, so a partial answer scores partially.** *"List the items
in paragraph 3"* has four right answers, and a model naming three of them has
not failed. Precision is the share of what the model said that was right; recall
is the share of the answer key it found; F1 is their harmonic mean and is the
number reported. Both halves are needed — precision alone rewards a model that
offers only its safest guess, recall alone rewards one that lists everything it
can think of, and F1 requires both.

Two matching modes, and they are not interchangeable. **Exact** matching
compares normalised strings, which is what `sft`, `probe`, `vlm` and `uc1` use.
**Fuzzy** matching counts a predicted line as a hit when it covers 60% of a gold
item's content words, which `uc5` uses because its answers are clause-length
prose a model will legitimately abbreviate or renumber. An F1 from one is not an
F1 from the other; the tracks say which they use.

**`label` — the first vocabulary word in the reply wins.** `uc3` puts a BIM
render and a site photograph side by side and asks for `match`, `partial_match`
or `mismatch`. Position rather than membership, because the vocabulary overlaps
itself — `partial_match` contains `match` — and a reply naming several labels
has to be read as its first commitment.

**`faithfulness` — abstention where abstention is the right answer.** Half the
items in `uc4` carry a swapped, unrelated passage; the other half carry the
genuine one.

| Item | Correct behaviour |
|---|---|
| passage swapped | abstain — say the passage does not support an answer |
| passage genuine | answer, and answer correctly |

The halves are equal in number on purpose: a model that abstained on everything
would score 100% on the swapped half alone. The abstention rate is reported next
to accuracy for the same reason, so that behaviour is visible rather than
inferred from a single figure.

**`verdict` — the judgement and the clause it rests on, scored apart.** A
threshold item states a limit; pairing it with a measured figure makes a
compliance question whose answer follows from the qualifier, so the key is
derived rather than written and `build_verdict.py` mines the track the same way
the rest of the benchmark is mined. Items come in compliant and violating pairs,
so answering the same way throughout scores 0.5.

| Column | Reads as |
|---|---|
| `correct` | the verdict matched |
| `evidence_hit` | it cited the clause the limit came from |
| `grounded` | **both** — the judgement and its support |
| `abstained` | it declined to judge |

`grounded` is the one to report. A model that reaches the right verdict off the
wrong clause has not read the regulation, and on a two-way judgement it gets
half of them right by guessing; the split is what separates the two. Replies
that name the verdict in prose are graded on `correct` alone, since an evidence
list needs the structured field.

**`mapping` — keys and values scored apart.** *"How many of each element
type?"* is two questions: did it name the right types (key F1), and did it count
them (value accuracy). A model that lists the catalogue correctly and guesses
every count is a different failure from one that misses half the types, and one
combined number would hide which happened.

**perplexity — the only metric here where lower is better.** No question is
asked: held-out text is run through the weights and the score is how surprised
the model was by it. The measure is
[Jelinek, Mercer, Bahl and Baker's (1977)](https://doi.org/10.1121/1.2016299),
proposed to say how hard a speech recognition task is and still the standard way
to state how well a language model fits a text. It says the model has grown
familiar with the prose. It does not say the model can answer a question about
it, and the difference is the reason the other five types exist — in the worked
example above, perplexity fell 40% while closed-book recall did not move at
all.

## 2. Reliability checks

**`no_answer` — the share of replies containing no answer at all.** A wrong
answer and a missing answer both score zero and mean opposite things. An
inference server that has gone away, an exhausted token budget, or a reasoning
model that emits an empty think block and stops all produce the second. This
field is what caught the worked example's serving bug: 2.9% correct looked like
a destroyed model until `no_answer` showed that 43% of the replies were empty.

**`*_ci95` — a 95%
[Wilson (1927)](https://doi.org/10.1080/01621459.1927.10502953) interval on
every proportion.** A 39-item track and a 320-item track can print the same
`0.15` and mean very different things by it, and the interval puts that
difference on the page instead of leaving it to be remembered. Wilson's
construction rather than the textbook normal one because these tracks sit in
exactly the regime — small n, p near 0 — where the normal interval puts the
lower bound below zero.

## 3. Calibration and consistency

**Expected Calibration Error — `cb.py ece`.** Not *is it right* but *does it
know when it is right*, because the dangerous failure is being wrong and sure.
A model that states no probability has to be asked more than once, so the same
question goes in eight times at temperature 0.7 and the modal answer's share
becomes its confidence — self-consistency, after
[Wang et al. (2022)](https://arxiv.org/abs/2203.11171). ECE is then the average
gap between that confidence and actual accuracy, taken over confidence bins:
the binned estimator of
[Naeini, Cooper and Hauskrecht (2015)](https://ojs.aaai.org/index.php/AAAI/article/view/9602),
in the form [Guo et al. (2017)](https://arxiv.org/abs/1706.04599) made standard
for neural networks. Lower is better and 0 is perfect. The
[Brier score (1950)](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml),
reported beside it, asks the same question without the bins and comes from
weather forecasting, where being confidently wrong has always been the
expensive failure. Covers `numeric` and `label` items only.

**Inconsistency — `cb.py selfcheck`.** A hallucination signal that needs no
answer key, so it also works on free-form answers no track can grade. Sample the
same question several times: a fact the weights hold comes back the same way, an
invented one drifts. That is
[SelfCheckGPT (Manakul, Liusie and Gales, EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.557/),
reduced here to the one comparison an extractive answer allows. It reports
`separation` as its own validation — how much higher the inconsistency runs on
answers that were in fact wrong. On the worked example that came out at 0.026,
meaning the detector does not work on this model, which is a result worth having
and is why the number is printed rather than the flag rate alone.

## 4. Significance testing

Everything above describes one checkpoint. `cb.py compare` is what the benchmark
exists to produce.

**Delta** is the subtraction, and on its own it is not evidence.

**[McNemar's exact test](https://doi.org/10.1007/BF02295996)**, for binary
metrics. McNemar wrote it in 1947 for correlated proportions, which is exactly
what two runs over one frozen item set produce: the runs answered the *same*
questions, so they are paired, and only the items whose verdict changed carry
information about the change. Reported as gained, lost and a p-value:

> 320 probe items, 26 gained, 24 lost, p = 0.89

A large p means the data cannot distinguish the change from noise. Read as an
unpaired difference of two aggregates, those same numbers read as a 0.6-point
improvement — which is the mistake the paired test exists to prevent.

**Paired bootstrap**, for continuous metrics such as F1, reporting a 95%
interval on the mean per-item change. The resampling procedure is
[Koehn's (2004)](https://aclanthology.org/W04-3250/), introduced for this
problem exactly — deciding whether one system really beats another on a test
set too small for a raw difference to be trusted:

| Interval | Reading |
|---|---|
| `[+0.03, +0.17]` | real improvement |
| `[−0.29, −0.12]` | real regression |
| `[−0.05, +0.08]` | indistinguishable from no change |

An interval straddling zero means the data does not separate this change from
nothing, whatever the point estimate says.

## The two axes of every score

No score above means anything without both of them stated.

**Closed book or open book.** Open book supplies the passage, so the item tests
reading comprehension; closed book withholds it, so the item tests what the
weights hold. Both are scored against the same answer key. Fine-tuning has
almost no room to move an open-book score — the answer is on the page — which is
why the closed-book number is the one this benchmark reports, and why the
open-book number is the one that matters for a system that will retrieve the
clause anyway.

**Probe or holdout.** The same kind of question mined from opposite sides of the
split: probe from documents the model trained on, holdout from documents
withheld from it. They are read as a pair, never singly — the table under
[Design: holdout and probe](../README.md#evaluation-design--held-out-and-contaminated-probe-sets) says what each
combination means, and step 6 of the [Workflow](../README.md#workflow) reads it against a
real run.

## Provenance of the metrics

Nothing above is this project's invention, which is the point: a reviewer should
be able to recognise what is being computed without reading the code.

| Metric | Source |
|---|---|
| perplexity | Jelinek, Mercer, Bahl & Baker, [*Perplexity — a measure of the difficulty of speech recognition tasks*](https://doi.org/10.1121/1.2016299), JASA 62 (1977) |
| Wilson interval | E. B. Wilson, [*Probable Inference, the Law of Succession, and Statistical Inference*](https://doi.org/10.1080/01621459.1927.10502953), JASA 22 (1927) |
| Expected Calibration Error | Naeini, Cooper & Hauskrecht, [*Obtaining Well Calibrated Probabilities Using Bayesian Binning*](https://ojs.aaai.org/index.php/AAAI/article/view/9602), AAAI (2015); Guo, Pleiss, Sun & Weinberger, [*On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599), ICML (2017) |
| confidence by self-consistency | Wang et al., [*Self-Consistency Improves Chain of Thought Reasoning in Language Models*](https://arxiv.org/abs/2203.11171) (2022) |
| Brier score | G. W. Brier, [*Verification of Forecasts Expressed in Terms of Probability*](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml), Monthly Weather Review 78 (1950) |
| inconsistency / selfcheck | Manakul, Liusie & Gales, [*SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*](https://aclanthology.org/2023.emnlp-main.557/), EMNLP (2023) |
| McNemar's test | Q. McNemar, [*Note on the sampling error of the difference between correlated proportions or percentages*](https://doi.org/10.1007/BF02295996), Psychometrika 12 (1947) |
| paired bootstrap | P. Koehn, [*Statistical Significance Tests for Machine Translation Evaluation*](https://aclanthology.org/W04-3250/), EMNLP (2004) |

The grading types themselves — which published benchmark each one follows, and
why — are tabulated separately in
[benchmark/README.md](../benchmark/README.md#precedent-for-each-grading-type).

