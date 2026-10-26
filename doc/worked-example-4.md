# Worked example-4: does a larger verified training set train a better model?

Generator v0.5.3 raised STaR acceptance from 38.4% to 91.5%
([worked example-2](worked-example-2.md) describes the v0.5.2 state; the
[generator repository](../gen_aec_syn_data) holds the change). The verified
training split grew from 2,202 rows to 12,725. This example asks the question
that growth invites: does the larger, cleaner set train a better model than the
old one — and if it does not, is the problem the quality of the rows or
something about the set as a whole?

The answer turned out to be the second thing, and the diagnosis is the useful
part: a task type that makes up 18% of the verified rows teaches the model to
abstain, the failure is stochastic rather than deterministic, and neither
capping the share nor halving the exposure repairs it. One seed of one arm
would have missed all of this.

## Setup

One recipe, one base model, one answer key. Each arm differs only in its
training file (and where stated, seed or epoch count).

- Base model Qwen3-8B, LoRA r=64 alpha=128, lr 5e-5 cosine, max length 3072,
  effective batch 32 (2 × accumulate 16), 2 epochs unless stated.
- Merged, converted to GGUF and quantized Q4_K_M, served by ollama with the
  same template as the earlier campaigns.
- Scored open and closed book on the v053 key (`config_kr_v053_ft.json`,
  which pins the judge panel), thinking off, 1 sample per item, all ten text
  tracks. Differences are tested pairwise per item (McNemar for binary
  metrics, bootstrap for means); "significant" below means that test, not an
  aggregate gap.

Every arm, data rule and comparison is declared in
[training/experiments/exp_v053cmp.json](../training/experiments/exp_v053cmp.json);
`run_experiment.sh` runs the file and skips finished steps, and
`build_subsets.py` rebuilds the two derived training files bit-exactly from
the rules in it.

| Arm | Training rows | Steps | Final loss | What it tests |
|---|---:|---:|---:|---|
| base | — | — | — | the untrained reference |
| v052 | 2,202 | 136 | 0.51 | the v0.5.2 verified rows, refiltered against the v053 holdout |
| n2202 | 2,202 | 136 | 0.44 | v053 rows subsampled to the v052 count: quality at equal size |
| v053 | 12,725 | 794 | 0.36 | all v053 verified rows: quality and quantity together |
| v053cap | 10,925 | 682 | 0.38 | v053 with refusal rows capped at 5%: the composition remedy |
| v053e1 | 12,725 | 397 | 0.43 | v053 at 1 epoch: the exposure remedy |

v052 and n2202 were each retrained under two further seeds (7 and 42); the
derived sets keep seed 20260814, so the rows are identical and only the
shuffle changes.

## Headline result

Open book, headline metric per track, first seed:

| Track | base | v052 | n2202 | v053 | v053cap | v053e1 |
|---|---:|---:|---:|---:|---:|---:|
| probe | 0.733 | 0.790 | 0.793 | 0.808 | 0.779 | 0.810 |
| track2 sft | 0.795 | 0.820 | 0.808 | 0.830 | 0.795 | 0.823 |
| uc1 safety | 0.785 | 0.800 | 0.797 | 0.828 | 0.814 | 0.804 |
| uc4 faithfulness | 0.931 | 0.956 | 0.944 | **0.725** | **0.675** | 0.825 |
| uc5 incident | 0.425 | 0.637 | 0.453 | **0.326** | **0.314** | 0.408 |
| uc6 verdict v2 | 0.631 | 0.779 | 0.754 | **0.244** | **0.297** | **0.364** |
| uc7 requirement v2 | 0.718 | 0.673 | 0.754 | 0.736 | 0.718 | 0.764 |

Two readings coexist in this table. On the recall-style tracks (probe, sft,
uc1) every trained arm beats the base and the full set is the best of them —
more data helps where the task is to answer in the corpus's terms. On the
judgement tracks (uc4, uc5, uc6) the full set is catastrophic: uc6_v2 falls
from 0.779 (v052) to 0.244, significantly below the untrained base (59 items
gained, 364 lost, p≈0).

## At equal size, the new rows are no better

n2202 holds quantity fixed: 2,202 rows of v053 data against 2,202 rows of
v052 data. The outcome is near parity — probe, sft, uc1 and uc4 are
indistinguishable, uc7_v2 grounding is significantly better (11 gained / 2
lost, p=0.022), and uc5 is significantly worse (mean −0.18). The tripled
STaR acceptance did not translate into measurably better rows; it translated
into more of them. What the acceptance rate actually certifies is that the
reply is consistent with its source chunk, and that property was already
holding up the v052 rows that survived the stricter filter.

## The failure is composition, not quality

The v053-trained model answers uc6_v2 items with "제시된 자료만으로는 확인할
수 없습니다" — it abstains, with the source clause in its context — on 60% of
items, against 2% for the v052 arm. The training sets explain it: 18.4% of
the v053 verified rows are the generator's `refusal` task type (the correct
answer is "cannot be determined from the given material"), against 0.7% in
v052. The generator emits them deliberately, as abstention training; STaR
verifies them easily, since refusing is always consistent with the source;
and SFT on them at this share teaches the model that refusing is what answers
look like.

## Two remedies, neither sufficient

**Capping the share.** v053cap rebuilds the set with refusal rows capped at
5% (546 of 10,925 rows, sampled with the declared seed, source order kept).
It did not repair the collapse: uc6_v2 0.297, abstention still 40%, uc4
*worse* than the uncapped arm (0.675). The fix is significantly better than
v053 on uc6_v2 (73/31, p=5e-5) and significantly worse on probe and sft —
it moved the mixture, not the failure mode.

**Halving the exposure.** v053e1 trains the full set for one epoch (397
steps against 794). Partial repair everywhere — uc4 0.825 (24 gained / 8
lost against v053, p=0.007), uc5 0.408, uc6_v2 0.364, abstention 43% — and
still significantly below the untrained base on uc6_v2 (78/288, p≈0). Less
exposure attenuates the behaviour; it does not remove it.

## The collapse is stochastic: seed repeats

Each small arm was retrained under seeds 7 and 42 (same rows, different
shuffle). v052 is stable: uc6_v2 0.769 ± 0.028 across three seeds,
abstention 2–6% in all of them. n2202 is not:

| n2202 seed | uc6_v2 correct | abstained |
|---|---:|---:|
| 20260814 | 0.754 | 0.082 |
| 7 | 0.673 | 0.176 |
| 42 | 0.185 | 0.843 |

Same 2,202 rows, same 17.8% refusal share as the full set, and one shuffle
in three tips the model into the abstention mode that the full-size arms
reach every time. Read together with the remedies, the picture is a
threshold behaviour: a double-digit refusal share puts the run at risk of
collapse, and total exposure raises the odds from "sometimes" (136 steps) to
"always" (794), while the v052 share of 0.7% never exhibited it. A single
seed of n2202 — the obvious economy — would have reported the new data as
safe at equal size and left the full-set collapse unexplained.

## Closed book: the same behaviour scored from the other side

Closed book, the trained arms move little on recall (0.02–0.08 throughout),
consistent with the earlier campaigns: weights do not become a citation
store at this scale. The one large mover is uc4, where the score *rewards*
declining to invent an answer when no source is supplied: n2202 0.481 and
v053e1 0.469 against base 0.094. The abstention that destroys open-book
judgement is, scored from the other side, honest calibration. The defect is
not that the model learned to abstain; it is that it learned to abstain
regardless of whether the material was in front of it.

## What this example adds

- A verification rate is not a data-quality number. STaR acceptance tripled
  and the rows were no better at equal size; acceptance certifies
  consistency with the source, which the refusal rows satisfy trivially.
- Task composition can dominate everything else about a training set. One
  task type at 18% share cost more open-book accuracy than the whole set
  gained, and the obvious remedies — cap the share, halve the epochs —
  attenuated without repairing.
- The failure mode is stochastic. Arms with the risky composition range from
  healthy to collapsed across nothing but the shuffle seed, so single-seed
  comparisons of training mixtures are unreliable exactly where they are
  most needed.
- The practical rule this run supports: keep explicit refusal rows out of an
  SFT mix, or near the sub-1% share that never misbehaved here, and train
  abstention with some mechanism other than imitation at scale.

## Reproducing it

```
cd training
./run_experiment.sh experiments/exp_v053cmp.json --plan   # what exists, what would run
./run_experiment.sh experiments/exp_v053cmp.json          # trains, registers, scores, compares
```

The experiment file names every arm, seed, epoch count and derived-set rule;
`build_subsets.py` refuses to overwrite a derived file it cannot reproduce
bit-exactly. Run artifacts land in `ground_truth_v053/runs/` as
`v053cmp-<arm>-{open,closed}.json` with paired comparison reports beside
them. The v053 training data downloads with `./download_dataset.sh v053`.
