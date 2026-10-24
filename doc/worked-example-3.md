# Worked example-3: US federal construction safety regulation (29 CFR 1926)

Worked examples [1](worked-example.md) and [2](worked-example-2.md) both run on
Korean construction regulation. This one runs the same method on US federal
safety regulation, in English, and reports what had to change for it to work.

The question it answers is narrow and worth stating plainly: a pipeline that
mines answer keys with rules is written against a language. How much of it is
the method, and how much is Korean? Porting it is the only way to find out, and
the cost of the port is the measurement.

The BIM assets are reused unchanged from the Korean campaigns. No IFC model was
collected or re-rendered, so the image track is disabled here; this example is
about text.

## What this example can and cannot show

It shows that the method transfers: one corpus, one config file, the same build
and scoring code, and a benchmark comes out the other side with the same track
structure and the same measurement controls.

It does not show that the two corpora are comparable as *difficulty*. The item
counts differ by a factor of three, the regulation is organised differently, and
the models evaluated are not the same set. Scores from the two are reported side
by side only where the comparison is about the instrument — whether a key defect
reproduces, whether a control behaves the same way — never as a claim that
English regulation is easier or harder than Korean.

## The corpus

29 CFR Part 1926, the US Occupational Safety and Health Administration's
construction standards, collected from the eCFR API by
[`scripts/collect_ecfr.py`](../scripts/collect_ecfr.py).

| | |
|---|---|
| Sections | 236 |
| Chunks | 12,540 |
| Held-out documents | 101 |
| Format | plain text, one file per section |

The collector needs `Accept-Encoding` set for compression or the API returns
truncated bodies, and it folds the source credit into the opening line rather
than emitting it as a line of its own. That second detail is not cosmetic — see
the routing section.

## What had to become configuration

Nine behaviours in the generator and the benchmark were written against Korean.
Each one was found by a build producing zero of something, and each is now a
config key with the Korean behaviour as its default, so
[`config_kr.json`](../benchmark/config_kr.json) carries none of these keys and
behaves as before while [`config_us.json`](../benchmark/config_us.json) sets
them.

| What broke | Symptom | Setting |
|---|---|---|
| Input discovery was `.pdf` only | no documents found | `input_extensions` |
| "Is this chunk informative" counted Hangul | 6 of 6 chunks dropped | `informative_script` |
| Source credit emitted as its own line | that line scored as non-informative, so every amended section routed to `retrieve`: 92 % of documents, against 57 % once fixed | collector change |
| Mining vocabulary was Korean units and qualifiers | 0 numeric items from 4,479 chunks | `mining.units`, `mining.qualifiers`, `mining.qualifier_leads`, `mining.verdict_qualifiers` |
| Enumeration patterns were Korean | uc5 built 0 items | `enumeration.*` |
| Enumerations assumed they start at 1 | items rejected | `enumeration.require_start_at_one` |
| Item text, clause labels and the abstain token were Korean | Korean strings inside English items | `mining.item_lang`, `clause_format`, `abstain_token` |
| Subject extraction assumed Korean word order | malformed subjects | `subject.direction`, subject rules |
| Chunk overlap lost sentence alignment | 81 % of chunks began mid-sentence | `chunk_boundary` |

Two further settings were not language problems but became configuration in the
same pass, because the English run is where they first mattered: `eval.models`
(a field that existed and was documented but that nothing read), and
`eval.headline_metrics` (see the scoring defects section).

The `--config` flag also had to change meaning. It discarded command-line
overrides, so an English run with a Korean config would have generated from the
Korean input directory; it is now a base that the command line overrides.

## Routing

| Route | Documents | Share |
|---|---:|---:|
| train | 135 | 57 % |
| both | 19 | 8 % |
| retrieve | 82 | 35 % |

The Korean corpus routes very differently: 235 train, 609 both, 1,103 retrieve
over 1,947 documents, or 12 / 31 / 57 %. English sends 57 % to training where
Korean sends 12 %, and 35 % to retrieval where Korean sends 57 %.

The router scores amendment exposure, so this says the Korean documents look far
more amendment-exposed than the CFR sections do. The reason is structural: KCS
and KDS documents carry revision marks and dated notes through the body text,
while a CFR section concentrates its amendment history in one credit line at the
end. The same rule, applied to two corpora, splits them differently because the
corpora record their own history differently — which is a property of the
regulation, not a defect in the port.

The 35 % is also what the collector change bought. Before the source credit was
folded into the opening line, 92 % of documents routed to `retrieve`.

A single line of provenance text, classified as non-informative, had been enough
to make every amended section look volatile.

## What the generator produced

Generation ran 1 h 51 min over 236 documents on one GB10.

| Dataset | Rows |
|---|---:|
| DAPT | 1,438 |
| SFT | 3,175 |
| STaR accepted | 3,175 |
| STaR rejected | 1,736 |
| DPO pairs | 1,020 |
| RLVR | 3,175 |
| RAG corpus | 3,622 |

STaR accepted 3,175 of 4,911 candidates, 64.7 %. The useful part of that number
is that it is neither 0 nor 100: the verification filter is discriminating on
English text, not passing or rejecting everything in a language it was not
written for. The Korean acceptance rate cannot be compared here — that build's
intermediate output is not on this machine.

## The benchmark

916 items across eight tracks, plus a 340-item repaired variant of uc6.

| Track | Items |
|---|---:|
| track2 (SFT) | 93 |
| probe | 213 |
| uc1_safety | 111 |
| uc2_rebar_spec | 21 |
| uc4_faithfulness | 85 |
| uc5_incident | 26 |
| uc6_verdict | 340 |
| uc7_requirement | 27 |
| uc6_verdict_v2 (repair) | 340 |

uc3 is disabled: no IFC was collected for this corpus.

## Results

Three models, open book and closed book, scored with a judge panel that
excludes the evaluated model's own family.

| Track | open 8b | open 14b | open gemma | closed 8b | closed 14b | closed gemma |
|---|---:|---:|---:|---:|---:|---:|
| track2_sft | 0.723 | 0.794 | 0.633 | 0.059 | 0.077 | 0.059 |
| probe | 0.684 | 0.787 | 0.687 | 0.075 | 0.086 | 0.082 |
| uc1_safety | 0.688 | 0.799 | 0.657 | 0.059 | 0.062 | 0.057 |
| uc2_rebar_spec | 0.857 | 0.857 | 0.905 | 0.000 | 0.095 | 0.000 |
| uc4_faithfulness | 0.835 | 0.871 | 0.859 | 0.224 | 0.435 | 0.482 |
| uc5_incident | 0.473 | 0.586 | 0.500 | 0.000 | 0.011 | 0.015 |
| uc6_verdict | 0.506 | 0.462 | 0.465 | 0.029 | 0.018 | 0.076 |
| uc6_verdict_v2 | 0.579 | 0.503 | 0.588 | 0.438 | 0.265 | 0.026 |
| uc7_requirement | 0.667 | 0.778 | 0.815 | 0.074 | 0.037 | 0.000 |

Models are `qwen3:8b`, `qwen3:14b` and `gemma2:9b`.

The open-to-closed gap is the first thing to read: uc1 falls from 0.688 to
0.059. The items ask for a figure stated in a specific CFR paragraph, and
without the paragraph in the prompt a model has nothing to recall. That is the
intended behaviour of the control — it says the open-book scores are measuring
use of supplied evidence, not memorised regulation — and it is the same shape as
the Korean result (0.8029 to 0.0765 on the same track).

## The uc6 key defect reproduces in English

uc6 asks whether a stated figure complies with a clause. Worked example 2 found
the Korean version of this track was measuring abstention rather than judgement,
and repaired the key as `uc6_verdict_v2`. The English build inherits the repair,
and the English runs reproduce the finding.

| Model | uc6_verdict | uc6_verdict_v2 | Change |
|---|---:|---:|---:|
| qwen3:8b | 0.5059 | 0.5794 | +0.0735 |
| qwen3:14b | 0.4618 | 0.5029 | +0.0411 |
| gemma2:9b | 0.4647 | 0.5882 | +0.1235 |

All three originals sit at or below the 0.500 a constant classifier scores on
this track, and all three move above it after the repair. The Korean campaign
measured the same pattern across six models, and the effect size there was inversely
correlated with the original score. Nothing about the defect was Korean: it was
a property of how the key was mined, and the mining rule crossed languages
intact.

One English-specific defect had to be fixed before this track scored at all: the
verdict instruction did not name the vocabulary it expected. Models answered
`Non-compliant` against a key of `contradict`, and every item scored 0.0. The
instruction now states the words.

## The remaining mining defect, measured

Subject extraction for English produces a malformed subject in 28 % of items —
a fragment such as `In areas` or `means an approved closed container, of` where
a complete noun phrase was intended. Rather than report this as an unquantified
caveat, [`tools_subject_split.py`](../benchmark/tools_subject_split.py) splits
every run by subject quality.

| Condition | Clean subject | Malformed subject | Difference | p |
|---|---:|---:|---:|---:|
| open, pooled over 3 models | 0.8876 (n=1,005) | 0.8799 (n=408) | −0.008 | 0.682 |
| closed, pooled over 3 models | 0.1771 | 0.1642 | −0.013 | 0.562 |

Per model the sign is not stable — open book gives +0.031, −0.050 and −0.004 —
so the defect does not systematically cost accuracy in either condition. The
28 % figure is a real limit on item presentation quality and it is reported as
one, but it does not qualify the scores above.

This is worth stating as a method point. The first single-model measurement
showed malformed items scoring *higher*, which invited an explanation; the
third model removed the effect entirely. A subgroup gap of 0.03 on n=136 is
within noise, and a decomposition is only worth acting on once it is stable
across models.

## Two scoring defects the English run exposed

Neither is specific to English. Both were in the summary layer, where a metric
a grader does not emit was being treated as a zero or dropped.

**The headline read a hardcoded metric list.** The per-track summary took
`correct`, then `f1`, then `key_f1`, defaulting to 0.0. The sentence grader
emits `grounded`, so uc7 was recorded as 0.0 in every run while its actual
scores were 0.667 to 0.815. Six Korean run files carried the same zero,
including the `v052-open` baseline, whose uc7 is 0.7109.
`eval.headline_metrics` now names the preference order, a track whose graders
emit none of them is omitted rather than scored zero, and `meta.headline_metrics`
records which metric each track was summarised by.

**The paired comparison read its own copy of the same list.** `compare.py`
matched items by id across two runs and chose a metric from
`("correct", "f1", "key_f1")`, so uc7 was silently excluded from every
significance test. Both call sites now read `common.score_metrics(cfg)`.

Repairing the first did not require rescoring: the per-item scores were already
stored, so [`tools_rehead.py`](../benchmark/tools_rehead.py) recomputes the
summary from them. It rewrites only entries recorded as 0.0 and adds or removes
no keys, because `rag` runs name their tracks differently and a wholesale
recompute renames them.

## Known defect: routing created a third provenance class

`cb.py verify` fails one check on this build: `in_train`, 34 of 57.

Twenty-three use-case items were mined from documents the router sent to
`retrieve`. Those documents emit no training data by design, but the benchmark's
split has two values — `holdout` and `train` (contaminated) — so the items were
labelled `train` and the check that they appear in training data fails. Document
routing created a third provenance class that a two-way split cannot represent.

Measured impact:

| Model | train subgroup as built | excluding the 23 | the 23 alone |
|---|---:|---:|---:|
| qwen3:8b | 0.8300 (n=270) | 0.8375 (n=247) | 0.7503 |
| qwen3:14b | 0.8455 | 0.8496 | 0.8007 |
| gemma2:9b | 0.8183 | 0.8222 | 0.7764 |

23 of 916 items, 8.5 % of the `train` subgroup. Headline and whole-track scores
pool every item and are unaffected; only the `by_split` rows move, by about
0.004. The 23 are consistently harder, which is the expected direction for items
whose source document was never in training data.

The fix is a third split value, and it requires a rebuild, which would
invalidate the six runs reported here. It is deferred rather than applied. The
Korean build cannot be re-verified on this machine — its `provenance.jsonl`
covers 394 track2 rows from an earlier run and the source corpus is not local —
so whether the Korean build carries the same 2.5 % is open.

## Against the Korean campaigns

| | Korean (example 2) | English (example 3) |
|---|---|---|
| Corpus | KCS/KDS and related, 1,947 documents | 29 CFR 1926, 236 sections |
| Chunks | 85,520 | 12,540 |
| Routing train / both / retrieve | 12 / 31 / 57 % | 57 / 8 / 35 % |
| Generation time | 12 h 15 min | 1 h 51 min |
| Benchmark items | 2,317 | 916 |
| Models scored | 6 | 3 |
| uc1 open to closed | 0.8029 to 0.0765 | 0.688 to 0.059 |
| uc6 repair effect | positive, 6 of 6 models | positive, 3 of 3 models |
| Code changes to port | — | 0 forks, 9 config axes |

The last row is the result this example exists for. No code path is selected by
language; the Korean behaviour is the default of each new setting, and the
English build is a config file.

## Limits

- Three models, all open-weight, 8B to 9B. No frontier model and no commercial
  reference point, so saturation is untested.
- Every answer key was mined by rule or model. No human expert has labelled a
  sample of this build, so the keys are validated by model agreement and by the
  controls above, not by domain review.
- 28 % of items carry a malformed subject. Measured as not affecting accuracy
  (above), but it is a presentation defect in the published items.
- 23 items are labelled with the wrong provenance class (above).
- uc2 has 21 items and uc5 has 26. Wilson intervals on those tracks are wide and
  the per-model differences there should not be read as model differences.
- The image track is disabled. Nothing here measures the cross-modal case the
  Korean campaigns cover.
- No fine-tuning was run on the English synthetic data, so this example reports
  what the benchmark measures, not whether the generated data trains a better
  model.
