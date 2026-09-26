# Retrieval ablation and item validity

Two analyses from the [worked example](worked-example.md): how much of the
score is decided by the retriever rather than the model, and how much of the
item set is measurable without an expert reviewer.

## Retrieval ablation — accuracy against retrieval quality

Closed and open book are the ends of a scale, not the whole of it. Closed book
is a model answering from memory; open book hands it the exact clause the item
was mined from, which is what a *perfect* retriever would do. A deployed system
sits between them, and `cb.py rag` scores that middle: the held-out chunks are
embedded, searched per question, and the top-k concatenated as the passage in
place of the gold clause. Everything downstream is identical to an open-book
run, so the numbers are directly comparable.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="rag-dark.png">
  <img alt="Numeric accuracy for the untrained base model under six retrieval conditions: closed book 0.147, RAG with an English-centred embedder 0.144, RAG bge-m3 top-3 0.397, bge-m3 top-10 0.481, arctic-embed2 top-10 0.478, open book 0.944." src="rag-light.png">
</picture>

| Condition | recall@k | numeric | nameset F1 |
|---|---:|---:|---:|
| closed book, no retrieval | — | 0.147 | 0.000 |
| RAG, `nomic-embed-text`, top-10 | 0.041 | 0.144 | 0.068 |
| RAG, `bge-m3`, top-3 | 0.240 | 0.397 | 0.236 |
| RAG, `bge-m3`, top-10 | 0.400 | 0.481 | 0.318 |
| RAG, `snowflake-arctic-embed2`, top-10 | 0.428 | 0.478 | 0.319 |
| open book, perfect retrieval | 1.000 | 0.944 | 0.587 |

`recall@k` is the share of items whose source chunk the retriever actually
returned, so it separates the two ways a RAG answer goes wrong. At top-10 the
retriever finds the right chunk 40% of the time and the system answers 48% of
items — the model is doing slightly better than the retriever, because some
questions are answerable from a neighbouring clause. The gap between 0.481 and
0.944 is not a model deficiency; it is retrieval headroom.

**This is the comparison a fine-tuning decision should be made against.** On
this corpus, moving from a mediocre retriever to a good one is worth roughly
0.35 accuracy, while four rounds of fine-tuning moved closed-book recall by
nothing at all. Retrieval quality dominates, and a project with a fixed budget
should spend it there first.

**The embedder decides whether any of this works.** Four embedding models were
swept over the same 5,381 chunks and the same 395 questions, scoring retrieval
alone:

| Embedder | r@1 | r@3 | r@10 | r@20 |
|---|---:|---:|---:|---:|
| `snowflake-arctic-embed2` (multilingual) | 0.170 | 0.306 | 0.428 | 0.496 |
| `bge-m3` (multilingual) | 0.134 | 0.241 | 0.400 | 0.468 |
| `nomic-embed-text` (English-centred) | 0.010 | 0.023 | 0.041 | 0.061 |
| `mxbai-embed-large` (English-centred) | 0.005 | 0.018 | 0.028 | 0.035 |

The split is not a matter of degree. The two multilingual models retrieve the
right chunk 10 to 15 times more often than the two English-centred ones, and
end to end the difference is total: **RAG with `nomic-embed-text` scores 0.144,
which is what the model scores with no retrieval at all (0.147).** Attaching a
retriever that cannot search the language buys exactly nothing, and it fails
silently — the pipeline returns passages, the model answers, and only the score
shows that the passages were unrelated. Verify the embedder against a held-out
set before trusting anything downstream of it.

Between the two working embedders the difference is small (0.481 against 0.478
at top-10), so the choice within a competent family matters far less than the
choice of family.

## Item validity — what is measurable without expert review

The items are mined by rule and verified against their source, not written by
domain experts, and no human has reviewed them. That is the benchmark's largest
methodological limitation and it is stated plainly in [Limits](../README.md#limits). Four
things are nevertheless measurable about item quality, and they are reported
here rather than left to be assumed.

**Admission is strict, and what it rejects is counted.** Mining produced 5,244
candidate items for the `sft` track and admitted 405 — a 7.7% acceptance rate.
Every rejection is attributed to a rule, in `data/rejections.json`:

| Why an item was rejected | Count |
|---|---:|
| no usable subject before the number | 1,966 |
| enumeration markers are not a run starting at 1 | 1,073 |
| enumeration has no lead-in sentence saying what it lists | 582 |
| subject too short or too long | 321 |
| a threshold with an overlapping subject shares this unit and qualifier | 192 |
| duplicate of an item already accepted | 167 |
| subject has an unbalanced bracket | 163 |
| ten further rules | 373 |

Most of those rules exist because an earlier build produced an item that turned
out to be unanswerable, so the filter is a record of failures already found.

**Every item traces to its source, mechanically.** `cb.py verify` re-checks four
things per item: the generated file it names is present, the row at that index
still hashes to the recorded digest, the document is on the reserved list, and
no training row carries that text by digest. So "the answer is in the document"
is not a claim requiring trust — what a human reviewer would still add is
whether the *question* reads naturally, which is a narrower job than validating
the set from scratch.

**The open/closed gap is evidence the items are answerable.** Six models that
never trained on this corpus, on the same items:

| Model | Open book | Closed book |
|---|---:|---:|
| qwen3:30b-a3b | 0.932 | 0.144 |
| qwen3:14b | 0.909 | 0.129 |
| qwen3:8b | 0.947 | 0.151 |
| glm4:9b | 0.909 | 0.099 |
| qwen2.5-coder:7b | 0.939 | 0.114 |
| llama3.2:latest | 0.849 | 0.114 |

A broken item is not answerable with the passage in front of you. Every model
clears 84% open book, so the items are solvable and what is hard about them is
the recall, which is what the benchmark is for. The consistency across six
architectures also rules out the gap being one model's quirk.

**The two sets are not differently biased.** Before any training, `probe` scores
0.169 and the held-out `sft` track 0.166. They are mined by the same rules from
different documents, and an untrained model should find them equally unfamiliar
— it does. Divergence after training is therefore attributable to training
rather than to the sets having been built differently.

**Three models were asked to check the answer keys.** Expert review is the right
instrument and it has not been done; this is the cheaper approximation. Each of
qwen3:14b, glm4:9b and qwen3:30b-a3b answered all 395 `sft` items **open book**,
with the clause supplied. An item all three get wrong with the passage in front
of them is more likely to be a bad item than three independent failures, so
their unanimous-wrong rate is an upper bound on the key error rate:

| Answer type | Items | All three wrong | Upper bound on key errors |
|---|---:|---:|---|
| `numeric` | 320 | 5 | **1.6%** [0.7%, 3.6%] |
| `nameset` | 75 | 18 | **24.0%** [15.8%, 34.8%] |

It is an upper bound, not an estimate: a genuinely hard item counts against it
too. The three models agree closely on the track overall — numeric 0.944, 0.925,
0.950 — so the disagreements are concentrated, not diffuse.

The gap between the two types is itself a finding. Numeric keys are a single
figure with a 2% tolerance, and 1.6% is a defensible error rate. Nameset keys
are a list, and inspecting the flagged items shows the models enumerating
*different but defensible* sets from the same clause: where a list begins and
ends in Korean regulation prose is often not unique, and the mined key picks one
reading. **`nameset` results should therefore be read as weaker evidence than
`numeric` ones throughout this benchmark**, and the enumeration findings in the
worked example carry that caveat.

One methodological note, since it nearly corrupted this check: qwen3:30b-a3b
scored 0.284 on the first pass because `--think off` does not suppress reasoning
on that model — its chain of thought landed in the answer field and graded as
wrong. Rerun with the flag omitted it scores 0.950, in line with the others. A
cross-check is only as good as the serving configuration underneath it.

