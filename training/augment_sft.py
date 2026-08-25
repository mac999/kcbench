#!/usr/bin/env python3
"""
Augment the SFT pairs toward closed-book recall and complete enumeration.

    python augment_sft.py                             # defaults from the flags below
    python augment_sft.py --closed-ratio 0.5 --enum-max 3000
    python augment_sft.py --no-enum                   # closed-book mixture only

Why this exists: the generated pairs are open-book by construction -- 95% carry
the source clause in the prompt -- so training on them teaches extraction, not
recall, and the benchmark showed exactly that (closed-book flat, p~0.9). The
second measured turn of the loop refined the recipe: bare closed-book variants
injected nothing and damaged abstention (0.925 -> 0.750), while enumeration
pairs recovered half the nameset regression. The transformations below encode
what the literature prescribes for each finding:

  paraphrases      a fact stated one way is stored but not extractable; restate
                   it several ways and extraction follows (Allen-Zhu & Li,
                   "Physics of Language Models" 3.1; Ovadia et al. on
                   fine-tuning vs RAG). Questions are rephrased by a local
                   model, answers stay fixed, and rewrites that leak the
                   answer, collapse to the original, or lose the subject are
                   filtered (Self-Instruct's dedupe-and-filter discipline).
  abstention pairs closed-book training teaches answering without support
                   unless refusal is trained beside it (R-Tuning, NAACL 2024).
                   A share of pairs get their clause swapped for an unrelated
                   one, verified not to contain the answer, with a refusal as
                   the target -- phrasing rotated so the model learns the
                   behaviour, not a string.
  template rotation one instruction format overfits to that format (FLAN/T0);
                   closed-book variants rotate through several.

All transformations, ratios and caps are flags:

  closed-book variants   a share of pairs duplicated with the clause removed,
                         so the same question is also asked from memory. The
                         open-book originals stay in, because extraction is a
                         skill worth keeping.
  enumeration pairs      clauses in the DAPT chunks that enumerate items become
                         "list them all" pairs whose output is the full list,
                         countering the short-single-answer style that damaged
                         nameset F1 (median output was 56 chars, 18% list-form).

Contamination: inputs are the already-split training files, which exclude the
held-out documents by digest, so everything derived here is clean by
construction. Nothing in this script reads the holdout or the evaluation sets.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent / "benchmark" / "data" / "train"

CB_TEMPLATES = [
    "{q}",
    "{q} 기억나는 대로 답하시오.",
    "조문을 보지 않고 답하시오. {q}",
    "{q} 관련 규정에 근거해 답하시오.",
]
REFUSALS = [
    "제시된 조문에서는 확인할 수 없습니다.",
    "주어진 조문으로는 답할 수 없습니다.",
    "해당 내용은 제시된 자료 없음.",
    "제시된 조문에서 관련 근거를 확인할 수 없습니다.",
]
WORD_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")

ITEM_RE = re.compile(r"^\s*(?:(\d{1,2})[.)]|([가나다라마바사아자차])[.)]|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮])\s+(.+)$")
LEAD_RE = re.compile(r"(다음\s*각\s*호|다음과\s*같다|다음\s*사항|아래와\s*같다|다음\s*기준|다음\s*조건|다음\s*서류|다음\s*항목|다음\s*내용|다음\s*각\s*목)")


def load(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def closed_variants(rows, ratio: float, rng: random.Random):
    """Duplicate a share of pairs with the clause stripped from the input."""
    out = []
    for r in rows:
        inp = r.get("input")
        ctx = (inp or {}).get("context", "") if isinstance(inp, dict) else ""
        if len(ctx.strip()) < 30 or rng.random() > ratio:
            continue
        v = json.loads(json.dumps(r, ensure_ascii=False))
        v["id"] = f"{r.get('id', 'sft')}_cb"
        v["instruction"] = rng.choice(CB_TEMPLATES).format(q=r["instruction"].strip())
        v["input"] = {k: val for k, val in (inp or {}).items() if k != "context"}
        v["input"]["context"] = ""
        v["task_type"] = "regulation_recall"
        out.append(v)
    return out


def content_words(text: str) -> set:
    return set(WORD_RE.findall(text))


def abstain_pairs(rows, ratio: float, rng: random.Random, hard: bool = False):
    """
    Swap a pair's clause for one that does not answer it, and make refusal the
    target. The swapped clause is checked not to contain the answer, so refusing
    is the only right response, and the refusal phrasing rotates -- the point is
    the behaviour, and a single fixed string would train a password instead.

    `hard` decides where the swapped clause comes from, and the measured turns
    say it decides whether the training transfers. Easy negatives are drawn from
    unrelated documents and are recognisable as unrelated from the vocabulary
    alone; a model trained on those learned nothing that held up on
    uc4_faithfulness, which swaps in plausible clauses from the same document
    (abstention stayed at 0.750). Hard negatives prefer a different clause of
    the same document, so the model has to read rather than pattern-match the
    subject matter.
    """
    pool = [r for r in rows
            if isinstance(r.get("input"), dict)
            and len((r["input"].get("context") or "")) > 60]
    by_doc = {}
    for r in pool:
        by_doc.setdefault(r.get("doc_id"), []).append(r)
    out = []
    for r in pool:
        if rng.random() > ratio:
            continue
        o = r.get("output")
        ans = (o.get("answer", "") if isinstance(o, dict) else str(o)).strip()
        # hard: same document first, falling back to the global pool when a
        # document holds only this one pair
        siblings = [x for x in by_doc.get(r.get("doc_id"), []) if x is not r]
        candidates = (siblings or pool) if hard else pool
        for _ in range(8):
            donor = rng.choice(candidates)
            ctx = donor["input"]["context"]
            if donor is r or (ans and ans in ctx):
                continue
            if not hard and donor.get("doc_id") == r.get("doc_id"):
                continue
            v = json.loads(json.dumps(r, ensure_ascii=False))
            v["id"] = f"{r.get('id', 'sft')}_ab"
            v["input"]["context"] = ctx
            v["output"] = {"answer": rng.choice(REFUSALS)}
            v["task_type"] = "regulation_abstain"
            v["negative"] = "hard" if (hard and siblings) else "easy"
            out.append(v)
            break
    return out


def paraphrase_pairs(pairs, n: int, model: str, url: str, rng: random.Random,
                     log_every: int = 200):
    """
    Ask a local model for n rephrasings of each closed-book question.

    Filters, in order: the rewrite must not contain the answer, must not
    collapse to the original, must keep at least half the question's content
    words (so the subject survives), and must be new across the whole batch.
    """
    import requests
    out, seen = [], set()
    for i, r in enumerate(pairs, 1):
        q = r["instruction"].strip()
        o = r.get("output")
        ans = (o.get("answer", "") if isinstance(o, dict) else str(o)).strip()
        prompt = (f"다음 질문을 뜻은 같지만 표현이 다른 질문 {n}개로 바꿔 쓰시오.\n"
                  f"규칙: 묻는 대상과 조건은 유지할 것. 답을 포함하지 말 것. "
                  f"각 줄에 하나씩, 번호 없이 쓸 것.\n\n질문: {q}")
        try:
            resp = requests.post(f"{url}/api/generate", json={
                "model": model, "prompt": prompt, "stream": False, "think": False,
                "options": {"temperature": 0.8, "num_predict": 400, "num_ctx": 4096}},
                timeout=180).json().get("response", "")
        except Exception:
            continue
        base_words = content_words(q)
        kept = 0
        for line in resp.splitlines():
            cand = line.strip().strip("-•").strip()
            if not (10 <= len(cand) <= 300):
                continue
            if cand == q or (ans and ans in cand) or cand in seen:
                continue
            if len(content_words(cand) & base_words) < len(base_words) * 0.4:
                continue
            v = json.loads(json.dumps(r, ensure_ascii=False))
            v["id"] = f"{r.get('id', 'sft')}_pp{kept}"
            v["instruction"] = cand
            v["task_type"] = "regulation_recall_paraphrase"
            out.append(v)
            seen.add(cand)
            kept += 1
            if kept >= n:
                break
        if i % log_every == 0:
            print(f"  paraphrase {i}/{len(pairs)} -> {len(out)}", flush=True)
    return out


def enum_pairs(chunks, max_pairs: int, min_items: int, max_per_doc: int,
               rng: random.Random):
    """
    Mine full enumerations out of the training-side chunks.

    A pair is admitted only when a lead-in sentence introduces the list and the
    list does not run to the chunk edge, for the same reasons the benchmark's
    miner demands both: without a lead-in the question is unanswerable, and a
    cut list makes the answer key incomplete.
    """
    out, per_doc = [], {}
    for c in chunks:
        doc = c.get("doc_id", "")
        if per_doc.get(doc, 0) >= max_per_doc:
            continue
        lines = c.get("text", "").splitlines()
        i = 0
        while i < len(lines):
            m = ITEM_RE.match(lines[i])
            if not m or m.group(1) not in ("1", None) and not m.group(2):
                i += 1
                continue
            # collect the run of consecutive items
            items, j = [], i
            while j < len(lines):
                mj = ITEM_RE.match(lines[j])
                if not mj:
                    break
                items.append(mj.group(3).strip())
                j += 1
            ended_inside = j < len(lines)          # the list did not hit the edge
            lead = lines[i - 1].strip() if i > 0 else ""
            if (len(items) >= min_items and ended_inside
                    and lead and LEAD_RE.search(lead)):
                out.append({
                    "id": f"enum_{c['id']}_{i}",
                    "task_type": "regulation_enumeration",
                    "doc_id": doc,
                    "source_doc_ids": [doc],
                    "instruction": f"「{doc}」에서 다음을 모두 나열하시오: {lead}",
                    "input": {"context": "", "metadata": {"language": "ko"}},
                    "output": {"answer": ", ".join(items)},
                })
                per_doc[doc] = per_doc.get(doc, 0) + 1
                if per_doc[doc] >= max_per_doc:
                    break
            i = j if j > i else i + 1
    rng.shuffle(out)
    return out[:max_pairs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--sft", type=Path, default=TRAIN / "train_sft.jsonl")
    ap.add_argument("--dapt", type=Path, default=TRAIN / "train_dapt.jsonl")
    ap.add_argument("-o", "--out", type=Path, default=TRAIN / "train_sft_aug.jsonl")
    ap.add_argument("--closed-ratio", type=float, default=0.4,
                    help="share of open-book pairs also emitted as closed-book "
                         "variants (default 0.4)")
    ap.add_argument("--no-enum", action="store_true",
                    help="skip the enumeration pairs")
    ap.add_argument("--enum-max", type=int, default=4000,
                    help="cap on mined enumeration pairs (default 4000)")
    ap.add_argument("--enum-min-items", type=int, default=3,
                    help="shortest list worth asking about (default 3)")
    ap.add_argument("--enum-max-per-doc", type=int, default=6,
                    help="per-document cap, so one code does not dominate (default 6)")
    ap.add_argument("--abstain-ratio", type=float, default=0.0,
                    help="share of open-book pairs re-issued with an unrelated "
                         "clause and a refusal target (R-Tuning style). 0 disables")
    ap.add_argument("--hard-negatives", action="store_true",
                    help="draw the swapped clause from the same document rather "
                         "than an unrelated one, so refusal cannot be decided "
                         "from vocabulary alone")
    ap.add_argument("--paraphrase-n", type=int, default=0,
                    help="rephrasings per closed-book question, generated by a "
                         "local model. 0 disables")
    ap.add_argument("--paraphrase-model", default="qwen3:8b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = list(load(args.sft))
    closed = closed_variants(base, args.closed_ratio, rng)
    enums = [] if args.no_enum else enum_pairs(
        load(args.dapt), args.enum_max, args.enum_min_items,
        args.enum_max_per_doc, rng)

    abstains = (abstain_pairs(base, args.abstain_ratio, rng, args.hard_negatives)
                if args.abstain_ratio > 0 else [])
    paras = (paraphrase_pairs(closed, args.paraphrase_n, args.paraphrase_model,
                              args.ollama_url, rng)
             if args.paraphrase_n > 0 and closed else [])

    merged = base + closed + enums + abstains + paras
    rng.shuffle(merged)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"open-book originals {len(base)}")
    print(f"closed-book variants {len(closed)}  (ratio {args.closed_ratio})")
    print(f"enumeration pairs    {len(enums)}")
    nhard = sum(1 for r in abstains if r.get("negative") == "hard")
    print(f"abstention pairs     {len(abstains)}  (ratio {args.abstain_ratio}, "
          f"hard {nhard}, easy {len(abstains) - nhard})")
    print(f"paraphrase pairs     {len(paras)}  (n {args.paraphrase_n})")
    print(f"total -> {args.out}  {len(merged)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
