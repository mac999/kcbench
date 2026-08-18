#!/usr/bin/env python3
"""
Build the use-case tracks defined in config.json's "usecases" section.

    python build_usecases.py
    python build_usecases.py --only uc4_faithfulness
    python build_usecases.py -i /data/ai_ready_v3 -o /tmp/bench

Each entry in the config names a builder and its parameters, so adding a use
case later is a config entry plus (at most) one builder function here. The
builders reuse the track 2 miners, which is what keeps the admission rules —
subject quality, lead-in requirements, duplicate handling — identical across
every track that asks about regulation text.

Items mined from documents on the training side of the split carry
split="train" and contamination="intentional", same as the probe set. They are
the open-book (RAG) measurement; only holdout-side items may be read as
closed-book evidence of generalisation.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

from kcbench.build_tracks import (CLAUSE_RE, Rejects, _base, _instr,
                          _nameset_candidate, _numeric_candidates, _provenance,
                          josa, load_reviews, reviewed_out)
from kcbench.common import (QUALIFIER_EN, UNIT_EN, add_common_args, describe,
                    generated_documents, item_id, log, resolve_config, utc_now,
                    write_json, write_jsonl)

LOG = log("usecases")

ABSTAIN_TOKEN_KO = "자료 없음"

FAITHFULNESS_INSTRUCTION = {
    "ko": ("주어진 조문에 답이 있으면 숫자와 단위만 답하시오. 조문에서 답을 "
           f"확인할 수 없으면 '{ABSTAIN_TOKEN_KO}'이라고만 답하시오."),
    "en": ("If the passage states the answer, reply with the number and unit only. "
           "If the answer cannot be found in the passage, reply exactly 'no data'."),
}

LABEL_VOCAB = {
    "bim_site_comparison": ["match", "partial_match", "mismatch", "unknown"],
    "progress_assessment": ["foundation", "structure", "finishing", "completed", "unknown"],
}

LABEL_QUESTION = {
    "bim_site_comparison": {
        "ko": "첫 번째 이미지는 BIM 렌더링이고 두 번째 이미지는 같은 시점의 현장 사진이다. "
              "설계와 현장이 일치하는지 판정하시오.",
        "en": "The first image is a BIM rendering and the second is a site photo of the "
              "same view. Judge whether the site matches the design.",
    },
    "progress_assessment": {
        "ko": "이 현장 사진의 시공 진행 단계를 판정하시오.",
        "en": "Judge the construction progress stage shown in this site photo.",
    },
}


def _label_instr(task: str, vocab: list | None = None) -> dict:
    words = ", ".join(vocab if vocab is not None else LABEL_VOCAB[task])
    return {"instruction_ko": f"다음 중 하나의 단어로만 답하시오: {words}",
            "instruction_en": f"Answer with exactly one of: {words}"}


def _side(doc: dict, held_keys: set) -> dict:
    if doc["generated_dir"] in held_keys:
        return {"split": "holdout"}
    return {"split": "train", "contamination": "intentional"}


def _match_docs(cfg, uc: dict, held_keys: set) -> List[dict]:
    """
    Documents this use case draws from, holdout side first.

    Filtering is on category and name pattern rather than a hand-kept list,
    so a corpus refresh that adds another 작업지침 picks it up without a config
    change.
    """
    cats = set(uc.get("categories") or [])
    pats = uc.get("doc_patterns") or []
    out = []
    for d in generated_documents(cfg):
        if cats and d["category"] not in cats:
            continue
        if pats and not any(p in d["generated_dir"] for p in pats):
            continue
        out.append(d)
    if not uc.get("include_train_side", False):
        out = [d for d in out if d["generated_dir"] in held_keys]
    out.sort(key=lambda d: (d["generated_dir"] not in held_keys, d["generated_dir"]))
    return out


# builder: doc_filtered_qa (UC1 safety, UC2 specification thresholds)

def build_doc_filtered_qa(cfg, holdout, key: str, uc: dict, rej: Rejects) -> List[dict]:
    """Track 2's numeric and nameset questions, restricted to the UC's documents."""
    held_keys = {d["generated_dir"] for d in holdout["pdf_documents"]}
    docs = _match_docs(cfg, uc, held_keys)
    LOG.info("%s: %d matching document(s), %d on the holdout side", key, len(docs),
             sum(1 for d in docs if d["generated_dir"] in held_keys))
    reviews = load_reviews(cfg)
    want = uc.get("target_items", 120)
    cap = uc.get("max_items_per_doc", 12)
    nameset_want = int(want * uc.get("nameset_share", 0.2))
    numeric_want = want - nameset_want

    rows: List[dict] = []
    seen: set = set()

    for d in docs:
        if len(rows) >= numeric_want:
            break
        made = 0
        for chunk in d["chunk_rows"]:
            if made >= cap or len(rows) >= numeric_want:
                break
            text = chunk["text"]
            clause = CLAUSE_RE.search(text)
            for f in _numeric_candidates(text, rej, cfg):
                subject, value, unit, qual = f["subject"], f["value"], f["unit"], f["qualifier"]
                if (subject, value, unit, qual) in seen:
                    continue
                answer_ko = f"{value} {unit}"
                if f"{value}{unit}" not in text.replace(" ", ""):
                    continue
                ident = item_id(key, chunk["sha256"], subject, value, unit, qual)
                if reviewed_out(reviews, ident, rej):
                    continue
                seen.add((subject, value, unit, qual))
                topic = josa(subject, "은", "는")
                rows.append({
                    **_base("usecase", "numeric", d),
                    "id": ident,
                    "usecase": key,
                    **_side(d, held_keys),
                    "lang": "ko",
                    "context": text,
                    "question_ko": f"{subject}{topic} 몇 {unit} {qual}이어야 하는가?",
                    "question_en": f"What is the stipulated threshold for '{subject}', "
                                   f"in {UNIT_EN[unit]} ({QUALIFIER_EN[qual]})?",
                    **_instr("numeric"),
                    "answer": answer_ko,
                    "answer_ko": answer_ko,
                    "answer_en": f"{value} {UNIT_EN[unit]}",
                    "answer_value": float(value.replace(",", "")),
                    "answer_unit": unit,
                    "answer_unit_en": UNIT_EN[unit],
                    "qualifier": qual,
                    "qualifier_en": QUALIFIER_EN[qual],
                    "clause": (f"제{clause.group(1)}조" if clause else None),
                    "verified_en": "stated threshold, present in the passage as keyed",
                    "verified_ko": "본문에 명시된 기준값.",
                    "provenance": _provenance(cfg, d, chunk, f["span"]),
                })
                made += 1
                if made >= cap or len(rows) >= numeric_want:
                    break

    numeric_n = len(rows)
    # Track 2 takes one enumeration per document; a use case drawing on a
    # narrow document pool cannot afford that, so the per-doc cap is a setting.
    ns_cap = uc.get("nameset_max_per_doc", 1)
    for d in docs:
        if len(rows) - numeric_n >= nameset_want:
            break
        made = 0
        for n, chunk in enumerate(d["chunk_rows"]):
            if made >= ns_cap:
                break
            prev_tail = d["chunk_rows"][n - 1]["text"][-400:] if n else ""
            found = _nameset_candidate(chunk["text"], rej, cfg, prev_tail)
            if not found:
                continue
            lead, kind, items, span, prefix = found
            if (lead, tuple(items)) in seen:
                continue
            ident = item_id(key, chunk["sha256"], lead)
            if reviewed_out(reviews, ident, rej):
                continue
            seen.add((lead, tuple(items)))
            if kind == "heading":
                q_ko = f"조문 {lead}에서 정한 항목을 모두 나열하시오."
                q_en = f"List every item stipulated in {lead} of the passage."
            else:
                q_ko = f"조문에 따르면 {lead}에 해당하는 항목을 모두 나열하시오."
                q_en = f"According to the passage, list every item that falls under '{lead}'."
            rows.append({
                **_base("usecase", "nameset", d),
                "id": ident,
                "usecase": key,
                **_side(d, held_keys),
                "lang": "ko",
                "context": prefix + chunk["text"],
                "question_ko": q_ko,
                "question_en": q_en,
                **_instr("nameset"),
                "answer": items,
                "answer_ko": items,
                "answer_lang": "ko",
                "lead_in": lead,
                "lead_in_kind": kind,
                "verified_en": "items parsed from a numbered list introduced by the passage",
                "verified_ko": "지문이 스스로 예고한 각 호 목록에서 추출한 항목.",
                "provenance": _provenance(cfg, d, chunk, span),
            })
            made += 1
    return rows


# builder: vlm_labels (UC3 site photo vs BIM render)

def build_vlm_labels(cfg, holdout, key: str, uc: dict, rej: Rejects) -> List[dict]:
    """
    Cross-image judgement items from the dataset's own VLM annotations.

    The ground truth is the label the generation pipeline attached
    (match/mismatch, progress stage), which is an annotation, not a measurement
    like the IFC catalogue. Every item says so in verified_*; treat scores as a
    weaker signal than tracks 2 and 3 until the labels have been reviewed.
    """
    tasks = uc.get("task_types", ["bim_site_comparison", "progress_assessment"])
    # 'unknown' as ground truth grades a model's honest judgement as wrong and
    # an annotator's shrug as right; excluded labels leave both the answer key
    # and the offered choices.
    excluded = set(uc.get("exclude_labels", ["unknown"]))
    vocab = {t: [w for w in LABEL_VOCAB[t] if w not in excluded] for t in LABEL_VOCAB}
    rows: List[dict] = []
    for m in holdout["ifc_models"]:
        base = Path("40_bim_models_ifc") / m["category"] / m["model"]
        path = cfg["generated_dir"] / base / "vlm_training_data.jsonl"
        if not path.exists():
            rej("holdout model has no vlm_training_data.jsonl", m["model"])
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            task = r.get("task_type")
            if task not in tasks:
                continue
            out = r.get("output") or {}
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except json.JSONDecodeError:
                    out = {}
            label = out.get("label")
            if label in excluded:
                rej("label excluded by config", f"{m['model']} {task} {label}")
                continue
            if label not in vocab[task]:
                rej("row has no label in the task's vocabulary", f"{m['model']} {task} {label}")
                continue
            imgs = [str(base / img) for img in (r.get("images") or [])]
            if task == "progress_assessment":
                imgs = imgs[-1:]  # judge the site photo alone; the render would leak the answer
            missing = [p for p in imgs if not (cfg["generated_dir"] / p).exists()]
            if not imgs or missing:
                rej("image file missing", f"{m['model']} {missing}")
                continue
            doc_like = {"stem": m["model"], "category": m["category"]}
            rows.append({
                **_base("usecase", "label", doc_like),
                "id": item_id(key, m["model"], task, i),
                "usecase": key,
                "cognitive_level": "understanding",
                "split": "holdout",
                "lang": "ko",
                "label_task": task,
                "question_ko": LABEL_QUESTION[task]["ko"],
                "question_en": LABEL_QUESTION[task]["en"],
                **_label_instr(task, vocab[task]),
                "answer": label,
                "answer_lang": "neutral",
                "label_vocab": vocab[task],
                "images": imgs,
                "verified_en": "generation-time annotation from the dataset itself, not an "
                               "independent measurement; weaker ground truth than the IFC catalogue",
                "verified_ko": "데이터셋 생성 시 부여된 주석 라벨. IFC 카탈로그보다 약한 근거.",
                "provenance": {
                    "dataset": cfg["generated_dir"].name,
                    "dataset_file": str(base / "vlm_training_data.jsonl"),
                    "row_index": i,
                    "document": m["model"],
                    "category": m["category"],
                    "source_ifc": m.get("source_ifc"),
                },
            })
    counts = collections.Counter((r["label_task"], r["answer"]) for r in rows)
    for (task, label), n in sorted(counts.items()):
        LOG.info("%s: %-22s %-13s %d", key, task, label, n)
    return rows


# builder: context_swap (UC4 faithfulness under retrieval failure)

def build_context_swap(cfg, holdout, key: str, uc: dict, rej: Rejects) -> List[dict]:
    """
    Track 2 numeric items with the passage swapped for an unrelated one.

    Retrieval fails in every deployed RAG system; the dominant failure mode of
    the agent is then a confident fabricated number. A swapped item is correct
    only when the model abstains, and the unswapped controls catch a model that
    learns to abstain on everything.
    """
    src = cfg["out_dir"] / uc.get("source_track", "track2_sft.jsonl")
    if not src.exists():
        LOG.error("%s: source track missing (%s) - build track 2 first", key, src)
        return []
    items = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    numeric = [i for i in items if i.get("eval_type") == "numeric" and i.get("context")]
    rng = random.Random(uc.get("seed", 20260814))
    rng.shuffle(numeric)
    numeric = numeric[: uc.get("max_items", 160)]
    n_swap = int(len(numeric) * uc.get("swap_share", 0.5))

    rows: List[dict] = []
    for idx, it in enumerate(numeric):
        swapped = idx < n_swap
        context = it["context"]
        if swapped:
            # A donor context from a different document that does not happen to
            # contain the keyed answer; without that check a "wrong" passage can
            # still hold the right number and the item grades the model unfairly.
            answer_flat = f"{it['answer'].split()[0]}{it['answer_unit']}".replace(" ", "")
            donor = next(
                (d for d in numeric[idx + 1:] + numeric[:idx]
                 if d["doc"] != it["doc"]
                 and answer_flat not in d["context"].replace(" ", "")), None)
            if donor is None:
                rej("no donor context available", it["id"])
                continue
            context = donor["context"]
        rows.append({
            **{k: it[k] for k in ("benchmark", "benchmark_version", "schema",
                                  "doc", "category")},
            "track": "usecase",
            "eval_type": "faithfulness",
            "cognitive_level": "grounding",
            "id": item_id(key, it["id"], "swap" if swapped else "match"),
            "usecase": key,
            "split": "holdout",
            "lang": "ko",
            "context": context,
            "context_matches": not swapped,
            "question_ko": it["question_ko"],
            "question_en": it["question_en"],
            "instruction_ko": FAITHFULNESS_INSTRUCTION["ko"],
            "instruction_en": FAITHFULNESS_INSTRUCTION["en"],
            "answer": ABSTAIN_TOKEN_KO if swapped else it["answer"],
            "answer_value": it["answer_value"],
            "answer_unit": it["answer_unit"],
            "abstain_token": ABSTAIN_TOKEN_KO,
            "source_item": it["id"],
            "verified_en": ("context replaced with an unrelated passage verified not to "
                            "contain the keyed answer" if swapped else
                            "unmodified track 2 item, serving as the abstention control"),
            "verified_ko": ("정답이 없음을 확인한 무관한 조문으로 교체된 문항." if swapped
                            else "원본 그대로의 대조 문항."),
            "provenance": it.get("provenance"),
        })
    kinds = collections.Counter("swapped" if not r["context_matches"] else "control" for r in rows)
    LOG.info("%s: %d item(s) %s", key, len(rows), dict(kinds))
    return rows


# builder: missing_measures (UC5 incident analysis)

def build_missing_measures(cfg, holdout, key: str, uc: dict, rej: Rejects) -> List[dict]:
    """
    Incident-shaped questions built from safety work-standard enumerations.

    The corpus holds no real incident reports, so the scenario is synthetic: the
    item shows the measures a (fictional) investigation confirmed on site and
    asks which required measures are missing. The ground truth is the withheld
    subset of the clause's own list, which keeps the answer mechanical to check
    while the question exercises the UC5 shape — situation plus checklist in,
    omissions out.
    """
    held_keys = {d["generated_dir"] for d in holdout["pdf_documents"]}
    docs = _match_docs(cfg, uc, held_keys)
    LOG.info("%s: %d matching document(s)", key, len(docs))
    reviews = load_reviews(cfg)
    min_items = uc.get("min_items_per_list", 4)
    share = uc.get("removed_share", 0.34)
    want = uc.get("max_items", 80)

    rows: List[dict] = []
    seen: set = set()
    for d in docs:
        if len(rows) >= want:
            break
        for n, chunk in enumerate(d["chunk_rows"]):
            if len(rows) >= want:
                break
            prev_tail = d["chunk_rows"][n - 1]["text"][-400:] if n else ""
            found = _nameset_candidate(chunk["text"], rej, cfg, prev_tail)
            if not found:
                continue
            lead, kind, items, span, prefix = found
            if len(items) < min_items:
                rej(f"list holds fewer than {min_items} measures", lead)
                continue
            if (lead, tuple(items)) in seen:
                continue
            ident = item_id(key, chunk["sha256"], lead)
            if reviewed_out(reviews, ident, rej):
                continue
            seen.add((lead, tuple(items)))

            # Deterministic per item, not per run: the removed subset must not
            # change when an unrelated document is added to the corpus.
            k = max(1, round(len(items) * share))
            pick = random.Random(f"{uc.get('seed', 20260814)}:{chunk['sha256']}:{lead}")
            removed_idx = sorted(pick.sample(range(len(items)), k))
            removed = [items[i] for i in removed_idx]
            kept = [it for i, it in enumerate(items) if i not in removed_idx]

            confirmed = "\n".join(f"- {it}" for it in kept)
            rows.append({
                **_base("usecase", "nameset", d),
                "id": ident,
                "usecase": key,
                **_side(d, held_keys),
                "lang": "ko",
                "context": prefix + chunk["text"],
                "scenario_ko": ("공사 현장에서 사고가 발생하여 안전조치 이행 여부를 "
                                "조사하고 있다. 현장에서 이행이 확인된 조치는 다음과 같다.\n"
                                f"{confirmed}"),
                "question_ko": (f"위 조사 결과를 '{lead}' 기준과 대조할 때, 기준이 요구하지만 "
                                "이행이 확인되지 않은 조치를 모두 나열하시오."),
                "question_en": (f"An incident is under investigation; the measures above were "
                                f"confirmed on site. Against the requirements of '{lead}', "
                                "list every required measure not confirmed."),
                # Not the generic nameset instruction: the answer is prose, so
                # the item asks for the clause's own wording and is graded with
                # token-coverage matching rather than exact lines.
                "instruction_ko": "누락된 조치를 조문의 문구를 사용해 한 줄에 하나씩 "
                                  "나열하시오. 다른 설명은 쓰지 마시오.",
                "instruction_en": "List each missing measure on its own line, using the "
                                  "clause's wording. Nothing else.",
                "match_mode": "fuzzy",
                "answer": removed,
                "answer_ko": removed,
                "answer_lang": "ko",
                "lead_in": lead,
                "lead_in_kind": kind,
                "measures_total": len(items),
                "measures_confirmed": kept,
                "verified_en": "the withheld subset of the clause's own enumerated measures",
                "verified_ko": "조문의 각 호 목록에서 제외한 항목이 곧 정답.",
                "provenance": _provenance(cfg, d, chunk, span),
            })
    return rows


BUILDERS = {
    "doc_filtered_qa": build_doc_filtered_qa,
    "vlm_labels": build_vlm_labels,
    "context_swap": build_context_swap,
    "missing_measures": build_missing_measures,
}


def build_prompt_extra(item: dict) -> str | None:
    """The scenario block, when an item carries one. Used by evaluate.py."""
    return item.get("scenario_ko")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the use-case tracks named in config.json",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--only", metavar="KEY", help="build a single use case")
    ap.add_argument("--holdout", metavar="FILE",
                    help="holdout.json to build from (default <out-dir>/holdout.json)")
    ap.add_argument("--report", metavar="FILE",
                    help="rejection tally (default <out-dir>/usecase_rejections.json)")
    args = ap.parse_args(argv)

    cfg = resolve_config(args)
    describe(cfg)
    usecases: Dict[str, Any] = {
        k: v for k, v in (cfg.get("usecases") or {}).items()
        if not k.startswith("_") and isinstance(v, dict)}
    if not usecases:
        LOG.error("config has no usecases section")
        return 1

    hp = Path(args.holdout) if args.holdout else cfg["out_dir"] / "holdout.json"
    if not hp.exists():
        LOG.error("run build_holdout.py first - %s is missing", hp)
        return 1
    holdout = json.loads(hp.read_text(encoding="utf-8"))

    reports = []
    for key, uc in usecases.items():
        if args.only and key != args.only:
            continue
        if not uc.get("enabled", True):
            LOG.info("%s: disabled, skipping", key)
            continue
        builder = BUILDERS.get(uc.get("builder", ""))
        if builder is None:
            LOG.error("%s: unknown builder %r - choose from %s",
                      key, uc.get("builder"), ", ".join(BUILDERS))
            return 1
        rej = Rejects()
        rows = builder(cfg, holdout, key, uc, rej)
        out = cfg["out_dir"] / uc.get("track_file", f"{key}.jsonl")
        write_jsonl(out, rows)
        kinds = collections.Counter(r["eval_type"] for r in rows)
        splits = collections.Counter(r.get("split", "?") for r in rows)
        LOG.info("%s: %d item(s) %s split %s -> %s",
                 key, len(rows), dict(kinds), dict(splits), out.name)
        reports.append({"usecase": key, **rej.report(key)})

    path = Path(args.report) if args.report else cfg["out_dir"] / "usecase_rejections.json"
    write_json(path, {"built_at": utc_now(), "reports": reports})
    LOG.info("rejection tally in %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
