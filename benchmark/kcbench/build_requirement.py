"""Mine a requirement track from threshold items already in the benchmark.

A threshold item asks for a figure and keys on it: "몇 % 이상인가" -> "90 %".
Answering it well is a narrow skill. The question a site engineer actually has
is the one the clause answers — what must be done, under which condition — and
the benchmark had no item shaped like that because it had no way to grade a
sentence. It has one now.

    python cb.py requirement                    # from uc2, uc1 and track2_sft
    python cb.py requirement --sources uc2_spec_threshold.jsonl --out req.jsonl
    python cb.py requirement --usecase uc7_requirement_v2   # the repaired track

Nothing here is written by a model. The answer is the sentence in the clause
that states the requirement, lifted verbatim, located by the figure and
qualifier the source item already carries. That keeps the key checkable against
the document the same way every other track is, and it is why the admission
filter below is strict: a sentence that cannot be shown to be a requirement is
dropped rather than paraphrased into one.

Read the scores open book. Closed book this asks a model to reproduce clause
text from memory, which for an amendment-exposed clause is the thing the
benchmark argues against training; the item is here to measure whether a model
given the passage can state a complete requirement instead of emitting a bare
number.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from kcbench.common import (item_lang, add_common_args, describe, item_id, log,
                            read_jsonl, resolve_config, write_jsonl)

LOG = log("requirement")

DEFAULT_SOURCES = ("uc2_spec_threshold.jsonl", "uc1_safety_qa.jsonl",
                   "track2_sft.jsonl")

# Korean regulation ends a sentence with 한다 / 할 것 / 아니된다 and their
# polite forms. Splitting on the full stop alone loses most of it, because the
# extracted text drops as many stops as it keeps.
_END = re.compile(r"(?<=다)\.|(?<=것)\.|(?<=다)(?=\s*\n)|(?<=할 것)(?=\s|$)|[.。](?=\s)")

# What makes a sentence a requirement rather than a definition or a reference.
_REQUIRES = re.compile(r"(한다|할 것|하여야 한다|이어야 한다|되어야 한다|아니된다|"
                       r"아니 된다|아니하여야 한다|하여야 함|해야 한다)[.」)]?$")

# Damage from the PDF text layer: a clause number or table caption spliced into
# the middle of a sentence, ruled lines, leader dots.
_NOISE = re.compile(r"\s\d+\.\d+\s|\s\(\d+\)\s.{0,6}\s\(\d+\)|[│┃|]{2,}|\.{4,}|\s{4,}")
_HEADING = re.compile(r"^제\d+조\s*\(")

MIN_CHARS, MAX_CHARS = 20, 180
MAX_COMMAS = 6

# Everything above is the Korean-regulation default. A requirement sentence in
# another jurisdiction ends differently, a different text layer damages
# differently, and a sentence length that reads as an enumeration here may be
# ordinary elsewhere -- so `config.json` replaces any of it under
# usecases.uc7_requirement.
_ASKS = re.compile(r"(은|는|이|가)?\s*(몇|얼마).*$")

# The source question often opens with the clause's own list marker -- "(3-1)",
# "1.", "가." -- which is part of the document's numbering, not of the subject.
_LEAD = re.compile(r"^\s*(?:\(\s*[0-9가-힣]+(?:[-.][0-9]+)*\s*\)|"
                   r"[0-9]+(?:[-.][0-9]+)*\s*[.)]|[가-힣]\s*[.)])\s*")

# The same numbering spliced into the middle of a subject -- "수직높이임 (2)
# 비탈면높" -- means the question line was damaged, not that the clause talks
# about such a thing.
_MARK = re.compile(r"\(\s*[0-9가-힣]+(?:[-.][0-9]+)*\s*\)|\s\d+\.\d+(?:\s|$)")

_DIGIT = re.compile(r"\d")

MINING_DEFAULTS: Dict[str, Any] = {
    "sentence_end": _END.pattern,
    "requirement_ending": _REQUIRES.pattern,
    "extraction_noise": _NOISE.pattern,
    "clause_heading": _HEADING.pattern,
    "asks_pattern": _ASKS.pattern,
    "min_chars": MIN_CHARS,
    "max_chars": MAX_CHARS,
    "max_commas": MAX_COMMAS,
    "min_subject_chars": 2,
    "max_subject_chars": 60,
    "sources": list(DEFAULT_SOURCES),
    # The sentences wrapped around a mined requirement. A citation convention
    # and a question frame are corpus properties, not method ones.
    "citation": {"document": "「{doc}」", "fallback": "해당 기준"},
    "question": "{where}에 따르면, {subject}에 관하여 무엇을 어떻게 하여야 하는가?",
    "instruction": "조문에 적힌 대로 한 문장만 쓰고 다른 설명은 쓰지 마시오.",
    "verified": "조문에서 그대로 옮긴 요건 문장. 수치와 한정어로 위치를 특정함.",
    # The three admission checks added after the v052 open-book run measured
    # the key set's error floor: of 37 failures at grounded 0.711, eleven keyed
    # a subject the clause states more than one requirement about (or a
    # damaged one), and six keyed a sentence carrying a second obligation the
    # question never asked. The original uc7_requirement entry in config.json
    # pins all three off so a rebuild reproduces the published track; the
    # repair lives in uc7_requirement_v2.
    "subject_checked": True,
    "rival_any_sentence": True,
    "split_coordination": True,
    "coordination_pattern": r"(?<=[고며]),\s*",
}


def mining(cfg: Dict[str, Any] | None = None,
           usecase: str = "uc7_requirement") -> Dict[str, Any]:
    """The mining parameters, with any config override compiled in."""
    over = (((cfg or {}).get("usecases") or {}).get(usecase) or {})
    out: Dict[str, Any] = {}
    for k, default in MINING_DEFAULTS.items():
        v = over.get(k, default)
        if isinstance(default, str) and k.endswith(("_end", "_ending", "_noise",
                                                    "_heading", "_pattern")):
            try:
                out[k] = re.compile(v)
            except re.error as exc:
                LOG.error("%s.%s is not a valid regex (%s); "
                          "using the default", usecase, k, exc)
                out[k] = re.compile(default)
        else:
            out[k] = v
    return out

# "표준설계응답스펙트럼은 몇 % 이상이어야 하는가?" names what the clause is about;
# without it a question that says only which document it comes from has many
# right answers and is not an item.


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def subject_of(question: str, m: Dict[str, Any] | None = None) -> Optional[str]:
    """What the source item was asking about, with the interrogative stripped."""
    m = m or mining()
    s = _LEAD.sub("", question or "")
    s = m["asks_pattern"].sub("", s).strip().rstrip("은는이가").strip()
    return s if m["min_subject_chars"] <= len(s) <= m["max_subject_chars"] else None


def sentences(text: str, m: Dict[str, Any] | None = None) -> List[tuple[str, int, int]]:
    """
    Each sentence as (verbatim slice, start, end).

    The slice is returned unmodified rather than whitespace-collapsed. An answer
    key that says it was lifted from the clause has to be findable in the clause,
    and normalising it for readability quietly breaks that.
    """
    m = m or mining()
    spans, last = [], 0
    for hit in m["sentence_end"].finditer(text or ""):
        a, b = last, hit.end()
        last = b
        if (text[a:b]).strip():
            spans.append((a, b))
    if (text or "")[last:].strip():
        spans.append((last, len(text)))
    out = []
    for a, b in spans:
        raw = text[a:b]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        out.append((text[a + lead:b - trail], a + lead, b - trail))
    return out


def rejection(s: str, m: Dict[str, Any] | None = None) -> Optional[str]:
    """Why this sentence cannot serve as an answer key, or None if it can."""
    m = m or mining()
    if not m["min_chars"] <= len(s) <= m["max_chars"]:
        return "length"
    if not m["requirement_ending"].search(s):
        return "not a requirement"
    if m["extraction_noise"].search(s):
        return "extraction noise"
    if m["clause_heading"].match(s):
        return "clause heading"
    if s.count(",") >= m["max_commas"]:
        return "enumeration"
    return None


def find_requirement(item: Dict[str, Any], m: Dict[str, Any] | None = None):
    """
    The one sentence in the item's own context that states its threshold.

    Matched on the figure and its unit with whitespace removed, because the
    text layer spaces numbers inconsistently — "5 년" in the key against "5년"
    on the page. Ambiguity is refused rather than resolved by picking the
    first: two sentences carrying the same limit mean the item does not
    identify one of them.
    """
    m = m or mining()
    value, unit = item.get("answer_value"), item.get("answer_unit")
    if value is None or not unit:
        return None, "no figure"
    num = str(int(value)) if float(value) == int(value) else str(value)
    key = _squash(num + str(unit))

    found = [t for t in sentences(item.get("context") or "", m)
             if key in _squash(t[0])]
    qual = item.get("qualifier")
    if qual:
        narrowed = [t for t in found if _squash(qual) in _squash(t[0])]
        found = narrowed or found
    if not found:
        return None, "figure not in a sentence"

    admitted = [t for t in found if rejection(t[0], m) is None]
    if not admitted:
        return None, rejection(found[0][0], m) or "rejected"
    if len(admitted) > 1:
        return None, "ambiguous"
    return admitted[0], "ok"


def _single_obligation(sentence: str, start: int, end: int, subject: str,
                       src: Dict[str, Any], m: Dict[str, Any]):
    """
    Cut a coordinated sentence down to the one obligation the item asks about.

    "평상시에도 항상 점등되어야 하고, 비상 상황 발생 시 60분 이상 계속
    점등되어야 한다" states two requirements; the question asks about one, and
    a key carrying both fails `covered` on any answer that gives only the
    asked one — measured at six of 37 open-book failures. The cut is a
    verbatim sub-span, located by the figure match that admitted the sentence,
    so the key stays findable in the clause. It is only made when the subject
    sits in the keyed segment and in no other figure-bearing one — otherwise
    the question does not say which obligation it means, and the sentence is
    refused rather than guessed at.

    Returns (sentence, start, end, truncated) or (None, None, None, reason).
    """
    parts, last = [], 0
    for mo in m["coordination_pattern"].finditer(sentence):
        parts.append((sentence[last:mo.start()], last))
        last = mo.end()
    parts.append((sentence[last:], last))
    if sum(1 for p, _ in parts if _DIGIT.search(p)) < 2:
        return sentence, start, end, False      # one obligation, nothing to cut

    value, unit = src.get("answer_value"), src.get("answer_unit")
    num = str(int(value)) if float(value) == int(value) else str(value)
    key = _squash(num + str(unit))
    holders = [(p, off) for p, off in parts if key in _squash(p)]
    if len(holders) != 1:
        return None, None, None, "coordinated obligations"

    seg, off = holders[0]
    sj = _squash(subject)
    if sj not in _squash(seg) or any(sj in _squash(p) and _DIGIT.search(p)
                                     for p, o in parts if o != off):
        return None, None, None, "coordinated obligations"

    kept = seg.strip()
    if len(kept) < m["min_chars"]:
        return None, None, None, "truncated too short"
    lead = len(seg) - len(seg.lstrip())
    a = start + off + lead
    return kept, a, a + len(kept), True


def _split_of(src: Dict[str, Any]) -> Optional[str]:
    """
    Which side of the contamination line the item sits on.

    Use-case tracks carry it because they are mined from both sides on purpose.
    track2 does not carry it because it is built only from held-out documents,
    so the field would be constant; a derived item still needs it stated, or
    the new track reports a by_split breakdown with a third of it missing.
    """
    if src.get("split"):
        return src["split"]
    return "holdout" if src.get("track") == "sft" else None


def build(rows: List[Dict[str, Any]],
          cfg: Dict[str, Any] | None = None,
          usecase: str = "uc7_requirement") -> tuple[List[dict], collections.Counter]:
    m = mining(cfg, usecase)
    out, why = [], collections.Counter()
    seen = set()
    for src in rows:
        hit, reason = find_requirement(src, m)
        why[reason] += 1
        if not hit:
            continue
        sentence, start, end = hit
        if _squash(sentence) in seen:      # the same clause reached twice
            why["duplicate"] += 1
            continue

        # Without a subject the question names only a document, and a document
        # sets many requirements — the item would have several right answers.
        subject = subject_of(src.get("question_ko") or "", m)
        if not subject:
            why["no subject"] += 1
            continue
        ctx = src.get("context") or ""
        if m["subject_checked"]:
            # A list marker spliced into the subject means the question line
            # was damaged in extraction, and a subject the clause never states
            # cannot anchor the rival scan below. Either way the item has no
            # single checkable answer.
            if _MARK.search(subject):
                why["subject mangled"] += 1
                continue
            if _squash(subject) not in _squash(ctx):
                why["subject not in clause"] += 1
                continue
        # The question names a document and a subject. If another sentence in
        # the same passage also states a requirement about that subject, the
        # question has two right answers and the key only records one. A rival
        # that fails admission — too long, noisy — is still a true alternative
        # answer to a model reading the clause, so by default anything shaped
        # like a requirement or carrying a figure counts; the pinned v1
        # behaviour counted admissible sentences only (6 of 134 at the time).
        others = [t[0] for t in sentences(ctx, m)
                  if _squash(t[0]) != _squash(sentence)
                  and _squash(subject) in _squash(t[0])]
        if m["rival_any_sentence"]:
            rivals = [s for s in others
                      if m["requirement_ending"].search(s) or _DIGIT.search(s)]
        else:
            rivals = [s for s in others if rejection(s, m) is None]
        if rivals:
            why["subject not unique"] += 1
            continue
        seen.add(_squash(sentence))

        truncated = False
        if m["split_coordination"]:
            sentence, start, end, cut = _single_obligation(
                sentence, start, end, subject, src, m)
            if sentence is None:
                why[cut] += 1
                continue
            truncated = bool(cut)

        doc, clause = src.get("doc"), src.get("clause")
        cite = m.get("citation") or {}
        if doc:
            where = str(cite.get("document", "「{doc}」")).format(doc=doc)
            if clause:
                where += " " + clause
        else:
            where = str(cite.get("fallback", "해당 기준"))
        span = [start, end]

        # A truncated key sits inside a sentence that also states obligations
        # the question never asked. The default instruction says to reproduce
        # the clause's sentence, which would make the faithful answer fail
        # `supported` against the shorter key — so these items ask for the one
        # asked requirement instead.
        if truncated:
            i_ko = "조문에 적힌 대로, 질문이 묻는 요건 한 가지만 쓰고 다른 설명은 쓰지 마시오."
            i_en = ("Only the one requirement the question asks about, "
                    "as the clause words it. Nothing else.")
        else:
            i_ko = "조문에 적힌 대로 한 문장만 쓰고 다른 설명은 쓰지 마시오."
            i_en = "One sentence, as the clause words it. Nothing else."

        out.append({
            "benchmark": src.get("benchmark", "kcbench"),
            "benchmark_version": src.get("benchmark_version"),
            "schema": src.get("schema"),
            "track": "usecase",
            "usecase": usecase,
            "eval_type": "sentence",
            "cognitive_level": "application",
            "id": item_id(usecase, doc, clause, sentence),
            "doc": doc,
            "clause": clause,
            "category": src.get("category"),
            "split": _split_of(src),
            "lang": item_lang(cfg),
            "context": ctx,
            "question_ko": (str(m.get("question", "{where}에 따르면, {subject}에 관하여 "
                                                  "무엇을 어떻게 하여야 하는가?"))
                            .format(where=where, subject=subject)),
            "question_en": (f"According to {where}, what is required regarding "
                            f"{subject}?"),
            "instruction_ko": (i_en if item_lang(cfg) == "en" else i_ko),
            "instruction_en": i_en,
            "answer": sentence,
            "answer_ko": sentence,
            "answer_lang": "ko",
            "answer_value": src.get("answer_value"),
            "answer_unit": src.get("answer_unit"),
            "qualifier": src.get("qualifier"),
            "asks_about": subject,
            "volatility": src.get("volatility"),
            "routing": src.get("routing"),
            "volatility_basis": src.get("volatility_basis"),
            "verified_ko": str(m.get("verified",
                                     "조문에서 그대로 옮긴 요건 문장. 수치와 한정어로 위치를 특정함.")),
            "verified_en": ("the requirement sentence lifted verbatim from the clause, "
                            "located by the figure and qualifier the source item keys on"),
            "answer_span": span,
            "provenance": {**(src.get("provenance") or {}),
                           "derived_from": src["id"],
                           "derived_from_track": src.get("usecase") or src.get("track")},
        })
    return out, why


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="mine a requirement-sentence track from threshold items",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--usecase", default="uc7_requirement",
                    help="which usecases.<name> entry supplies the mining "
                         "parameters and names the items")
    ap.add_argument("--sources", default=None,
                    help="comma-separated item files to derive from "
                         "(default: usecases.<usecase>.sources)")
    ap.add_argument("--out", default=None,
                    help="output file (default: the usecase's track_file)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be mined and write nothing")
    args = ap.parse_args(argv)
    cfg = resolve_config(args)
    describe(cfg)

    data = Path(cfg["out_dir"])
    rows: List[dict] = []
    names = ([x.strip() for x in args.sources.split(",") if x.strip()]
             if args.sources else list(mining(cfg, args.usecase)["sources"]))
    for name in names:
        p = data / name
        if not p.is_file():
            LOG.warning("no such source: %s", p.name)
            continue
        rows.extend(read_jsonl(p))
    if not rows:
        LOG.error("no source items")
        return 1

    items, why = build(rows, cfg, args.usecase)
    print(f"from {len(rows)} source item(s):")
    for reason, n in why.most_common():
        print(f"  {reason:24s} {n:5d}")
    print(f"\nmined {len(items)}")
    by = collections.Counter(r.get("split") for r in items)
    print("  by split:", dict(by))
    if items:
        print(f"\n  example: {items[0]['answer'][:110]}")

    if args.dry_run:
        return 0
    out_name = args.out or ((cfg.get("usecases") or {}).get(args.usecase)
                            or {}).get("track_file", f"{args.usecase}.jsonl")
    write_jsonl(data / out_name, items)
    LOG.info("wrote %s", data / out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
