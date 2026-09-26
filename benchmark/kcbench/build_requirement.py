"""Mine a requirement track from threshold items already in the benchmark.

A threshold item asks for a figure and keys on it: "몇 % 이상인가" -> "90 %".
Answering it well is a narrow skill. The question a site engineer actually has
is the one the clause answers — what must be done, under which condition — and
the benchmark had no item shaped like that because it had no way to grade a
sentence. It has one now.

    python cb.py requirement                    # from uc2, uc1 and track2_sft
    python cb.py requirement --sources uc2_spec_threshold.jsonl --out req.jsonl

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

from kcbench.common import (add_common_args, describe, item_id, log,
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

# "표준설계응답스펙트럼은 몇 % 이상이어야 하는가?" names what the clause is about;
# without it a question that says only which document it comes from has many
# right answers and is not an item.
_ASKS = re.compile(r"(은|는|이|가)?\s*(몇|얼마).*$")


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def subject_of(question: str) -> Optional[str]:
    """What the source item was asking about, with the interrogative stripped."""
    s = _ASKS.sub("", question or "").strip().rstrip("은는이가").strip()
    return s if 2 <= len(s) <= 60 else None


def sentences(text: str) -> List[tuple[str, int, int]]:
    """
    Each sentence as (verbatim slice, start, end).

    The slice is returned unmodified rather than whitespace-collapsed. An answer
    key that says it was lifted from the clause has to be findable in the clause,
    and normalising it for readability quietly breaks that.
    """
    spans, last = [], 0
    for m in _END.finditer(text or ""):
        a, b = last, m.end()
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


def rejection(s: str) -> Optional[str]:
    """Why this sentence cannot serve as an answer key, or None if it can."""
    if not MIN_CHARS <= len(s) <= MAX_CHARS:
        return "length"
    if not _REQUIRES.search(s):
        return "not a requirement"
    if _NOISE.search(s):
        return "extraction noise"
    if _HEADING.match(s):
        return "clause heading"
    if s.count(",") >= MAX_COMMAS:
        return "enumeration"
    return None


def find_requirement(item: Dict[str, Any]):
    """
    The one sentence in the item's own context that states its threshold.

    Matched on the figure and its unit with whitespace removed, because the
    text layer spaces numbers inconsistently — "5 년" in the key against "5년"
    on the page. Ambiguity is refused rather than resolved by picking the
    first: two sentences carrying the same limit mean the item does not
    identify one of them.
    """
    value, unit = item.get("answer_value"), item.get("answer_unit")
    if value is None or not unit:
        return None, "no figure"
    num = str(int(value)) if float(value) == int(value) else str(value)
    key = _squash(num + str(unit))

    found = [t for t in sentences(item.get("context") or "") if key in _squash(t[0])]
    qual = item.get("qualifier")
    if qual:
        narrowed = [t for t in found if _squash(qual) in _squash(t[0])]
        found = narrowed or found
    if not found:
        return None, "figure not in a sentence"

    admitted = [t for t in found if rejection(t[0]) is None]
    if not admitted:
        return None, rejection(found[0][0]) or "rejected"
    if len(admitted) > 1:
        return None, "ambiguous"
    return admitted[0], "ok"


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


def build(rows: List[Dict[str, Any]]) -> tuple[List[dict], collections.Counter]:
    out, why = [], collections.Counter()
    seen = set()
    for src in rows:
        hit, reason = find_requirement(src)
        why[reason] += 1
        if not hit:
            continue
        sentence, start, end = hit
        if _squash(sentence) in seen:      # the same clause reached twice
            why["duplicate"] += 1
            continue

        # Without a subject the question names only a document, and a document
        # sets many requirements — the item would have several right answers.
        subject = subject_of(src.get("question_ko") or "")
        if not subject:
            why["no subject"] += 1
            continue
        seen.add(_squash(sentence))

        doc, clause = src.get("doc"), src.get("clause")
        where = f"「{doc}」" + (f" {clause}" if clause else "") if doc else "해당 기준"
        ctx = src.get("context")
        span = [start, end]

        out.append({
            "benchmark": src.get("benchmark", "kcbench"),
            "benchmark_version": src.get("benchmark_version"),
            "schema": src.get("schema"),
            "track": "usecase",
            "usecase": "uc7_requirement",
            "eval_type": "sentence",
            "cognitive_level": "application",
            "id": item_id("uc7_requirement", doc, clause, sentence),
            "doc": doc,
            "clause": clause,
            "category": src.get("category"),
            "split": _split_of(src),
            "lang": "ko",
            "context": ctx,
            "question_ko": (f"{where}에 따르면, {subject}에 관하여 무엇을 어떻게 "
                            f"하여야 하는가?"),
            "question_en": (f"According to {where}, what is required regarding "
                            f"{subject}?"),
            "instruction_ko": "조문에 적힌 대로 한 문장만 쓰고 다른 설명은 쓰지 마시오.",
            "instruction_en": "One sentence, as the clause words it. Nothing else.",
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
            "verified_ko": "조문에서 그대로 옮긴 요건 문장. 수치와 한정어로 위치를 특정함.",
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
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help="comma-separated item files to derive from")
    ap.add_argument("--out", default="uc7_requirement.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be mined and write nothing")
    args = ap.parse_args(argv)
    cfg = resolve_config(args)
    describe(cfg)

    data = Path(cfg["out_dir"])
    rows: List[dict] = []
    for name in [x.strip() for x in args.sources.split(",") if x.strip()]:
        p = data / name
        if not p.is_file():
            LOG.warning("no such source: %s", p.name)
            continue
        rows.extend(read_jsonl(p))
    if not rows:
        LOG.error("no source items")
        return 1

    items, why = build(rows)
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
    write_jsonl(data / args.out, items)
    LOG.info("wrote %s", data / args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
