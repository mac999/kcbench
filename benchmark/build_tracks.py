#!/usr/bin/env python3
"""
Build the three evaluation tracks from the held-out slice.

    python build_tracks.py
    python build_tracks.py -i /data/ai_ready_v3 -o /tmp/bench --tracks 2,3
    python build_tracks.py --report rejects.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from common import (BENCHMARK_NAME, BENCHMARK_VERSION, QUALIFIER_EN,
                    SCHEMA_VERSION, UNIT_EN, add_common_args, describe,
                    generated_documents, item_id, log, normalise,
                    resolve_config, sha256_text, utc_now, write_json,
                    write_jsonl)

LOG = log("tracks")

UNITS = tuple(UNIT_EN)

# Regulations state a requirement as "<subject> ... <number> <unit> 이상/이하/…".
FACT_RE = re.compile(
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(" + "|".join(map(re.escape, UNITS)) + r")\s*(이상|이하|미만|초과|이내)")
CLAUSE_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*제\s*(\d+)\s*항)?")

# Enumerated lists. Arabic and Korean ordinals both appear, often in the same
# document, and the marker has to be captured so a broken sequence can be seen.
ENUM_ARABIC = re.compile(r"(?:^|\n)\s*(\d{1,2})\.\s*([^\n]{2,120})")
ENUM_HANGUL = re.compile(r"(?:^|\n)\s*([가나다라마바사아자차카타파하])\.\s*([^\n]{2,120})")
HANGUL_ORDER = "가나다라마바사아자차카타파하"

# The sentence that introduces a list. Korean statute is formulaic here, which is
LEAD_IN_RE = re.compile(
    r"([^\n.]{6,80}?(?:다음\s*각\s*호(?:의\s*어느\s*하나)?(?:에\s*해당하는\s*[^\n.]{0,20})?|"
    r"다음\s*각\s*목|다음과\s*같다|다음\s*사항|아래와\s*같다|다음\s*기준|"
    r"다음\s*방법|다음\s*내용|포함되어야\s*한다|포함하여야\s*한다))[^\n]{0,10}$")

# Failing an explicit lead-in, the clause heading that governs the list. Korean
HEADING_RE = re.compile(r"제\s*(\d+)\s*조\s*\(([^)]{2,40})\)")

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9()·\-]+")
# A token ending one of these connects to something that follows, so a subject
# ending there has been cut before its head noun.
STOP_END = ("하여", "되어", "에서", "부터", "까지", "따라", "대하여", "의하여", "위하여",
            "으로", "로서", "이고", "하고", "및", "또는", "경우", "때", "에는", "에도",
            "이상", "이하", "미만", "초과", "이내")
# Adnominal and particle endings: valid inside a phrase, never at its head.
TAIL_BAD = ("인", "한", "된", "는", "할", "될", "를", "을", "에", "와", "과", "로", "며", "고")
# A token ending in one of these is the tail of the *previous* clause. v1 kept
# them, which is how "있는 유량측정장치를 펌프 정격토출량은" became a question.
HEAD_BAD = ("를", "을", "에", "로", "며", "고", "서", "만", "도", "과", "와", "의",
            "은", "는", "이", "가")
HEAD_OK = ("관한", "대한", "위한", "따른", "의한", "관하여", "대하여")
STOP_TOK = {"는", "은", "이", "가", "을", "를", "의", "최대", "최소", "각", "그", "해당",
            "다음", "단", "또한", "위", "아래", *UNITS}
PARTICLE_RE = re.compile(r"(은|는|이|가|을|를|의)$")

# Cognitive levels, following AECBench's taxonomy so a reader coming from that
LEVEL = {"numeric": "memorization", "nameset": "understanding", "mapping": "understanding",
         "label": "understanding", "faithfulness": "grounding"}

# Answering instructions, given verbatim to the model under evaluation. Kept
# here rather than in the harness so the prompt is versioned with the items.
INSTRUCTION = {
    "numeric": {
        "ko": "주어진 조문에서 답을 찾아 숫자와 단위만 답하시오. 설명은 쓰지 마시오.",
        "en": "Answer with the number and its unit only, taken from the passage. No explanation.",
    },
    "nameset": {
        "ko": "해당하는 항목을 한 줄에 하나씩 나열하시오. 다른 설명은 쓰지 마시오.",
        "en": "List the applicable items, one per line. Nothing else.",
    },
    "mapping": {
        "ko": "각 항목과 그 개수를 'IfcWall: 12' 형식으로 한 줄에 하나씩 제시하시오.",
        "en": "Give each item and its count as 'IfcWall: 12', one per line.",
    },
}


def load_reviews(cfg) -> Dict[str, dict]:
    """
    Human verdicts from earlier builds, keyed by item id.

    The id hashes the question's content, so a verdict survives a rebuild for as
    long as the question does. Reviewing is the expensive step and the corpus
    keeps growing; a verdict that only applied to one build would have to be
    redone every time the dataset did.
    """
    path = cfg["out_dir"] / "reviews.json"
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8")).get("items") or {}
    dropped = sum(1 for v in items.values() if v.get("verdict") in ("broken", "ambiguous"))
    LOG.info("reviews.json: %d verdict(s) on file, %d item(s) will be excluded",
             len(items), dropped)
    return items


def reviewed_out(reviews: Dict[str, dict], ident: str, rej: "Rejects") -> bool:
    """True when a reviewer has ruled this item out."""
    verdict = (reviews.get(ident) or {}).get("verdict")
    if verdict in ("broken", "ambiguous"):
        rej(f"reviewer marked it {verdict}", ident)
        return True
    return False


class Rejects:
    """Why candidates were turned down, so a thin yield can be diagnosed."""

    def __init__(self) -> None:
        self.counts: collections.Counter = collections.Counter()
        self.samples: Dict[str, List[str]] = collections.defaultdict(list)

    def __call__(self, reason: str, sample: str = "") -> None:
        self.counts[reason] += 1
        if sample and len(self.samples[reason]) < 5:
            self.samples[reason].append(sample[:160])

    def report(self, label: str) -> dict:
        for reason, n in self.counts.most_common():
            LOG.info("  rejected - %-42s %5d", reason, n)
        return {"track": label, "counts": dict(self.counts), "samples": dict(self.samples)}


def josa(word: str, with_batchim: str, without: str) -> str:
    """
    Pick the Korean particle that agrees with the preceding syllable.

    '높이는' but '연면적은'. Writing '높이은(는)' everywhere would be a giveaway
    that the questions are machine-made, and it reads badly to the engineers who
    are supposed to sanity-check them.
    """
    if not word:
        return without
    last = word[-1]
    if not ("가" <= last <= "힣"):
        return without
    return with_batchim if (ord(last) - 0xAC00) % 28 else without


def subject_before(text: str) -> str:
    """
    The noun phrase immediately preceding a number.

    Walks backwards from the number and keeps tokens while they still look like
    part of one phrase. A regex anchored to the left instead grabs whatever
    happens to sit there — half a preceding clause, a stray parenthesis — and
    produces questions no one can answer.

    Unlike v1 this also rejects the phrase when its *first* token is the tail of
    the previous clause, rather than shipping it as the subject.
    """
    toks = text.split()
    out: List[str] = []
    for t in reversed(toks[-6:]):
        if not TOKEN_RE.fullmatch(t) or t in STOP_TOK or t.endswith(STOP_END):
            break
        out.insert(0, t)
        if len(out) == 4:
            break
    if not out or out[-1].endswith(TAIL_BAD):
        return ""
    while len(out) > 1 and out[0].endswith(HEAD_BAD) and not out[0].endswith(HEAD_OK):
        out = out[1:]
    if not out:
        return ""
    phrase = PARTICLE_RE.sub("", " ".join(out)).strip()
    # A list marker dragged in from the start of the line — "(2) 수직풍도의…".
    phrase = re.sub(r"^(?:\(?\d{1,2}[.)]|[①-⑮]|[가-하]\.)\s*", "", phrase).strip()
    if not any(len(t) >= 2 for t in phrase.split()):
        return ""
    return phrase


def subject_is_clean(phrase: str) -> str:
    """
    Why a subject phrase is unusable, or "" if it is fine.

    These are the shapes that produced unanswerable v1 questions. Each is cheap
    to detect and each one, left in, costs an item that a person holding the
    document could not answer either.
    """
    if phrase.count("(") != phrase.count(")"):
        return "subject has an unbalanced bracket"
    if re.search(r"[0-9][0-9,.]*\s*(?:" + "|".join(map(re.escape, UNITS)) + r")", phrase):
        return "subject swallowed another measurement"
    # Two topic-marked nouns means the phrase spans a clause boundary:
    # "옥내와의 차압은 최소차압" is two subjects, and the question asks about both.
    if re.search(r"(?:은|는)\s", phrase):
        return "subject spans two clauses"
    if re.search(r"(?:이상|이하|미만|초과|이내)", phrase):
        return "subject contains a qualifier from another threshold"
    # An adverb is not a subject: "유효하게" asks nothing. The question needs a
    # noun the reader can look up in the passage.
    if phrase.split()[-1].endswith(("게", "히", "서", "써", "라", "랑", "듯")):
        return "subject ends in an adverbial, not a noun"
    # A verb ending carried over from the previous clause — "졸업한 후", "취득한
    if re.search(r"(?:한|된|하는|되는|받은|취득한|졸업한)\s*(?:후|뒤|때|경우)?$", phrase):
        return "subject is a verb phrase carried over from the previous clause"
    # Table wreckage: a run of digits glued to words is a row of a table the
    # extractor flattened, not a sentence. "4우수 5내구성 경과년수 내용연수".
    if len(re.findall(r"(?<![0-9.])\d+(?=[가-힣])", phrase)) >= 2:
        return "subject is flattened table wreckage"
    # Leading punctuation means the phrase starts mid-token: "·정격하중".
    if phrase[0] in "·ㆍ,.;:)]}" or phrase.startswith(("및 ", "또는 ")):
        return "subject starts mid-token"
    return ""


def _instr(kind: str) -> dict:
    return {"instruction_ko": INSTRUCTION[kind]["ko"], "instruction_en": INSTRUCTION[kind]["en"]}


def _provenance(cfg, doc: dict, chunk: dict, span: tuple[int, int] | None = None) -> dict:
    """
    Where this item came from, in enough detail to check without asking.

    The generated file and row index locate it; the digest survives the file
    being rebuilt; the source PDF and its own digest reach past the generated
    dataset to the document itself.
    """
    prov = {
        "dataset": cfg["generated_dir"].name,
        "dataset_file": doc["dataset_file"],
        "row_index": chunk["row_index"],
        "chunk_sha256": chunk["sha256"],
        "generated_dir": doc["generated_dir"],
        "document": doc["stem"],
        "category": doc["category"],
        "source_name": doc.get("source_name"),
        "source_org": doc.get("source_org"),
        "source_date": doc.get("source_date"),
    }
    if span:
        prov["char_start"], prov["char_end"] = span
    return prov


def held_documents(cfg, holdout) -> list[dict]:
    """
    The held-out documents, with their chunk text, in a fixed order.

    Matching is on the folder path rather than the document name, because names
    repeat across categories and a name match would pull in documents the split
    never selected.
    """
    keys = {d["generated_dir"] for d in holdout["pdf_documents"]}
    docs = [d for d in generated_documents(cfg) if d["generated_dir"] in keys]
    missing = keys - {d["generated_dir"] for d in docs}
    if missing:
        LOG.warning("%d held-out document(s) named in holdout.json are not present in %s",
                    len(missing), cfg["generated_dir"].name)
    return sorted(docs, key=lambda d: d["generated_dir"])


def _base(track: str, kind: str, doc: dict) -> dict:
    return {
        "benchmark": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "schema": SCHEMA_VERSION,
        "track": track,
        "eval_type": kind,
        "cognitive_level": LEVEL[kind],
        "doc": doc["stem"],
        "category": doc["category"],
    }


# track 1

def build_track1(cfg, holdout) -> Path:
    """Held-out text for perplexity. One record per chunk, no labels needed."""
    rows = []
    for d in held_documents(cfg, holdout):
        for chunk in d["chunk_rows"]:
            text = chunk["text"]
            rows.append({
                **_base("dapt", "numeric", d),
                "id": item_id("t1", chunk["sha256"]),
                "eval_type": "perplexity",
                "cognitive_level": "language_modelling",
                "chunk_index": chunk["row_index"],
                "text": text,
                "chars": len(text),
                "provenance": _provenance(cfg, d, chunk),
            })
    out = write_jsonl(cfg["out_dir"] / "track1_dapt.jsonl", rows)
    LOG.info("track 1: %d chunk(s), %.2f M characters", len(rows),
             sum(r["chars"] for r in rows) / 1e6)
    return out


# track 2

def _numeric_candidates(text: str, rej: Rejects, cfg) -> List[dict]:
    """Every stipulated threshold in one passage that survives admission."""
    t2 = cfg["track2_sft"]
    found = []
    for m in FACT_RE.finditer(text):
        value, unit, qual = m.groups()
        # A four-digit year with 년 is a date, not a duration.
        if unit == "년" and re.fullmatch(r"(19|20)\d\d", value):
            rej("number is a calendar year, not a duration", m.group(0))
            continue
        found.append({"value": value, "unit": unit, "qualifier": qual,
                      "span": (m.start(), m.end()),
                      "subject": subject_before(text[:m.start()])})

    # Subject quality first. A phrase that is not a usable subject cannot
    usable = []
    for f in found:
        subject = f["subject"]
        if not subject:
            rej("no usable subject before the number", f"…{f['value']}{f['unit']} {f['qualifier']}")
            continue
        if not (t2["min_subject_len"] <= len(subject) <= t2["max_subject_len"]):
            rej("subject too short or too long", subject)
            continue
        why = subject_is_clean(subject)
        if why:
            rej(why, subject)
            continue
        usable.append(f)

    if t2.get("numeric_uniqueness", "strict") != "strict":
        counts = collections.Counter((f["subject"], f["unit"], f["qualifier"]) for f in usable)
        return [f for f in usable if counts[(f["subject"], f["unit"], f["qualifier"])] == 1]

    # Two thresholds may share a unit and a qualifier and still be separate
    out = []
    for f in usable:
        rivals = [g for g in usable
                  if g is not f and (g["unit"], g["qualifier"]) == (f["unit"], f["qualifier"])]
        clash = next((g for g in rivals
                      if normalise(g["subject"]) in normalise(f["subject"])
                      or normalise(f["subject"]) in normalise(g["subject"])), None)
        if clash:
            rej("a threshold with an overlapping subject shares this unit and qualifier",
                f"{f['subject']} / {clash['subject']}")
            continue
        out.append(f)
    return out


def build_track2(cfg, holdout, rej: Rejects) -> Path:
    """
    Questions whose answer is a string the document itself contains.

    Every candidate is verified by searching the answer back in the source text,
    so a mis-parsed number never reaches the benchmark. Items are capped per
    document so one long specification cannot dominate the set.
    """
    t2 = cfg["track2_sft"]
    want = t2["target_items"]
    per_doc_cap = t2.get("max_items_per_doc", 4)
    docs = held_documents(cfg, holdout)
    reviews = load_reviews(cfg)
    rows: List[dict] = []
    seen: set[tuple] = set()

    for d in docs:
        if len(rows) >= want:
            break
        made = 0
        for chunk in d["chunk_rows"]:
            if made >= per_doc_cap or len(rows) >= want:
                break
            text = chunk["text"]
            clause = CLAUSE_RE.search(text)
            for f in _numeric_candidates(text, rej, cfg):
                subject, value, unit, qual = f["subject"], f["value"], f["unit"], f["qualifier"]
                key = (subject, value, unit, qual)
                if key in seen:
                    rej("duplicate of an item already accepted", subject)
                    continue
                # The answer has to be findable in the passage exactly as keyed.
                answer_ko = f"{value} {unit}"
                if f"{value}{unit}" not in text.replace(" ", ""):
                    rej("answer not recoverable from the passage as written", answer_ko)
                    continue
                item = item_id("t2", chunk["sha256"], subject, value, unit, qual)
                if reviewed_out(reviews, item, rej):
                    continue
                seen.add(key)

                topic = josa(subject, "은", "는")
                rows.append({
                    **_base("sft", "numeric", d),
                    "id": item,
                    "lang": "ko",
                    "context": text,
                    "question_ko": f"{subject}{topic} 몇 {unit} {qual}이어야 하는가?",
                    # The subject is a legal term of art; translating it would
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
                    "clause": (f"제{clause.group(1)}조" + (f"제{clause.group(2)}항" if clause.group(2) else "")
                               if clause else None),
                    "verified_en": ("stated threshold; the only figure in this passage with this "
                                    "unit and qualifier, and present in the passage as keyed"),
                    "verified_ko": "본문에 명시된 기준값. 같은 지문에서 이 단위·한정어를 갖는 유일한 수치.",
                    "provenance": _provenance(cfg, d, chunk, f["span"]),
                })
                made += 1
                if made >= per_doc_cap or len(rows) >= want:
                    break

    numeric_n = len(rows)
    LOG.info("track 2: %d numeric item(s)", numeric_n)

    # Enumerations become nameset items, which is the answer type a scalar-only
    # benchmark misses and where models are known to degrade.
    for d in docs:
        if len(rows) >= want:
            break
        made = 0
        for n, chunk in enumerate(d["chunk_rows"]):
            if made or len(rows) >= want:
                break
            prev_tail = d["chunk_rows"][n - 1]["text"][-400:] if n else ""
            found = _nameset_candidate(chunk["text"], rej, cfg, prev_tail)
            if not found:
                continue
            lead, kind, items, span, prefix = found
            key = (lead, tuple(items))
            if key in seen:
                rej("duplicate enumeration", lead)
                continue
            item = item_id("t2", chunk["sha256"], lead)
            if reviewed_out(reviews, item, rej):
                continue
            seen.add(key)
            if kind == "heading":
                q_ko = f"조문 {lead}에서 정한 항목을 모두 나열하시오."
                q_en = f"List every item stipulated in {lead} of the passage."
            else:
                q_ko = f"조문에 따르면 {lead}에 해당하는 항목을 모두 나열하시오."
                q_en = f"According to the passage, list every item that falls under '{lead}'."
            rows.append({
                **_base("sft", "nameset", d),
                "id": item,
                "lang": "ko",
                "context": prefix + chunk["text"],
                "context_extended": bool(prefix),
                "question_ko": q_ko,
                "question_en": q_en,
                **_instr("nameset"),
                "answer": items,
                "answer_ko": items,
                "answer_lang": "ko",
                "lead_in": lead,
                "lead_in_kind": kind,
                "verified_en": "items parsed from a numbered list introduced by the passage itself",
                "verified_ko": "지문이 스스로 예고한 각 호 목록에서 추출한 항목.",
                "provenance": _provenance(cfg, d, chunk, span),
            })
            made += 1

    out = write_jsonl(cfg["out_dir"] / "track2_sft.jsonl", rows)
    kinds = collections.Counter(r["eval_type"] for r in rows)
    LOG.info("track 2: %d item(s) %s over %d document(s)",
             len(rows), dict(kinds), len({r["doc"] for r in rows}))
    return out


def _clean_item(raw: str) -> str:
    """One enumerated entry, trimmed to what a person would read out."""
    s = re.sub(r"\s+", " ", raw).strip()
    # A trailing fragment of the next sentence is common where the extractor
    # lost a line break; cut at the first sentence end rather than keeping it.
    s = re.split(r"(?<=다)\.\s|\.\s+(?=[가-힣])", s)[0]
    return s.strip(" .,;:··．")


def _nameset_candidate(text: str, rej: Rejects, cfg,
                       prev_tail: str = "") -> tuple[str, List[str], tuple[int, int], str] | None:
    """
    A numbered list, together with the sentence that introduced it.

    Both halves are required. The list alone gives a question with no subject —
    v1 used the chunk's first line, which is a mid-sentence fragment more often
    than not — and the lead-in alone has nothing to key.

    The lead-in may sit at the end of the previous chunk: the document is
    continuous and the chunk boundary is an artefact of how it was split for
    training. When it is found there it is prepended to the context, so the item
    a model sees still contains the sentence its question quotes.
    """
    t2 = cfg["track2_sft"]
    for pattern, ordered in ((ENUM_ARABIC, "arabic"), (ENUM_HANGUL, "hangul")):
        matches = list(pattern.finditer(text))
        if len(matches) < t2["nameset_min_items"]:
            continue

        # The markers have to run in order from the start of the list. A chunk
        run: List[Any] = []
        for m in matches:
            marker = m.group(1)
            pos = int(marker) if ordered == "arabic" else HANGUL_ORDER.index(marker) + 1
            if not run and pos != 1:
                continue
            if run and pos != run[-1][0] + 1:
                break
            run.append((pos, m))
        if len(run) < t2["nameset_min_items"]:
            rej("enumeration markers are not a run starting at 1", text[:80])
            continue

        first = run[0][1]
        lead, prefix = "", ""
        if t2.get("nameset_require_lead_in", True):
            own = text[:first.start()].rstrip()
            m = LEAD_IN_RE.search(own)
            if not m and prev_tail:
                m = LEAD_IN_RE.search((prev_tail + "\n" + own).rstrip())
                if m:
                    prefix = prev_tail.rstrip() + "\n"
            if m:
                lead = re.sub(r"\s+", " ", m.group(1)).strip()
                lead = re.sub(r"^제\s*\d+\s*조\s*(\([^)]*\))?\s*", "", lead)
                lead = re.sub(r"^[①-⑮\d]+\s*", "", lead).strip()
                # Drop the "…는 다음 각 호와 같다" tail: it announces the list, it
                lead = re.sub(r"\s*(?:은|는|이|가|에는|에게는)?\s*(?:다음|아래).*$", "", lead).strip()
                lead = re.sub(r"(?:은|는|이|가|의|에)$", "", lead).strip()
                kind = "sentence"
            else:
                head = None
                for h in HEADING_RE.finditer(prev_tail + "\n" + own):
                    head = h
                if not head:
                    rej("enumeration has no lead-in sentence saying what it lists",
                        own[-80:] if own else text[:80])
                    continue
                lead = f"제{head.group(1)}조({re.sub(r'\\s+', ' ', head.group(2)).strip()})"
                kind = "heading"
                if not own.strip():
                    prefix = prev_tail.rstrip() + "\n"
            if len(lead) < 6:
                rej("lead-in too short to identify the list", lead)
                continue

        # A list running to the end of the chunk was cut by the chunker, and its
        last = run[-1][1]
        if len(text) - last.end() < 4:
            rej("enumeration runs to the end of the chunk and is probably truncated",
                last.group(2)[-60:])
            continue

        items, bad = [], ""
        for _, m in run:
            s = _clean_item(m.group(2))
            if not (4 <= len(s) <= t2["nameset_max_item_len"]):
                bad = "an entry is too short or too long after cleaning"
                break
            if s.startswith("<") or re.match(r"^(삭제|<삭제)", s):
                bad = "list contains repealed entries"
                break
            items.append(s)
        if bad:
            rej(bad, "; ".join(items)[:120] or text[:80])
            continue
        if len(set(normalise(i) for i in items)) != len(items):
            rej("enumeration has duplicate entries after normalisation", "; ".join(items)[:120])
            continue
        return lead, kind, items, (first.start(), run[-1][1].end()), prefix
    return None


# track 3

def build_track3(cfg, holdout, rej: Rejects) -> Path:
    """
    IFC element questions. The catalogue written during generation is the key.

    The catalogue is per model, so only the whole-model render (group 0) has an
    exact answer; storey and space groups hold a subset the catalogue does not
    distinguish. Recording group membership in ifc_processor.py would lift this
    from ~36 items to several hundred.
    """
    t3 = cfg["track3_vlm"]
    reviews = load_reviews(cfg)
    rows = []
    for m in holdout["ifc_models"]:
        model, types = m["model"], m["element_types"]
        base = cfg["generated_dir"] / "40_bim_models_ifc" / m["category"] / model
        photos = sorted((base / "images/site_photo").glob("*.png"))
        if t3["whole_model_groups_only"]:
            photos = [p for p in photos if p.name.replace(model + "_", "").startswith("0_")]
        if not photos:
            rej("no whole-model render for this model", model)
            continue
        for p in photos:
            stem = p.name.replace("_site.png", "")
            prov = {
                "dataset": cfg["generated_dir"].name,
                "dataset_file": m.get("catalogue_file", f"40_bim_models_ifc/{m['category']}/{model}/bim_elements.json"),
                "catalogue_sha256": m.get("catalogue_sha256"),
                "generated_dir": f"40_bim_models_ifc/{m['category']}/{model}",
                "document": model,
                "category": m["category"],
                "source_ifc": m["source_ifc"],
                "element_total": m["element_total"],
            }
            common = {
                "doc": m["source_ifc"] or model, "model": model, "category": m["category"],
                "image": str(p.relative_to(cfg["generated_dir"])),
                "image_sha256": sha256_text(str(p.relative_to(cfg["generated_dir"]))),
                "bim_render": str((base / "images/bim_render" / f"{stem}.png")
                                  .relative_to(cfg["generated_dir"])),
                "ground_truth_source": f"{model}/bim_elements.json",
                "provenance": prov,
            }
            doc_like = {"stem": model, "category": m["category"]}
            rows.append({
                **_base("vlm", "nameset", doc_like),
                "id": item_id("t3", model, stem, "nameset"),
                "lang": "ko",
                "question_ko": "이 건설 현장 사진에 보이는 구조 요소의 종류를 IFC 명칭으로 모두 나열하시오.",
                "question_en": "List every structural element type visible in this "
                               "construction site photo, using IFC class names.",
                **_instr("nameset"),
                "answer": sorted(types),
                "answer_lang": "neutral",
                "verified_en": "element catalogue written from the IFC file during generation",
                "verified_ko": "생성 시 IFC 파일에서 추출한 요소 카탈로그.",
                **common,
            })
            total = m["element_total"]
            cap = t3.get("mapping_max_total_elements", 60)
            if len(types) < t3["min_types_for_mapping"]:
                continue
            if total > cap:
                rej(f"model holds more than {cap} elements — counting them from a photo "
                    f"is not a fair question", f"{model} ({total})")
                continue
            rows.append({
                **_base("vlm", "mapping", doc_like),
                "id": item_id("t3", model, stem, "mapping"),
                "lang": "ko",
                "question_ko": "이 사진의 건물 모델을 구성하는 구조 요소를 IFC 명칭과 종류별 개수로 제시하시오.",
                "question_en": "Give the structural elements of the building model in this photo "
                               "as IFC class names with a count for each.",
                **_instr("mapping"),
                "answer": dict(sorted(types.items())),
                "answer_lang": "neutral",
                "verified_en": "per-type counts from the same catalogue",
                "verified_ko": "동일 카탈로그의 종류별 개수.",
                **common,
            })
    out = write_jsonl(cfg["out_dir"] / "track3_vlm.jsonl", rows)
    kinds = collections.Counter(r["eval_type"] for r in rows)
    LOG.info("track 3: %d item(s) %s over %d model(s)",
             len(rows), dict(kinds), len({r["model"] for r in rows}))
    return out


BUILDERS = {"1": build_track1, "2": build_track2, "3": build_track3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the evaluation tracks",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--tracks", default="1,2,3", metavar="LIST",
                    help="which tracks to build, e.g. 2,3 (default all)")
    ap.add_argument("--holdout", metavar="FILE",
                    help="holdout.json to build from (default <out-dir>/holdout.json)")
    ap.add_argument("--report", metavar="FILE",
                    help="write the rejection tally here (default <out-dir>/rejections.json)")
    args = ap.parse_args(argv)

    cfg = resolve_config(args)
    describe(cfg)

    hp = Path(args.holdout) if args.holdout else cfg["out_dir"] / "holdout.json"
    if not hp.exists():
        LOG.error("run build_holdout.py first - %s is missing", hp)
        return 1
    holdout = json.loads(hp.read_text(encoding="utf-8"))

    wanted = [t.strip() for t in args.tracks.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in BUILDERS]
    if unknown:
        LOG.error("unknown track(s): %s - choose from 1, 2, 3", ", ".join(unknown))
        return 1

    reports = []
    for t in wanted:
        rej = Rejects()
        if t == "1":
            BUILDERS[t](cfg, holdout)
        else:
            BUILDERS[t](cfg, holdout, rej)
            reports.append(rej.report(f"track{t}"))

    path = Path(args.report) if args.report else cfg["out_dir"] / "rejections.json"
    write_json(path, {"built_at": utc_now(), "benchmark": BENCHMARK_NAME,
                      "version": BENCHMARK_VERSION, "reports": reports})
    LOG.info("rejection tally in %s", path)
    LOG.info("done - artifacts in %s", cfg["out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
