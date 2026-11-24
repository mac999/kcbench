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
import functools
import re
from pathlib import Path
from typing import Any, Dict, List

from kcbench.common import (clause_label, clause_for, verified_note, BENCHMARK_NAME, BENCHMARK_VERSION, QUALIFIER_EN,
                    item_lang,
                    mining_vocab, qualifier_leads,
                            TRACKS_HELP, resolve_tracks,
                    SCHEMA_VERSION, UNIT_EN, add_common_args, describe,
                    generated_documents, item_id, log, nameset_match_mode, normalise,
                    resolve_config, sha256_text, utc_now, write_json,
                    write_jsonl)

LOG = log("tracks")

UNITS = tuple(UNIT_EN)

# Regulations state a requirement as "<subject> ... <number> <unit> 이상/이하/…".
FACT_RE = re.compile(
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(" + "|".join(map(re.escape, UNITS)) + r")\s*(이상|이하|미만|초과|이내)")


@functools.lru_cache(maxsize=8)
def _fact_re(units: tuple, quals: tuple, leads: bool):
    """
    The threshold pattern for one vocabulary and word order.

    Korean puts the qualifier after the figure ("30 cm 이상"), English before
    it ("at least 30 inches"). Both orders return (value, unit, qualifier) in
    that order so the caller does not branch.
    """
    num = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
    u = "(" + "|".join(map(re.escape, units)) + ")"
    q = "(" + "|".join(map(re.escape, quals)) + ")"
    if leads:
        # reordered to (value, unit, qualifier) by the caller
        return re.compile(q + r"\s+" + num + r"\s*" + u + r"\b", re.I), True
    return re.compile(num + r"\s*" + u + r"\s*" + q), False
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

# The four patterns above are Korean statute's own shapes. A federal rule
# enumerates as (a)(1)(iii), introduces a list with "shall include" rather
# than 다음 각 호, and heads a section with "§ 1926.501 Duty to have fall
# protection" -- so uc5 mined 0 items from 236 documents, every one rejected
# for a marker run that does not start at 1 or a missing lead-in. Each is a
# setting; the defaults are the Korean patterns unchanged.
ENUM_DEFAULTS = {
    "enum_primary": ENUM_ARABIC.pattern,
    "enum_secondary": ENUM_HANGUL.pattern,
    "secondary_order": HANGUL_ORDER,
    "lead_in": LEAD_IN_RE.pattern,
    "heading": HEADING_RE.pattern,
    # KCS/KDS-style numbered headings ("8.2.2 교육훈련계획"); off unless set
    "heading_alt": "",
    "clause": CLAUSE_RE.pattern,
    # Korean statute restarts its 각 호 numbering in every clause, so a list
    # that does not open at 1 is a fragment. A federal rule numbers straight
    # through a section, and a chunk cut from its middle legitimately opens at
    # (d) -- 567 of 771 candidates were rejected for exactly that.
    "require_start_at_one": True,
}


@functools.lru_cache(maxsize=8)
def _enum_set(frozen: tuple):
    over = dict(frozen)
    out = {}
    for k, default in ENUM_DEFAULTS.items():
        v = over.get(k, default)
        if k in ("secondary_order", "require_start_at_one"):
            out[k] = v
            continue
        if k == "heading_alt" and not v:
            out[k] = None
            continue
        try:
            out[k] = re.compile(v, re.M)
        except re.error as exc:
            LOG.error("enumeration.%s is not a valid regex (%s); using the default", k, exc)
            out[k] = re.compile(default, re.M)
    return out


def enum_patterns(cfg=None):
    """Enumeration markers, list lead-ins and clause headings for this corpus."""
    over = (cfg or {}).get("enumeration") or {}
    return _enum_set(tuple(sorted(over.items())))


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

# The lists above are Korean morphology: particles, adnominal endings, the
# tokens that end a phrase. English marks the same boundaries with function
# words in a different position -- a Korean threshold reads "<noun phrase>
# <number> <unit> 이상", an English one "<subject> shall be at least <number>
# <unit>", so walking back from the number lands on "shall be" rather than on
# the subject. 41 of 85 uc1 subjects came out as fragments ("capacity of",
# "step stools shall be") before this was configurable.
SUBJECT_DEFAULTS = {
    "direction": "before",          # before | sentence_subject
    "stop_tokens": sorted(STOP_TOK),
    "stop_endings": list(STOP_END),
    "tail_bad": list(TAIL_BAD),
    "head_bad": list(HEAD_BAD),
    "head_ok": list(HEAD_OK),
    "particle": PARTICLE_RE.pattern,
    "max_tokens": 4,
    "window": 6,
}


def subject_rules(cfg=None):
    over = (cfg or {}).get("subject") or {}
    out = dict(SUBJECT_DEFAULTS)
    out.update({k: v for k, v in over.items() if not k.startswith("_")})
    out["stop_tokens"] = set(out["stop_tokens"]) | set(UNITS)
    out["particle_re"] = re.compile(out["particle"]) if out["particle"] else None
    return out


# An English requirement names its subject before the modal: "Step stools
# shall be at least 8 inches". Taking the span from the start of the clause up
# to the modal gives the subject the question needs.
_MODAL = re.compile(
    r"(?<!\bthat )(?<!\bwhich )(?<!\bwho )(?<!\bwhere )(?<!\bwhen )"
    r"\b(shall|must|may|is|are|was|were|has|have|had)\b", re.I)
# A participial or relative clause opens a new predicate -- "a liquid
# having a vapor pressure", "equipment that has a rated capacity" -- and
# the subject the question needs stands before it, not inside it.
_CLAUSE_OPEN = re.compile(r"\b(that|which|who|whose|where|when|having|containing|used|intended|designed|installed|forming|meeting|exceeding)\b", re.I)


def subject_sentence(text: str, rules) -> str:
    """The clause subject: what stands before the first modal or copula."""
    tail = re.split(r"(?<=[.;:])\s+", text)[-1] if text else ""
    tail = re.sub(r"^\s*\(?[a-z0-9]{1,3}\)\s*", "", tail).strip()
    # Cut at whichever comes first: the main-clause modal, or the opener of a
    # relative/participial clause. Taking the modal alone left the relative
    # clause as the subject -- "requirements apply to equipment that".
    cuts = [m.start() for m in (_MODAL.search(tail), _CLAUSE_OPEN.search(tail)) if m]
    phrase = (tail[:min(cuts)] if cuts else tail).strip(" ,;:")
    toks = phrase.split()
    if len(toks) > rules["max_tokens"] + 2:
        toks = toks[-(rules["max_tokens"] + 2):]
    phrase = " ".join(toks)
    return phrase if len(phrase) >= 3 and any(len(t) > 2 for t in toks) else ""

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


def subject_is_clean(phrase: str, strict: bool = False) -> str:
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
    if strict and re.search(r"(?:지만|하나|으나|이며|면서|제외|포함)$", phrase):
        return "subject ends in a conjunction carried over from the previous clause"
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


def _instr(kind: str, cfg=None) -> dict:
    """The instruction pair, with the keyed slot holding the item language."""
    lang = item_lang(cfg)
    return {"instruction_ko": INSTRUCTION[kind][lang],
            "instruction_en": INSTRUCTION[kind]["en"]}


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


def _side_fields(cfg) -> dict:
    """The fixed tracks are mined from held-out documents only; say so on the
    item when the config asks, so every track carries a split like the use
    cases do."""
    return {"split": "holdout"} if (cfg.get("track2_sft") or {}).get("mark_split") else {}


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
    LOG.info("dapt: %d chunk(s), %.2f M characters", len(rows),
             sum(r["chars"] for r in rows) / 1e6)
    return out


# track 2

# What the expert review of the v0.5.3 keys found behind the defective
# threshold items, and the shape each one has in the text. A figure in a
# grading band ("2년 미만 0.5점, 4년 미만 1점") or a distance table is not a
# requirement; a figure that opens a condition ("70% 미만이면 수정을 요구할 수
# 있다") states when a rule applies, not what it demands; "할 수 있다" permits;
# "초과하지 않는" turns the qualifier round. Each check is a setting under
# track2_sft so the v0.5.3 build still reproduces with them off.
DEONTIC_RE = re.compile(
    r"(?:이어야|되어야|하여야|해야|할\s*것|않아야|않도록|아니\s*된다|아니된다|않는다|"
    r"넘지\s*않|초과하지\s*않|으로\s*한다|로\s*한다|되도록|확보|유지|설치|두어야|갖추어야|"
    r"이상이어야|이하이어야|않을\s*것|되게|하도록|따른다|한다|일\s*것|을\s*것|를\s*것|"
    r"안\s*된다|안된다|되어서는\s*안|금지)")
CONDITION_RE = re.compile(r"^\s*(?:이면|인\s*경우|일\s*때|이라면|의\s*경우|이고|이거나|인\s*때|인\s*(?!것)[가-힣])")
PERMISSIVE_RE = re.compile(r"(?:할\s*수\s*있|수\s*있다|수\s*있습니다|권장|바람직|할\s*수도)")
BAND_RE = re.compile(r"(?:평점|배점|등급|점수|평가표|평가기준표|가중치|점\b)")
# notice boilerplate; words like 서식 or 신청서 also occur inside requirement
# sentences ("별지 제6호서식에 따라 보고해야 한다") and are not form markers
FORM_RE = re.compile(r"(?:행정예고|보내실\s*곳|접수번호|접수일자|처리기간|전자우편|팩스)")
# a requirement expressed as a list entry carries its verb in the lead-in:
# "다음 각 호의 성능을 확보하여야 하며 ... 1. 비차열 1시간 이상"
LEAD_DEONTIC_RE = re.compile(r"(?:다음|아래)[^\n]{0,40}(?:각\s*호|각\s*목|사항|기준|성능|요건|같다|같이)")
VERB_END_RE = re.compile(r"(?:하는|되는|있는|없는|않는|따르는|관한|대한|위한|의한|인|한|된|할|될)$")
FORM_CHUNK_RE = re.compile(r"\[\s*별지\s*제?\s*\d+\s*호[^\]]{0,20}서식\s*\]|■\s*[^\n]{0,40}(?:서식|대장)")
NEGATED_RE = re.compile(r"^\s*(?:하지\s*않|않는|않아야|않도록|이\s*되어서는\s*아니|되어서는\s*아니|이\s*아닌)")
FLIP = {"초과": "이하", "미만": "이상", "이상": "미만", "이하": "초과", "이내": "초과"}


# A sentence boundary in this text layer: a sentence ending, or a list marker
# or bullet opening a new line. A bare line break is not one -- the PDF layer
# breaks lines mid-sentence ("초과하지\n아니하도록") and cutting there loses the
# predicate that says what the figure requires.
_SENT_BOUNDARY = re.compile(
    r"(?<=다)\.|(?<=것)\.|(?<=함)\.|(?<=음)\.|(?<=요)\.|[;；]|"
    r"\n\s*(?=\(\d{1,2}\)|\d{1,2}\.\s|\d+(?:\.\d+)+\s|[가-하]\.\s|[①-⑮]|[-•·▪◦]\s|제\s*\d+\s*조)")


def _sentence_at(text: str, start: int, end: int) -> tuple[str, int]:
    """The sentence holding [start, end), within 300 characters either way."""
    lo, hi = max(0, start - 300), min(len(text), end + 300)
    a = lo
    for m in _SENT_BOUNDARY.finditer(text, lo, start):
        a = m.end()
    m = _SENT_BOUNDARY.search(text, end, hi)
    b = m.end() if m else hi
    return text[a:b], a


def _threshold_admissible(text: str, f: dict, t2: dict, dense: bool = False) -> str:
    """Why this figure is not a stated requirement, or "" if it may be."""
    start, end = f["span"]
    sent, a = _sentence_at(text, start, end)
    after = text[end:end + 12]
    tail = sent[sent.find(f["value"]):] if f["value"] in sent else sent
    deontic = bool(DEONTIC_RE.search(tail))
    if t2.get("reject_conditions") and CONDITION_RE.match(after):
        return "figure opens a condition, not a limit"
    if t2.get("reject_forms") and (FORM_CHUNK_RE.search(text) or (FORM_RE.search(sent) and not deontic)):
        return "passage is a form or a notice, not a rule"
    if t2.get("reject_grading_bands"):
        if BAND_RE.search(sent):
            return "figure sits in a grading band or table row"
        # a flattened table is a run of short cells: many line breaks and no
        # sentence end anywhere near the figure
        if sent.count("\n") >= 6 and not deontic:
            return "figure sits in a table laid out as cells"
        # three figures with one unit and qualifier close together are a table
        # when no verb says what they require; "130mm 이상 150mm 이하여야 한다"
        # next to two more such sentences is three requirements
        if dense and not deontic:
            return "figure sits in a run of like figures with no requirement stated"
    if t2.get("require_requirement_sentence"):
        if PERMISSIVE_RE.search(tail) and not deontic:
            return "sentence permits or recommends, it does not require"
        if not deontic:
            # a list entry without a verb inherits the lead-in's
            lead = text[max(0, a - 250):a]
            if not (LEAD_DEONTIC_RE.search(lead) and DEONTIC_RE.search(lead)):
                return "sentence states no requirement after the figure"
    return ""


def _band_members(text: str, rx, leads, window: int = 150) -> set:
    """
    Positions of figures that sit in a run of three or more with one unit and
    qualifier within `window` characters: a flattened table or a grading band
    ("2년 미만 0.5 / 4년 미만 1.0 / 6년 미만 1.5"). Three such figures spread
    over a long clause are three requirements and are left alone.
    """
    hits = []
    for m in rx.finditer(text):
        g = m.groups()
        _, unit, qual = (g[1], g[2], g[0]) if leads else g
        hits.append((m.start(), unit, qual))
    out = set()
    for pos, unit, qual in hits:
        near = sum(1 for p, u, q in hits if (u, q) == (unit, qual) and abs(p - pos) <= window)
        if near >= 3:
            out.add(pos)
    return out


def _topic_subject(text: str, start: int) -> str:
    """
    For "개구부의 길이는 개구부 폭의 2배 이하": the thing bounded is the topic
    (길이), not the noun the ratio is taken of (폭). Walk back to the last
    topic-marked phrase in the sentence.
    """
    sent, a = _sentence_at(text, start, start)
    before = sent[:start - a]
    # the last topic-marked noun phrase; "담당하는", "아니하는" carry the same
    # particle letter as an adnominal ending and are skipped
    for m in reversed(list(re.finditer(r"([가-힣A-Za-z0-9·()\-\s]{2,40}?)(?:은|는)\s", before))):
        phrase = m.group(1).strip()
        phrase = re.sub(r"^(?:\(?\d{1,2}[.)]|[①-⑮]|[가-하]\.|\d+(?:\.\d+)+)\s*", "", phrase)
        phrase = phrase.split(",")[-1].strip()
        if not phrase or VERB_END_RE.search(phrase):
            continue
        head = subject_before(phrase + " ")
        if head and not subject_is_clean(head, True):
            return head
    return ""


def _numeric_candidates(text: str, rej: Rejects, cfg) -> List[dict]:
    """Every stipulated threshold in one passage that survives admission."""
    t2 = cfg["track2_sft"]
    units, quals = mining_vocab(cfg)
    rx, leads = _fact_re(tuple(units), tuple(quals), qualifier_leads(cfg))
    found = []
    bands = _band_members(text, rx, leads) if t2.get("reject_grading_bands") else set()
    for m in rx.finditer(text):
        g = m.groups()
        value, unit, qual = (g[1], g[2], g[0]) if leads else g
        # A four-digit year with 년 is a date, not a duration.
        if unit == "년" and re.fullmatch(r"(19|20)\d\d", value):
            rej("number is a calendar year, not a duration", m.group(0))
            continue
        if t2.get("flip_negated_qualifier") and NEGATED_RE.match(text[m.end():m.end() + 14]):
            if qual in FLIP:
                qual = FLIP[qual]
        rules = subject_rules(cfg)
        subj = (subject_sentence(text[:m.start()], rules)
                if rules["direction"] == "sentence_subject"
                else subject_before(text[:m.start()]))
        # "X의 2배", "X의 90%": the bounded quantity is the sentence topic
        if t2.get("ratio_subject_from_topic") and unit in ("배", "%") \
                and re.search(r"의\s*$", text[:m.start()][-12:]):
            topic = _topic_subject(text, m.start())
            if not topic:
                rej("ratio taken of another quantity and no topic names the bounded one", m.group(0))
                continue
            subj = topic
        f = {"value": value, "unit": unit, "qualifier": qual,
             "span": (m.start(), m.end()), "subject": subj}
        why = _threshold_admissible(text, f, t2, m.start() in bands)
        if why:
            rej(why, text[max(0, m.start() - 30):m.end() + 20].replace("\n", " "))
            continue
        found.append(f)

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
        why = subject_is_clean(subject, bool(t2.get("reject_conjunction_subjects")))
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


def clause_match(cfg, text: str, pos: int | None = None):
    """
    The clause an item at `pos` belongs to.

    The first 제N조 in a chunk is the wrong one whenever the chunk opens in the
    tail of one article and the figure sits in the next, or the reverse: three
    of the 25 reviewed uc7 v2 items cited the heading that followed their
    sentence. Under clause_position "before_figure" the last heading that
    starts before the figure wins, and a heading marked 삭제 is skipped.
    """
    pat = enum_patterns(cfg)["clause"]
    if pos is None or (cfg.get("track2_sft") or {}).get("clause_position", "first") == "first":
        return pat.search(text)
    last = None
    for m in pat.finditer(text, 0, pos):
        if re.match(r"\s*<?삭제", text[m.end():m.end() + 6]):
            continue
        last = m
    return last


def drop_conflicting_questions(rows: List[dict], t2: dict, rej: Rejects) -> List[dict]:
    """
    One question, two keyed answers, from two chunks of the same document
    ("기계 하중은 몇 배 이하": 3배 in one clause, 5배 in another). Whichever the
    model answers, one key marks it wrong. Under drop_conflicting_questions
    both go; off, the rows are returned as they were.
    """
    if not t2.get("drop_conflicting_questions"):
        return rows
    answers: Dict[str, set] = {}
    for r in rows:
        answers.setdefault(r.get("question_ko") or "", set()).add(json.dumps(r.get("answer"), ensure_ascii=False))
    bad = {q for q, a in answers.items() if len(a) > 1}
    kept = [r for r in rows if (r.get("question_ko") or "") not in bad]
    for q in bad:
        rej("the same question is keyed with different answers", q)
    return kept


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
            for f in _numeric_candidates(text, rej, cfg):
                clause = clause_match(cfg, text, f["span"][0])
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

                units, quals = mining_vocab(cfg)
                # The keyed question is written in the item language; the
                # Korean form needs a topic particle, the English one does not.
                en_order = item_lang(cfg) == "en"
                topic = "" if en_order else josa(subject, "은", "는")
                rows.append({
                    **_base("sft", "numeric", d),
                    **_side_fields(cfg),
                    "id": item,
                    "lang": item_lang(cfg),
                    "context": text,
                    "question_ko": (f"What is the stipulated threshold for '{subject}', "
                                    f"in {units.get(unit, unit)} ({quals.get(qual, qual)})?"
                                    if en_order else
                                    f"{subject}{topic} 몇 {unit} {qual}이어야 하는가?"),
                    # The subject is a legal term of art; translating it would
                    "question_en": f"What is the stipulated threshold for '{subject}', "
                                   f"in {units.get(unit, unit)} ({quals.get(qual, qual)})?",
                    **_instr("numeric", cfg),
                    "answer": answer_ko,
                    "answer_ko": answer_ko,
                    "answer_en": f"{value} {units.get(unit, unit)}",
                    "answer_value": float(value.replace(",", "")),
                    "answer_unit": unit,
                    "answer_unit_en": units.get(unit, unit),
                    "qualifier": qual,
                    "qualifier_en": quals.get(qual, qual),
                    "clause": clause_for(cfg, clause, d.get("stem")),
                    "verified_en": ("stated threshold; the only figure in this passage with this "
                                    "unit and qualifier, and present in the passage as keyed"),
                    "verified_ko": verified_note(cfg, "본문에 명시된 기준값. 같은 지문에서 이 단위·한정어를 갖는 유일한 수치.", "A figure stated in the passage, and the only one there with this unit and qualifier."),
                    "provenance": _provenance(cfg, d, chunk, f["span"]),
                })
                made += 1
                if made >= per_doc_cap or len(rows) >= want:
                    break

    numeric_n = len(rows)
    LOG.info("sft: %d numeric item(s)", numeric_n)

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
                q_ko = (f"List every item stipulated in {lead}."
                        if item_lang(cfg) == "en" else
                        f"조문 {lead}에서 정한 항목을 모두 나열하시오.")
                q_en = f"List every item stipulated in {lead} of the passage."
            else:
                q_ko = (f"List every item that falls under {lead}."
                        if item_lang(cfg) == "en" else
                        f"조문에 따르면 {lead}에 해당하는 항목을 모두 나열하시오.")
                q_en = f"According to the passage, list every item that falls under '{lead}'."
            rows.append({
                **_base("sft", "nameset", d),
                **_side_fields(cfg),
                "id": item,
                "lang": item_lang(cfg),
                "context": prefix + chunk["text"],
                "context_extended": bool(prefix),
                "question_ko": q_ko,
                "question_en": q_en,
                **_instr("nameset", cfg),
                "match_mode": nameset_match_mode(items),
                "answer": items,
                "answer_ko": items,
                "answer_lang": "ko",
                "lead_in": lead,
                "lead_in_kind": kind,
                "verified_en": "items parsed from a numbered list introduced by the passage itself",
                "verified_ko": verified_note(cfg, "지문이 스스로 예고한 각 호 목록에서 추출한 항목.", "Entries taken from a list the passage itself announces."),
                "provenance": _provenance(cfg, d, chunk, span),
            })
            made += 1

    rows = drop_conflicting_questions(rows, t2, rej)
    out = write_jsonl(cfg["out_dir"] / "track2_sft.jsonl", rows)
    kinds = collections.Counter(r["eval_type"] for r in rows)
    LOG.info("sft: %d item(s) %s over %d document(s)",
             len(rows), dict(kinds), len({r["doc"] for r in rows}))
    return out


def _clean_item(raw: str, cut_headings: bool = False) -> str:
    """One enumerated entry, trimmed to what a person would read out."""
    s = re.sub(r"\s+", " ", raw).strip()
    # A trailing fragment of the next sentence is common where the extractor
    # lost a line break; cut at the first sentence end rather than keeping it.
    s = re.split(r"(?<=다)\.\s|\.\s+(?=[가-힣])", s)[0]
    if cut_headings:
        s = re.split(r"(?:\s|(?<=[가-힣]))제\s*\d+\s*(?:장|절)\s", s)[0]
    return s.strip(" .,;:··．")


def _nameset_candidate(text: str, rej: Rejects, cfg,
                       prev_tail: str = "",
                       over: dict | None = None) -> tuple[str, List[str], tuple[int, int], str] | None:
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
    t2 = {**cfg["track2_sft"], **(over or {})}
    ep = enum_patterns(cfg)
    skip_terms = [w for w in (t2.get("nameset_skip_terms") or []) if w]
    for pattern, ordered in ((ep["enum_primary"], "arabic"),
                             (ep["enum_secondary"], "secondary")):
        matches = list(pattern.finditer(text))
        if len(matches) < t2["nameset_min_items"]:
            continue

        # The markers have to run in order from the start of the list. A chunk
        run: List[Any] = []
        for m in matches:
            marker = m.group(1)
            if ordered == "arabic":
                pos = int(marker)
            else:
                order = ep["secondary_order"]
                pos = (order.index(marker) + 1) if marker in order else -1
                if pos < 0:
                    break
            if not run and ep["require_start_at_one"] and pos != 1:
                continue
            if run and pos != run[-1][0] + 1:
                break
            run.append((pos, m))
        if len(run) < t2["nameset_min_items"]:
            rej("enumeration markers are not a consecutive run", text[:80])
            continue

        # The run above stops at the first marker out of sequence and keeps the
        # prefix. When the next marker exists further down -- the text layer
        # glued "4." onto the previous line, or a sub-list intervened -- that
        # prefix is a truncated list, and ten of the 25 reviewed uc5 items were
        # keyed on one. Under nameset_reject_gaps the list is refused instead.
        if t2.get("nameset_reject_gaps"):
            nxt = run[-1][0] + 1
            if ordered == "arabic":
                probe = re.compile(r"(?<![\d.])" + str(nxt) + r"\.\s")
            else:
                order = ep["secondary_order"]
                probe = (re.compile(r"(?<![가-힣])" + re.escape(order[nxt - 1]) + r"\.\s")
                         if nxt <= len(order) else None)
            # search from the start of the last entry: the text layer glues the
            # next marker onto the previous line ("...할 것4. 송수구에는")
            if probe and probe.search(text, run[-1][1].start() + 2):
                rej("list continues after a displaced marker; the captured run is a prefix",
                    text[run[-1][1].start():run[-1][1].start() + 60])
                continue

        first = run[0][1]
        lead, prefix = "", ""
        if t2.get("nameset_require_lead_in", True):
            own = text[:first.start()].rstrip()
            window = int(t2.get("nameset_lead_in_window") or 0)
            # The sentence that announces a list is the one right before it;
            # a lead-in found further up belongs to an earlier list.
            near = own[-window:] if window else own
            m = ep["lead_in"].search(near)
            if not m and prev_tail:
                joined = (prev_tail + "\n" + own).rstrip()
                m = ep["lead_in"].search(joined[-window:] if window else joined)
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
                # A 가나다 list is a sub-list of one 호; the article heading does
                # not say which, so under nameset_heading_for_secondary=false
                # only an explicit lead-in admits it.
                if ordered == "secondary" and not t2.get("nameset_heading_for_secondary", True):
                    rej("sub-list without its own lead-in", own[-80:] if own else text[:80])
                    continue
                head, alt = None, False
                for h in ep["heading"].finditer(prev_tail + "\n" + own):
                    head = h
                if ep.get("heading_alt"):
                    for h in ep["heading_alt"].finditer(prev_tail + "\n" + own):
                        if head is None or h.start() > head.start():
                            head, alt = h, True
                if not head:
                    rej("enumeration has no lead-in sentence saying what it lists",
                        own[-80:] if own else text[:80])
                    continue
                if alt:
                    lead = f"{head.group(1)} " + re.sub(r"\s+", " ", head.group(2)).strip()
                else:
                    lead = clause_label(cfg, head.group(1), None,
                                        re.sub(r"\s+", " ", head.group(2)).strip())
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
        for n, (_, m) in enumerate(run):
            if t2.get("nameset_join_lines"):
                # the entry runs to the next marker, however many lines the text
                # layer broke it into; a heading that follows the list is cut off
                stop = run[n + 1][1].start() if n + 1 < len(run) else _entry_end(text, m.end())
                raw = text[m.start(2):stop]
            else:
                raw = m.group(2)
            s = _clean_item(raw, bool(t2.get("nameset_join_lines")))
            if not (4 <= len(s) <= t2["nameset_max_item_len"]):
                bad = "an entry is too short or too long after cleaning"
                break
            if skip_terms and any(w in s for w in skip_terms):
                bad = "entry carries a term the track excludes"
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
        if skip_terms and any(w in lead for w in skip_terms):
            rej("lead-in carries a term the track excludes", lead)
            continue
        return lead, kind, items, (first.start(), run[-1][1].end()), prefix
    return None


_ENTRY_STOP = re.compile(r"\n\s*\n|\n\s*제\s*\d+\s*(?:장|절|조)\b|\n\s*\d+(?:\.\d+)+\s|\n\s*[①-⑮]")


def _entry_end(text: str, start: int) -> int:
    """Where the last entry of a list ends: a blank line, a heading or the chunk."""
    m = _ENTRY_STOP.search(text, start)
    return m.start() if m else len(text)


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
                **_side_fields(cfg),
                "id": item_id("t3", model, stem, "nameset"),
                "lang": item_lang(cfg),
                "question_ko": "이 건설 현장 사진에 보이는 구조 요소의 종류를 IFC 명칭으로 모두 나열하시오.",
                "question_en": "List every structural element type visible in this "
                               "construction site photo, using IFC class names.",
                **_instr("nameset", cfg),
                "match_mode": nameset_match_mode(sorted(types)),
                "answer": sorted(types),
                "answer_lang": "neutral",
                "verified_en": "element catalogue written from the IFC file during generation",
                "verified_ko": verified_note(cfg, "생성 시 IFC 파일에서 추출한 요소 카탈로그.", "An element catalogue extracted from the IFC file at generation time."),
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
                **_side_fields(cfg),
                "id": item_id("t3", model, stem, "mapping"),
                "lang": item_lang(cfg),
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
    LOG.info("vlm: %d item(s) %s over %d model(s)",
             len(rows), dict(kinds), len({r["model"] for r in rows}))
    return out


BUILDERS = {"1": build_track1, "2": build_track2, "3": build_track3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the evaluation tracks",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--tracks", default="1,2,3", metavar="LIST",
                    help=f"which tracks to build, e.g. 2,3 (default all). {TRACKS_HELP}")
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

    wanted = resolve_tracks(args.tracks)
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
