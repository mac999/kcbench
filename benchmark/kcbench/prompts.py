#!/usr/bin/env python3
"""
Every prompt the benchmark sends to a model, in one place and overridable.

    from kcbench.prompts import render
    text = render(cfg, "qa.open.ko", passage=ctx, question=q, instruction=instr)

A prompt is domain-specific in the same way `config.json` already is. The
wording that suits Korean ministerial notices is not the wording for clinical
guidelines or contract law, and a reader who has to patch a string inside a
module to change it cannot keep a local variant across updates. So the
defaults live here and `config.json` may replace any of them by key:

    "prompts": {
      "judge.supported.ko": "[근거]\\n{gold}\\n\\n…"
    }

Two consequences are deliberate. An override is recorded in the run file, so a
score can be traced to the wording that produced it — prompt phrasing moved
this benchmark's own judge separation from 0.00 to 0.62, which is larger than
most effects it is used to measure. And a template that omits a field the
caller supplies is reported when it is used rather than failing silently, so a
typo in a key does not quietly fall back to the default.
"""
from __future__ import annotations

from typing import Any, Dict

from kcbench.common import log

LOG = log("prompts")

# Defaults. Keys read `<area>.<variant>.<lang>`; an area is one job, so a new
# language or a reworded variant is a new key rather than a code change.
DEFAULTS: Dict[str, str] = {
    # Scoring a keyed item. Open book attaches the clause the item was mined
    # from; closed book withholds it and names the document instead, so the
    # question stays answerable in principle from domain knowledge alone.
    "qa.open.ko": ("다음 조문을 읽고 질문에 답하시오.\n\n"
                   "[조문]\n{passage}\n\n[질문]\n{question}\n\n{instruction}"),
    "qa.open.en": ("Read the passage and answer the question.\n\n"
                   "[Passage]\n{passage}\n\n[Question]\n{question}\n\n{instruction}"),
    "qa.closed.ko": ("다음 질문에 답하시오.\n\n[질문]\n{question}\n\n{instruction}"),
    "qa.closed.en": ("Answer the question.\n\n[Question]\n{question}\n\n{instruction}"),

    # Retrieved context replaces the gold clause; everything else is identical
    # to open book so the two numbers stay comparable.
    "rag.ko": ("다음 조문을 읽고 질문에 답하시오.\n\n"
               "[조문]\n{passage}\n\n[질문]\n{question}\n\n{instruction}"),
    "rag.en": ("Read the passage and answer the question.\n\n"
               "[Passage]\n{passage}\n\n[Question]\n{question}\n\n{instruction}"),

    # Judging a prose answer. Asked in the negative ("is anything added?")
    # every judge tested rejected everything, valid paraphrases included —
    # 0.00 separation across all three conditions. Positive framing separates.
    # Reword with that measurement in mind.
    "judge.covered.ko": ("[기준]\n{gold}\n\n[답변]\n{pred}\n\n"
                         "답변이 기준의 내용을 빠짐없이 담고 있습니까? "
                         "yes 또는 no 한 단어로만 답하시오."),
    "judge.supported.ko": ("[근거]\n{gold}\n\n[문장]\n{pred}\n\n"
                           "문장에 담긴 내용이 모두 근거만으로 뒷받침됩니까? "
                           "근거에 없는 내용이 하나라도 있으면 no, 전부 뒷받침되면 yes. "
                           "한 단어로만 답하시오."),
    "judge.covered.en": ("[Reference]\n{gold}\n\n[Answer]\n{pred}\n\n"
                         "Does the answer contain everything the reference states? "
                         "Reply with one word, yes or no."),
    "judge.supported.en": ("[Evidence]\n{gold}\n\n[Statement]\n{pred}\n\n"
                           "Is everything the statement claims supported by the evidence "
                           "alone? If anything is not in the evidence, no; otherwise yes. "
                           "Reply with one word."),

    # Deciding whether a revision would change an item's answer. Used only for
    # the band the rules decline; the rules themselves are in volatility.py.
    "volatility.adjudicate.ko": (
        "[조문]\n{context}\n\n[질문]\n{question}\n\n[정답]\n{answer}\n\n"
        "이 정답은 소관 고시·기준이 개정되면 달라집니까? "
        "수치 한도·기간·비율처럼 개정으로 바뀌는 값이면 volatile, "
        "정의·원리·계산방법처럼 개정돼도 유지되는 내용이면 stable. "
        "한 단어로만 답하시오."),
}


def render(cfg: Dict[str, Any], key: str, **fields: Any) -> str:
    """
    The template for `key`, with `fields` substituted.

    Falls back to the built-in default when the config does not override it. A
    key that exists in neither is a programming error and raises, rather than
    sending an empty prompt to a model and scoring whatever comes back.
    """
    tpl = template(cfg, key)
    try:
        return tpl.format(**fields)
    except KeyError as exc:
        raise KeyError(
            f"prompt {key!r} uses {exc} but the caller did not supply it; "
            f"an override must keep the fields the default declares: "
            f"{sorted(_fields(DEFAULTS.get(key, '')))}") from exc


def template(cfg: Dict[str, Any], key: str) -> str:
    over = (cfg.get("prompts") or {}).get(key)
    if over is not None:
        return over
    if key not in DEFAULTS:
        raise KeyError(f"no prompt registered under {key!r}; "
                       f"known keys: {sorted(DEFAULTS)}")
    return DEFAULTS[key]


def _fields(tpl: str) -> set:
    import string
    return {f for _, f, _, _ in string.Formatter().parse(tpl) if f}


def overrides(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Which prompts this run did not use the default for.

    Recorded in the run file. A score produced under a reworded judge prompt is
    not comparable with one produced under the shipped wording, and the only
    way a reader can tell is if the run says so.
    """
    out = {}
    for k, v in (cfg.get("prompts") or {}).items():
        if k.startswith("_"):        # the section's own documentation
            continue
        if k not in DEFAULTS:
            LOG.warning("config overrides prompt %r, which nothing reads", k)
        if DEFAULTS.get(k) != v:
            missing = _fields(DEFAULTS.get(k, "")) - _fields(v)
            if missing:
                LOG.warning("prompt %r drops field(s) %s the default supplies",
                            k, sorted(missing))
            out[k] = v
    return out
