#!/usr/bin/env python3
"""
Pick and constrain the models that grade free-form answers.

A judge that shares a family with the model under test is grading its own
relatives, and the score stops being a measurement of the answer. The rule is
therefore enforced here rather than left to whoever writes the command line:
ask for a panel, name the model being scored, and a same-family judge is
dropped with a warning instead of quietly counting.

    from kcbench.judges import panel_for, family
    judges = panel_for("qwen3-sft:v4")     # -> the qwen-free panel

Three things follow from treating this as code rather than convention. The
panel is recorded in the run file, so a number can always be traced to who
produced it. A verdict needs agreement, so one judge's idiosyncrasy does not
decide an item. And the prompt polarity is fixed in one place: asking "is
anything unsupported here" collapses to answering yes for everything, while
asking "is all of this supported" separates — measured, not assumed.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

from kcbench.common import log

LOG = log("judges")

# Model names carry their lineage in the first token; a fine-tune keeps the
# base's family whatever it is renamed to, which is exactly the case that
# matters here (qwen3-sft and qwen3-dapt are both qwen).
_FAMILY = re.compile(r"^([a-z]+?)(?:[0-9._-]|$)")

# Aliases where the name does not give the family away.
FAMILY_ALIASES = {
    "exaone": "exaone", "glm": "glm", "gemma": "gemma",
    "mistral": "mistral", "mixtral": "mistral", "llama": "llama",
    "qwen": "qwen", "phi": "phi", "deepseek": "deepseek",
}

# Judges in preference order. Korean-native first: the corpus is Korean
# regulation and a judge that reads it natively disagrees with the answer key
# for fewer uninteresting reasons.
CANDIDATES = ["exaone3.5:7.8b", "glm4:9b", "gemma2:9b", "mistral-nemo:12b"]

# Three is the smallest panel where a majority is not unanimity. With two, one
# judge's refusal sinks an item, and the measured pass rate reflects the
# strictest member rather than the answer.
PREFERRED_PANEL = 3

# Below this the panel cannot outvote a single judge's quirk, so a run that
# ends up here says so rather than reporting a clean number.
MIN_PANEL = 2


def family(model: str) -> str:
    """The lineage a model name implies, lowercased."""
    name = model.split("/")[-1].split(":")[0].lower()
    m = _FAMILY.match(name)
    stem = m.group(1) if m else name
    return FAMILY_ALIASES.get(stem, stem)


def panel_for(target: str, available: Sequence[str] | None = None,
              candidates: Sequence[str] | None = None) -> List[str]:
    """
    The judges that may score `target`, in preference order.

    Anything sharing the target's family is excluded. What is left is filtered
    against what the server actually has, because a panel that names a model
    nobody pulled is a panel of one pretending to be three.
    """
    banned = family(target)
    pool = list(candidates if candidates is not None else CANDIDATES)
    keep = [m for m in pool if family(m) != banned]
    dropped = [m for m in pool if family(m) == banned]
    for m in dropped:
        LOG.warning("judge %s shares family %r with %s; dropped", m, banned, target)
    if available is not None:
        have = {a.split(":")[0]: a for a in available}
        present, missing = [], []
        for m in keep:
            if m in available:
                present.append(m)
            elif m.split(":")[0] in have:
                present.append(have[m.split(":")[0]])
            else:
                missing.append(m)
        for m in missing:
            LOG.warning("judge %s is not installed; not in the panel", m)
        keep = present
    if len(keep) < MIN_PANEL:
        LOG.warning("panel of %d judge(s) for %s: below the %d needed to outvote "
                    "a single judge, so report it as provisional",
                    len(keep), target, MIN_PANEL)
    elif len(keep) < PREFERRED_PANEL:
        LOG.warning("panel of %d judge(s) for %s: a majority here is unanimity, "
                    "so the pass rate tracks the strictest judge. Pull one more "
                    "family from %s", len(keep), target,
                    [m for m in CANDIDATES if family(m) != banned and m not in keep])
    return keep


def prompt_for(cfg, kind: str, lang: str = "ko") -> str:
    """
    The judge template, from the config when it overrides one.

    Asked in the negative ("is anything added?") every judge tested rejected
    everything, valid paraphrases included — 0.00 separation across all three
    conditions. Positive framing separates. A local rewording is legitimate,
    which is why this reads the registry, but it should be re-measured against
    the same conditions before its scores are believed.
    """
    from kcbench.prompts import render, template
    return template(cfg, f"judge.{kind}.{lang}")


def vote(answers: Sequence[bool]) -> Dict[str, float]:
    """
    Majority, with the split reported.

    `agreement` is what says whether the majority meant anything: three judges
    at 2-1 and three at 3-0 are not the same evidence, and collapsing them to
    one boolean hides which one you have.
    """
    n = len(answers)
    if not n:
        return {"value": 0.0, "agreement": 0.0, "n_judges": 0}
    yes = sum(1 for a in answers if a)
    return {"value": float(yes * 2 > n),
            "agreement": max(yes, n - yes) / n,
            "n_judges": n}
