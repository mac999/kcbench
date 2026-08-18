#!/usr/bin/env python3
"""
Compare two runs — the whole point of this benchmark.

    python compare.py --base runs/base.json --after runs/sft-v1.json
    python compare.py --base base --after sft-v1 --markdown report.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List

from corpusbench.common import add_common_args, log, resolve_config

LOG = log("compare")

LOWER_IS_BETTER = ("perplexity", "no_answer")

# Resampling seed. Fixed so the same two runs always produce the same interval —
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_ROUNDS = 10000
ALPHA = 0.05


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p=0.5, for McNemar on small counts."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(pairs: List[tuple]) -> Dict[str, Any]:
    """
    The paired test for a binary metric on identical items.

    Comparing two aggregate accuracies treats the runs as independent samples,
    which they are not: they answered the same questions. Only the items whose
    verdict changed carry information about the change, and there are usually far
    fewer of those than the item count suggests — which is why an unpaired
    reading calls a two-point move significant when it is noise.
    """
    gained = sum(1 for b, a in pairs if a > b)
    lost = sum(1 for b, a in pairs if a < b)
    n = gained + lost
    return {"n_items": len(pairs), "changed": n, "gained": gained, "lost": lost,
            "net": gained - lost, "p_value": round(binom_two_sided(min(gained, lost), n), 5),
            "test": "mcnemar-exact"}


def paired_bootstrap(pairs: List[tuple], rounds: int = BOOTSTRAP_ROUNDS,
                     seed: int = BOOTSTRAP_SEED, alpha: float = ALPHA) -> Dict[str, Any]:
    """95% interval on the mean per-item change, for continuous metrics."""
    diffs = [a - b for b, a in pairs]
    if not diffs:
        return {"n_items": 0}
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(rounds):
        means.append(statistics.fmean(rng.choices(diffs, k=n)))
    means.sort()
    lo, hi = means[int(alpha / 2 * rounds)], means[int((1 - alpha / 2) * rounds)]
    return {"n_items": n, "mean_delta": round(statistics.fmean(diffs), 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            # An interval straddling zero means the data does not distinguish
            # this change from no change, whatever the point estimate says.
            "significant": lo > 0 or hi < 0, "test": "paired-bootstrap"}


def paired_metrics(base: dict, after: dict, cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Per-eval-type significance, matching items by id across the two runs."""
    out: Dict[str, Any] = {}
    for track, body in (after.get("tracks") or {}).items():
        b_items = {i["id"]: i for i in ((base.get("tracks") or {}).get(track) or {}).get("detail") or []}
        a_items = {i["id"]: i for i in (body.get("detail") or [])}
        shared = sorted(set(b_items) & set(a_items))
        if not shared:
            continue
        kinds: Dict[str, List[str]] = {}
        for i in shared:
            kinds.setdefault(a_items[i]["eval_type"], []).append(i)
        for kind, ids in sorted(kinds.items()):
            metric = next((m for m in ("correct", "f1", "key_f1")
                           if m in a_items[ids[0]]["score"]), None)
            if not metric:
                continue
            pairs = [(b_items[i]["score"].get(metric, 0.0), a_items[i]["score"].get(metric, 0.0))
                     for i in ids]
            binary = all(v in (0.0, 1.0) for p in pairs for v in p)
            c = (cfg or {}).get("compare", {})
            out[f"t{track}.{kind}.{metric}"] = mcnemar(pairs) if binary else paired_bootstrap(
                pairs, c.get("bootstrap_rounds", BOOTSTRAP_ROUNDS),
                c.get("bootstrap_seed", BOOTSTRAP_SEED), c.get("alpha", ALPHA))
    return out


def load_run(spec: str, runs_dir: Path) -> dict:
    p = Path(spec)
    if not p.exists():
        p = runs_dir / (spec if spec.endswith(".json") else spec + ".json")
    if not p.exists():
        raise SystemExit(f"run not found: {spec} (looked in {runs_dir})")
    return json.loads(p.read_text(encoding="utf-8"))


def delta(metric: str, base: float, after: float) -> Dict[str, Any]:
    lower = any(k in metric for k in LOWER_IS_BETTER)
    raw = after - base
    pct = (raw / base * 100) if base else 0.0
    return {"base": base, "after": after, "delta": round(raw, 4),
            "pct": round(-pct if lower else pct, 2),
            "improved": (raw < 0) if lower else (raw > 0)}


def collect(run: dict) -> Dict[str, float]:
    """Headline numbers plus every per-type metric, flattened."""
    flat: Dict[str, float] = dict(run.get("headline", {}))
    for track, body in run.get("tracks", {}).items():
        for kind, metrics in (body.get("by_type") or {}).items():
            for m, v in metrics.items():
                if m != "n":
                    flat[f"t{track}.{kind}.{m}"] = v
    return flat


def by_category(run: dict) -> Dict[str, float]:
    """Mean score per corpus category — shows where training actually landed."""
    acc: Dict[str, list] = {}
    for body in run.get("tracks", {}).values():
        for item in body.get("detail") or []:
            cat = item.get("category")
            if not cat:
                continue
            s = item["score"]
            acc.setdefault(cat, []).append(
                s.get("correct", s.get("f1", s.get("key_f1", 0.0))))
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(acc.items()) if v}


def render_paired(paired: Dict[str, Any], alpha: float = ALPHA) -> List[str]:
    if not paired:
        return []
    out = ["", "## Is the change real? / 유의성", "",
           "| Metric | n | Evidence | Verdict |", "|---|---:|---|---|"]
    for k, s in paired.items():
        if s.get("test") == "mcnemar-exact":
            sig = s["p_value"] < alpha
            ev = (f"{s['gained']} gained / {s['lost']} lost of {s['changed']} changed, "
                  f"p={s['p_value']}")
        else:
            sig = s.get("significant")
            ev = f"mean {s['mean_delta']:+.4f}, 95% CI [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}]"
        out.append(f"| {k} | {s.get('n_items', 0)} | {ev} | "
                   f"{'significant' if sig else 'not distinguishable from noise'} |")
    out += ["", "Items are the same in both runs, so the comparison is paired: only the items "
                "whose verdict changed carry information. An aggregate difference without this "
                "test can be entirely noise.",
            "",
            "두 실행은 동일 문항이므로 대응 비교입니다. 판정이 바뀐 문항만 정보를 가지며, "
            "이 검정 없이 집계 차이만 보면 노이즈를 개선으로 읽게 됩니다."]
    return out


def render(base: dict, after: dict, rows: Dict[str, Dict[str, Any]],
           cats: Dict[str, Dict[str, Any]], paired: Dict[str, Any] | None = None) -> str:
    out = [f"# Benchmark comparison — {base['tag']} → {after['tag']}", "",
           f"- base  / 기준 모델: `{base['model']}`",
           f"- after / 학습 모델: `{after['model']}`",
           f"- prompt language: {after.get('lang', 'ko')}, "
           f"{after.get('repeats')} sample(s) per item", "",
           "## Headline / 주요 지표", "",
           "| Metric | Base | After | Delta | % | Verdict |",
           "|---|---:|---:|---:|---:|---|"]
    for k, d in rows.items():
        mark = "better" if d["improved"] else ("same" if d["delta"] == 0 else "worse")
        out.append(f"| {k} | {d['base']} | {d['after']} | {d['delta']:+} | {d['pct']:+.1f}% | {mark} |")
    out += render_paired(paired or {})
    if cats:
        out += ["", "## By category / 분야별", "",
                "| Category | Base | After | Delta |", "|---|---:|---:|---:|"]
        for k, d in cats.items():
            out.append(f"| {k} | {d['base']} | {d['after']} | {d['delta']:+} |")
    out += ["", "## Reading this / 해석", "",
            "Absolute values are not comparable to published leaderboards — the items were "
            "mined by rule from held-out Korean regulation, not written and reviewed by "
            "engineers. The delta on identical frozen items is the measurement.",
            "",
            "절대값은 공개 리더보드와 비교할 수 없습니다. 동일한 고정 문항에 대한 "
            "학습 전후 차이가 이 벤치마크가 측정하는 값입니다."]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare two evaluation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("-b", "--base", required=True, help="run file or tag before fine-tuning")
    ap.add_argument("-a", "--after", required=True, help="run file or tag after fine-tuning")
    ap.add_argument("--runs-dir", help="where run files live (default <out-dir>/runs)")
    ap.add_argument("--markdown", metavar="FILE", help="also write a markdown report")
    args = ap.parse_args()

    cfg = resolve_config(args)
    runs_dir = Path(args.runs_dir) if args.runs_dir else cfg["out_dir"] / "runs"
    base, after = load_run(args.base, runs_dir), load_run(args.after, runs_dir)

    if base.get("benchmark_dir") != after.get("benchmark_dir"):
        LOG.warning("runs were scored against different benchmark directories - "
                    "the comparison is only valid on identical items")

    bf, af = collect(base), collect(after)
    rows = {k: delta(k, bf[k], af[k]) for k in sorted(bf) if k in af}
    only_after = sorted(set(af) - set(bf))
    if only_after:
        LOG.warning("present only in the 'after' run, not compared: %s", ", ".join(only_after))

    bc, ac = by_category(base), by_category(after)
    cats = {k: delta(k, bc[k], ac[k]) for k in sorted(bc) if k in ac}
    paired = paired_metrics(base, after, cfg)

    for k, d in rows.items():
        LOG.info("%-30s %8.4f → %8.4f  %+8.4f  (%+.1f%%) %s",
                 k, d["base"], d["after"], d["delta"], d["pct"],
                 "better" if d["improved"] else "worse")

    for k, st in paired.items():
        LOG.info("%-30s %s", k, st)

    report = render(base, after, rows, cats, paired)
    payload = {"base": base["tag"], "after": after["tag"],
               "headline": rows, "by_category": cats, "significance": paired}
    (runs_dir / f"compare_{base['tag']}_vs_{after['tag']}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
        LOG.info("wrote %s", args.markdown)
    else:
        print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
