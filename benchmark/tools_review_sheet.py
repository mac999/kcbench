"""Turn a review queue into a spreadsheet a domain expert can fill in, and back.

The queue is JSONL carrying model replies and the reason each item was flagged.
A reviewer who can see that an item is there because three models failed it
will read it differently from one drawn at random, and the random rows are what
the error-rate estimate rests on -- so by default the sheet hides both and
shuffles the order. --unblind keeps them, for triaging rather than estimating.

    python tools_review_sheet.py export -c config_us.json
    python tools_review_sheet.py import -c config_us.json --sheet reviewed.csv
    python cb.py review -c config_us.json --queue ../ground_truth_us/reviewed.jsonl --report
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kcbench.common import log, read_jsonl, resolve_config   # noqa: E402

LOG = log("sheet")

COLUMNS = ["id", "track", "eval_type", "category", "document", "clause",
           "question", "key_answer", "context", "verdict", "fixed_answer", "note"]
BLIND_EXTRA = ["reasons", "model_replies", "model_scores"]
SEED = 20260814
INSTRUCTIONS = (
    "verdict: ok | broken | ambiguous. ok = the key answers the question from the "
    "context shown. broken = the key is wrong, or the question cannot be answered "
    "from the context. ambiguous = a specialist could defend either reading. "
    "fixed_answer: fill in only when verdict is broken and the right answer is "
    "clear. Leave verdict empty for a row you did not judge."
)


def settings(cfg):
    s = (cfg or {}).get("review_sheet") or {}
    return (list(s.get("columns") or COLUMNS),
            str(s.get("delimiter", ",")),
            bool(s.get("blind", True)),
            int(s.get("seed", SEED)))


def _row(rec, columns):
    out = {}
    for c in columns:
        if c == "track":
            out[c] = rec.get("usecase") or str(rec.get("id", "")).split("-")[0]
        elif c == "question":
            out[c] = rec.get("question_ko") or rec.get("question") or ""
        elif c == "key_answer":
            out[c] = rec.get("answer", "")
        elif c in ("verdict", "fixed_answer", "note"):
            out[c] = rec.get(c) or ""
        else:
            v = rec.get(c)
            out[c] = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else (v or "")
    return out


def do_export(cfg, args):
    queue = cfg["out_dir"] / "review_queue.jsonl"
    rows = list(read_jsonl(queue))
    columns, delim, blind, seed = settings(cfg)
    if args.unblind:
        blind = False
    if not blind:
        columns = columns + [c for c in BLIND_EXTRA if c not in columns]
    else:
        random.Random(seed).shuffle(rows)
    out = Path(args.sheet) if args.sheet else cfg["out_dir"] / "review_sheet.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, delimiter=delim)
        w.writeheader()
        for r in rows:
            w.writerow(_row(r, columns))
    LOG.info("wrote %s: %d row(s), %d column(s), %s",
             out, len(rows), len(columns), "blinded and shuffled" if blind else "unblinded")
    LOG.info("%s", INSTRUCTIONS)
    return 0


def do_import(cfg, args):
    if not args.sheet:
        raise SystemExit("--sheet is required for import")
    queue = cfg["out_dir"] / "review_queue.jsonl"
    by_id = {r["id"]: r for r in read_jsonl(queue)}
    _, delim, _, _ = settings(cfg)
    filled = unknown = 0
    with Path(args.sheet).open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter=delim):
            rec = by_id.get((row.get("id") or "").strip())
            if rec is None:
                unknown += 1
                continue
            v = (row.get("verdict") or "").strip().lower()
            if not v:
                continue
            rec["verdict"] = v
            rec["fixed_answer"] = (row.get("fixed_answer") or "").strip()
            rec["note"] = (row.get("note") or "").strip()
            filled += 1
    out = Path(args.out) if args.out else cfg["out_dir"] / "reviewed.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in by_id.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if unknown:
        LOG.warning("%d sheet row(s) carried an id not in the queue; ignored", unknown)
    LOG.info("wrote %s: %d of %d row(s) carry a verdict", out, filled, len(by_id))
    LOG.info("next: cb.py review --queue %s --report", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--sheet")
    ap.add_argument("--out", help="import: where the merged queue goes")
    ap.add_argument("--unblind", action="store_true",
                    help="export: keep the flag reasons and model replies")
    a = ap.parse_args()
    cfg = resolve_config(argparse.Namespace(config=a.config))
    return do_export(cfg, a) if a.action == "export" else do_import(cfg, a)


if __name__ == "__main__":
    raise SystemExit(main())
