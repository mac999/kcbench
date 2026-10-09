"""Repair the headline block of finished run files.

The per-track summary is derived from `by_type`, which every run already
carries, so a defect in the summary layer is repairable without spending GPU
hours again. By default only entries recorded as 0.0 are rewritten and no key
is added or dropped: `rag` runs name their tracks differently, and a wholesale
recompute would silently rename them. Prints what changed, writes on --apply.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kcbench.common import resolve_config        # noqa: E402
from kcbench.evaluate import headline           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--all-keys", action="store_true",
                    help="rewrite every entry, not just the ones stored as 0.0. "
                         "Renames tracks on rag runs; use on eval runs only")
    a = ap.parse_args()
    cfg = resolve_config(argparse.Namespace(config=a.config))
    for p in a.runs:
        d = json.loads(p.read_text())
        before = d.get("headline", {})
        after = headline(d, cfg)
        if a.all_keys:
            keys = set(before) | set(after)
        else:
            keys = {k for k in before if before[k] == 0.0 and k in after}
        diff = {k: (before.get(k), after.get(k)) for k in keys
                if before.get(k) != after.get(k)}
        print(f"{p.name}: {len(diff)} changed")
        for k, (b, v) in sorted(diff.items()):
            print(f"  {k:<28} {b} -> {v}")
        if a.apply and diff:
            d["headline"] = dict(before)
            d["headline"].update({k: after[k] for k in diff})
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
