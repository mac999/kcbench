#!/usr/bin/env python3
# Rebuild the derived training subsets an experiment file declares, bit-exactly,
# so any version of the comparison can be reproduced from train_star.jsonl and
# the experiment json alone. Methods:
#   shuffle_head   shuffle the parsed rows with the seed, keep the first n
#   cap_task_type  keep every row of other task types in source order; of the
#                  named task_type keep a seeded sample sized to the ratio
import argparse, json, random, sys
from pathlib import Path

def rebuild(dest: Path, spec: dict, base: Path) -> bool:
    src = (base / spec["source"]).resolve()
    rows = [json.loads(l) for l in src.open()]
    rng = random.Random(spec["seed"])
    if spec["method"] == "shuffle_head":
        rng.shuffle(rows)
        out = rows[:spec["n"]]
    elif spec["method"] == "cap_task_type":
        tt = spec["task_type"]
        ref = [i for i, r in enumerate(rows) if r["task_type"] == tt]
        other = len(rows) - len(ref)
        n = round(spec["ratio"] * other / (1 - spec["ratio"]))
        keep = set(rng.sample(ref, n))
        out = [r for i, r in enumerate(rows) if r["task_type"] != tt or i in keep]
    else:
        raise SystemExit(f"unknown method {spec['method']}")
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    dest = (base / dest).resolve()
    if dest.exists() and dest.read_text() == text:
        print(f"{dest.name}: unchanged ({len(out)} rows)")
        return True
    if dest.exists():
        print(f"{dest.name}: REBUILD DIFFERS from the existing file - not overwritten")
        dest.with_suffix(".rebuilt.jsonl").write_text(text)
        return False
    dest.write_text(text)
    print(f"{dest.name}: written ({len(out)} rows)")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", type=Path)
    args = ap.parse_args()
    exp = json.loads(args.experiment.read_text())
    base = args.experiment.resolve().parent.parent
    ok = all(rebuild(Path(d), s, base) for d, s in exp.get("datasets", {}).items())
    sys.exit(0 if ok else 1)
