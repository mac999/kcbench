#!/usr/bin/env python3
"""
kcbench - build a benchmark from a document corpus, then score models on it.

    python cb.py build                                   build every stage
    python cb.py eval -m qwen3:8b -t base --tracks 2 --closed-book
    python cb.py compare --base base --after ft-v1

Each command takes its own flags; pass -h after the command to see them:

    python cb.py eval -h

The commands are thin wrappers around the modules in kcbench/, which can
also be run directly with python -m kcbench.evaluate if you prefer.
"""
from __future__ import annotations

import runpy
import sys

COMMANDS = {
    # build
    "build":    ("kcbench.build_all", "run every build stage in order"),
    "holdout":  ("kcbench.build_holdout", "choose the documents to withhold"),
    "tracks":   ("kcbench.build_tracks", "mine tracks 1-3 from held-out documents"),
    "probe":    ("kcbench.build_probe", "mine the probe set from trained-on documents"),
    "usecases": ("kcbench.build_usecases", "build the use-case tracks"),
    "split":    ("kcbench.make_train_split", "write the training split, holdout excluded"),
    "verify":   ("kcbench.verify_provenance", "trace every item to its source"),
    # score
    "eval":     ("kcbench.evaluate", "score a model over the generation tracks"),
    "ppl":      ("kcbench.perplexity", "score track 1 perplexity locally"),
    "compare":  ("kcbench.compare", "compare two runs, with a significance test"),
    "ece":      ("kcbench.calibration", "calibration error: is its confidence worth anything"),
    "selfcheck": ("kcbench.selfcheck", "hallucination signal from sampling consistency, no answer key"),
    "matrix":   ("kcbench.run_matrix", "score several models and tabulate"),
    # review and packaging
    "triage":   ("kcbench.triage_items", "pick the items a human should look at"),
    "review":   ("kcbench.apply_review", "fold review decisions back into the set"),
    "export":   ("kcbench.export_dataset", "package the built benchmark"),
}


def usage(out=sys.stdout) -> None:
    print(__doc__.strip(), file=out)
    print("\ncommands:", file=out)
    for name, (_, help_text) in COMMANDS.items():
        print(f"  {name:9s} {help_text}", file=out)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        usage()
        return 0
    name = sys.argv[1]
    if name not in COMMANDS:
        print(f"unknown command: {name}\n", file=sys.stderr)
        usage(sys.stderr)
        return 2

    module = COMMANDS[name][0]
    # argv[0] is what argparse prints in its usage line, so make it read like
    # the command the user actually typed.
    sys.argv = [f"cb.py {name}", *sys.argv[2:]]
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
