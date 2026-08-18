#!/usr/bin/env python3
"""
Build the whole benchmark: holdout, the three tracks, then the use-case tracks.

    python build_all.py
    python build_all.py -i /data/ai_ready_v3 -o /tmp/bench --config my.json
    python build_all.py --skip-holdout --tracks 2,3
"""
from __future__ import annotations

import argparse
import runpy
import sys

from kcbench import build_holdout
from kcbench import build_tracks
from kcbench.common import add_common_args, log

LOG = log("build")


def _run(module: str, argv: list[str]) -> int:
    """
    Run a stage that only has a main() taking sys.argv.

    Cheaper than a subprocess and it keeps one log stream, which matters when a
    five-stage build fails at stage four and someone has to read why.
    """
    saved = sys.argv
    sys.argv = [module.rsplit(".", 1)[-1] + ".py", *argv]
    try:
        runpy.run_module(module, run_name="__main__")
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = saved


def _shared(args: argparse.Namespace) -> list[str]:
    """Rebuild the common flags as an argument list for each stage."""
    out: list[str] = []
    for flag, key in (("--config", "config"), ("--corpus-dir", "corpus_dir"),
                      ("--generated-dir", "generated_dir"), ("--metadata-dir", "metadata_dir"),
                      ("--pipeline-dir", "pipeline_dir"), ("--out-dir", "out_dir")):
        val = getattr(args, key, None)
        if val:
            out += [flag, str(val)]
    if args.seed is not None:
        out += ["--seed", str(args.seed)]
    if args.verbose:
        out.append("--verbose")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the full benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--fraction", type=float, help="holdout share per category")
    ap.add_argument("--tracks", default="1,2,3", help="tracks to build")
    ap.add_argument("--mirror", metavar="FILE",
                    help="where to mirror holdout.json for run_corpus.py")
    ap.add_argument("--skip-holdout", action="store_true",
                    help="reuse the existing holdout.json instead of re-splitting")
    ap.add_argument("--skip-probe", action="store_true",
                    help="do not rebuild the probe set")
    ap.add_argument("--skip-usecases", action="store_true",
                    help="do not rebuild the use-case tracks")
    ap.add_argument("--skip-split", action="store_true",
                    help="do not rebuild the training split")
    ap.add_argument("--skip-verify", action="store_true",
                    help="do not check provenance and contamination")
    ap.add_argument("--skip-export", action="store_true",
                    help="do not write the manifest, card and harness files")
    ap.add_argument("--strict", action="store_true",
                    help="fail the build if any contamination check fails")
    args = ap.parse_args()

    shared = _shared(args)

    if args.skip_holdout:
        LOG.info("stage 1/7: holdout - reusing the existing split")
    else:
        LOG.info("stage 1/7: holdout")
        extra = []
        if args.fraction:
            extra += ["--fraction", str(args.fraction)]
        if args.mirror:
            extra += ["--mirror", args.mirror]
        if build_holdout.main(shared + extra) != 0:
            return 1

    LOG.info("stage 2/7: tracks")
    if build_tracks.main(shared + ["--tracks", args.tracks]) != 0:
        return 1

    # The split has to be rebuilt before verification, not after: the check that
    if args.skip_probe:
        LOG.info("stage 3/7: probe set - skipped")
    else:
        LOG.info("stage 3/7: probe set")
        if _run("kcbench.build_probe", shared) != 0:
            return 1

    # After the probe: uc4 rewrites track 2 items, so track 2 must exist, and
    # the use-case files are not inputs to the split or the verifier.
    if args.skip_usecases:
        LOG.info("stage 4/7: use-case tracks - skipped")
    else:
        LOG.info("stage 4/7: use-case tracks")
        if _run("kcbench.build_usecases", shared) != 0:
            return 1

    if args.skip_split:
        LOG.info("stage 5/7: training split - skipped")
    else:
        LOG.info("stage 5/7: training split")
        if _run("kcbench.make_train_split", shared) != 0:
            return 1

    if args.skip_verify:
        LOG.info("stage 6/7: provenance - skipped")
    else:
        LOG.info("stage 6/7: provenance and contamination")
        rc = _run("kcbench.verify_provenance", shared + ["--tracks", args.tracks]
                  + (["--strict"] if args.strict else []))
        if rc != 0:
            return rc

    if args.skip_export:
        LOG.info("stage 7/7: export - skipped")
    else:
        LOG.info("stage 7/7: export")
        if _run("kcbench.export_dataset", shared) != 0:
            return 1

    LOG.info("build complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
