#!/usr/bin/env python3
"""
Collect US construction safety regulation from eCFR into the corpus layout.

    python scripts/collect_ecfr.py                      29 CFR 1926, one file per section
    python scripts/collect_ecfr.py --date 2025-01-01 --out ../data_us

Why this source. The Korean corpus is ministerial notices and technical
standards: numeric limits, obligation sentences, and a revision history
printed on the document. 29 CFR Part 1926 is the same shape in English --
"shall not exceed", "at least 30 inches", and a source credit line that names
every amendment. That correspondence is what makes it a test of whether the
mining rules port across languages rather than a second corpus for its own
sake.

The API requires compressed responses and serves a whole part in one request,
so this makes one call and splits locally rather than hammering per section.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml?part={part}"
# "[61 FR 5509, Feb. 13, 1996, as amended at 63 FR 3814, Jan. 27, 1998]" --
# the revision signal the volatility rules key on, in its US form.
AMEND = re.compile(r"as amended at|\bFR\b\s+\d+")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={
        "Accept-Encoding": "gzip",
        "User-Agent": "kcbench-corpus-collector/1.0 (research)",
    })
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    return gzip.decompress(raw) if r.headers.get("Content-Encoding") == "gzip" else raw


def text_of(sec: ET.Element) -> str:
    """Section body as running text, headings kept so a chunk reads in context."""
    parts = []
    head = (sec.findtext("HEAD") or "").strip()
    if head:
        parts.append(head)
    for el in sec.iter():
        if el.tag in ("P", "FP"):
            t = " ".join("".join(el.itertext()).split())
            if t:
                parts.append(t)
    return "\n".join(parts)


def credit_of(sec: ET.Element) -> str:
    return " ".join(" ".join("".join(c.itertext()).split())
                    for c in sec.findall(".//CITA"))


def safe(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name).strip("_")[:120]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="collect 29 CFR 1926 into the corpus layout")
    ap.add_argument("--title", default="29")
    ap.add_argument("--part", default="1926")
    ap.add_argument("--date", default="2025-01-01", help="eCFR snapshot date")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "train_data_us" / "data" / "01_osha_1926"))
    ap.add_argument("--min-chars", type=int, default=400,
                    help="skip sections too short to mine a question from")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    url = API.format(date=args.date, title=args.title, part=args.part)
    print("fetching", url)
    root = ET.fromstring(fetch(url))

    kept = skipped = 0
    index = []
    for sec in root.findall(".//DIV8"):
        num = (sec.get("N") or "").strip()
        head = (sec.findtext("HEAD") or "").strip()
        body = text_of(sec)
        if len(body) < args.min_chars:
            skipped += 1
            continue
        credit = credit_of(sec)
        # The credit line goes to the top as well as the bottom. A Korean
        # notice announces its amendments in the title, which is what the
        # router reads -- it scores the filename and the opening text. eCFR
        # prints the same information as a source credit at the end of the
        # section, so a document that is amended every other year looked
        # untouched: 92% routed to training against 12% for the Korean
        # corpus. Leading with it puts the signal where the router looks,
        # without inventing anything the document does not say.
        # Fold the credit into the opening line rather than standing it on its
        # own. As a separate short line it became its own chunk, and the
        # non-informative filter dropped it before the router ever saw it:
        # the signal was in the file and absent from chunk 0, so every amended
        # section scored 0.0 and routed to training.
        lines = body.split("\n", 1)
        head_line = lines[0] + (("  [Source: %s]" % credit) if credit else "")
        rest = ("\n" + lines[1]) if len(lines) > 1 else ""
        full = head_line + rest + (("\n\n" + credit) if credit else "")
        title = re.sub(r"^§\s*", "", head) or num
        fname = safe("%s %s" % (num, re.sub(r"^%s\s*" % re.escape(num), "", title))) + ".txt"
        (out / fname).write_text(full, encoding="utf-8")
        index.append({"section": num, "title": title, "file": fname,
                      "chars": len(full), "amended": bool(AMEND.search(credit)),
                      "credit": credit})
        kept += 1

    meta = Path(args.out).parent.parent / "metadata"   # train_data_us/metadata
    meta.mkdir(parents=True, exist_ok=True)
    cat = meta / ("catalog_ecfr_%s_%s.json" % (args.title, args.part))
    cat.write_text(json.dumps({"source": "eCFR", "title": args.title, "part": args.part,
                               "snapshot": args.date, "url": url,
                               "sections": index}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    amended = sum(1 for i in index if i["amended"])
    print("kept %d section(s), skipped %d under %d chars" % (kept, skipped, args.min_chars))
    print("  with an amendment credit: %d (%.0f%%)" % (amended, 100 * amended / max(kept, 1)))
    print("  wrote %s" % cat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
