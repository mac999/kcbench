#!/usr/bin/env python3
"""
Move a review queue between jsonl and a spreadsheet.

    python cb.py sheet --queue review_queue.jsonl --xlsx review.xlsx
    python cb.py sheet --queue review_queue.jsonl --merge reviewed.xlsx

The queue that `triage` writes is a jsonl file, which works for the webview's
review tab but not for a reviewer outside this repository. `--xlsx` renders
the queue as a workbook an external expert can fill in (column names in
English, since the reviewer may not read Korean), and `--merge` folds the
filled verdict / fixed answer / note columns back into the queue by item id.
`review` then applies the queue as usual.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from kcbench.common import log, read_jsonl

LOG = log("sheet")

VERDICTS = ("ok", "broken", "ambiguous")

XLSX_HINT = ("the spreadsheet step needs openpyxl, which the rest of the "
             "benchmark does not:\n    pip install openpyxl")

# queue field -> column header, in sheet order
COLUMNS = [
    ("_no",          "no"),
    ("id",           "item id"),
    ("eval_type",    "eval type"),
    ("category",     "category"),
    ("document",     "source document"),
    ("question_ko",  "question"),
    ("answer",       "benchmark answer"),
    ("context",      "source excerpt"),
    ("reasons",      "queued because"),
    ("verdict",      "verdict"),
    ("fixed_answer", "fixed answer"),
    ("note",         "note"),
]
EDITABLE = ("verdict", "fixed answer", "note")

GUIDE = [
    ("How to fill this sheet", ""),
    ("", ""),
    ("verdict", "one of: ok / broken / ambiguous (a dropdown is provided)"),
    ("  ok", "the benchmark answer follows from the source excerpt"),
    ("  broken", "the answer is wrong, incomplete, or cites the wrong clause"),
    ("  ambiguous", "the question has more than one defensible answer"),
    ("fixed answer", "only when the verdict is broken and the correct answer "
                     "is clear from the excerpt"),
    ("note", "one line on what you saw; required for broken and ambiguous"),
    ("", ""),
    ("Judge each item against its source excerpt only. The excerpt is copied "
     "verbatim from the regulation the item was mined from; the question is "
     "whether the benchmark answer follows from it.", ""),
]


def _cell(row: Dict[str, Any], field: str) -> Any:
    v = row.get(field)
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def export_xlsx(queue: List[Dict[str, Any]], out: Path) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError as exc:                 # noqa: PLC0415
        raise SystemExit(XLSX_HINT) from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    head = Font(name="Arial", size=10, bold=True)
    body = Font(name="Arial", size=10)
    wrap = Alignment(wrap_text=True, vertical="top")
    fill = PatternFill("solid", fgColor="FFF2CC")   # the cells to fill in

    for c, (_, title) in enumerate(COLUMNS, start=1):
        cell = ws.cell(1, c, title)
        cell.font = head
        if title in EDITABLE:
            cell.fill = fill
    for r, row in enumerate(queue, start=2):
        for c, (field, title) in enumerate(COLUMNS, start=1):
            cell = ws.cell(r, c, r - 1 if field == "_no" else _cell(row, field))
            cell.font = body
            cell.alignment = wrap
            if title in EDITABLE:
                cell.fill = fill

    dv = DataValidation(type="list", formula1='"' + ",".join(VERDICTS) + '"',
                        allow_blank=True, showErrorMessage=True)
    ws.add_data_validation(dv)
    v_col = get_column_letter([t for _, t in COLUMNS].index("verdict") + 1)
    dv.add(f"{v_col}2:{v_col}{len(queue) + 1}")

    widths = {"no": 5, "item id": 26, "eval type": 12, "category": 18,
              "source document": 28, "question": 40, "benchmark answer": 30,
              "source excerpt": 50, "queued because": 16, "verdict": 12,
              "fixed answer": 24, "note": 36}
    for c, (_, title) in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(c)].width = widths[title]
    ws.freeze_panes = "A2"

    guide = wb.create_sheet("guide")
    for r, (a, b) in enumerate(GUIDE, start=1):
        guide.cell(r, 1, a).font = head if r == 1 else body
        guide.cell(r, 2, b).font = body
        guide.cell(r, 1).alignment = wrap
        guide.cell(r, 2).alignment = wrap
    guide.column_dimensions["A"].width = 24
    guide.column_dimensions["B"].width = 80

    wb.save(out)


def export_bytes(queue: List[Dict[str, Any]]) -> bytes:
    """The same workbook, for a caller that serves it instead of saving it."""
    import io
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "queue.xlsx"
        export_xlsx(queue, out)
        return out.read_bytes()


def merge_xlsx(queue_path: Path, xlsx: Path) -> Dict[str, int]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:                 # noqa: PLC0415
        raise SystemExit(XLSX_HINT) from exc

    ws = load_workbook(xlsx, data_only=True)["review"]
    titles = [c.value for c in ws[1]]
    try:
        cols = {t: titles.index(t) for t in ("item id", "verdict",
                                             "fixed answer", "note")}
    except ValueError as exc:
        raise SystemExit(f"{xlsx.name} is missing a column: {exc}") from exc

    verdicts: Dict[str, Dict[str, str]] = {}
    bad: List[str] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        item_id = (row[cols["item id"]] or "").strip()
        if not item_id:
            continue
        v = str(row[cols["verdict"]] or "").strip()
        if v and v not in VERDICTS:
            bad.append(f"{item_id}: {v}")
            continue
        verdicts[item_id] = {
            "verdict": v,
            "fixed_answer": str(row[cols["fixed answer"]] or "").strip(),
            "note": str(row[cols["note"]] or "").strip(),
        }
    if bad:
        raise SystemExit("unknown verdict(s), expected "
                         + " / ".join(VERDICTS) + ":\n  " + "\n  ".join(bad))

    rows = read_jsonl(queue_path)
    applied = 0
    for r in rows:
        v = verdicts.get(r.get("id", ""))
        if v and v["verdict"]:
            r.update(v)
            applied += 1
    backup = queue_path.with_suffix(queue_path.suffix + ".bak")
    backup.write_bytes(queue_path.read_bytes())
    tmp = queue_path.with_suffix(queue_path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    tmp.replace(queue_path)
    return {"rows": len(rows), "judged_in_sheet": sum(1 for v in verdicts.values()
                                                      if v["verdict"]),
            "applied": applied}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Review queue to spreadsheet and back",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--queue", required=True, metavar="FILE",
                    help="the review_queue.jsonl to render or update")
    ap.add_argument("--xlsx", metavar="FILE",
                    help="write the queue as this workbook")
    ap.add_argument("--merge", metavar="FILE",
                    help="fold a filled workbook back into the queue")
    args = ap.parse_args()

    queue_path = Path(args.queue)
    if not queue_path.is_file():
        raise SystemExit(f"no such queue: {queue_path}")
    if bool(args.xlsx) == bool(args.merge):
        raise SystemExit("pass exactly one of --xlsx or --merge")

    if args.xlsx:
        queue = read_jsonl(queue_path)
        export_xlsx(queue, Path(args.xlsx))
        LOG.info("wrote %s - %d row(s)", args.xlsx, len(queue))
    else:
        stats = merge_xlsx(queue_path, Path(args.merge))
        LOG.info("merged %s into %s - %d verdict(s) applied over %d row(s) "
                 "(previous queue kept as .bak)", args.merge, queue_path.name,
                 stats["applied"], stats["rows"])
        LOG.info("next: python cb.py review --queue %s --apply", queue_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
