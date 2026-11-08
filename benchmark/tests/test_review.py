"""
Tests for the review tab's backend: the queue listing, the context window,
the spreadsheet round trip, and that none of it opened a new write path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
openpyxl = pytest.importorskip("openpyxl")

from kcbench.review_sheet import export_xlsx, merge_xlsx  # noqa: E402
from kcbench.webview import Jobs, create_app  # noqa: E402


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    dirs = {}
    for name in ("corpus", "generated", "metadata", "out"):
        d = tmp_path / name
        d.mkdir()
        dirs[f"{name}_dir"] = d
    dirs["_config_path"] = str(tmp_path / "config.json")
    return dirs


@pytest.fixture()
def client(cfg):
    app = create_app(cfg, Jobs())
    app.testing = True
    return app.test_client()


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


QUEUE = [
    {"id": "a1", "eval_type": "numeric", "category": "01", "document": "d",
     "question_ko": "q1", "answer": "1 m", "context": "...", "reasons": ["x"],
     "verdict": "ok", "fixed_answer": "", "note": ""},
    {"id": "a2", "eval_type": "numeric", "category": "01", "document": "d",
     "question_ko": "q2", "answer": "2 m", "context": "...", "reasons": [],
     "verdict": "", "fixed_answer": "", "note": ""},
]


# ── queue listing ────────────────────────────────────────────────────────

def test_queues_finds_verdict_files_only(client, cfg):
    write_jsonl(cfg["out_dir"] / "review_queue.jsonl", QUEUE)
    write_jsonl(cfg["out_dir"] / "track2_sft.jsonl",
                [{"id": "t", "question_ko": "q"}])
    body = client.get("/api/review/queues").get_json()
    assert [q["file"] for q in body["queues"]] == ["review_queue.jsonl"]
    q = body["queues"][0]
    assert q["total"] == 2
    assert q["counts"] == {"ok": 1, "blank": 1}


# ── context window ───────────────────────────────────────────────────────

def test_context_returns_neighbours(client, cfg):
    chunks = cfg["generated_dir"] / "01" / "doc"
    chunks.mkdir(parents=True)
    write_jsonl(chunks / "rag_corpus.jsonl",
                [{"text": f"chunk {i}"} for i in range(5)])
    body = client.get("/api/review/context?path=01/doc/rag_corpus.jsonl"
                      "&row=2&around=1").get_json()
    assert [r["index"] for r in body["rows"]] == [1, 2, 3]
    assert [r["is_source"] for r in body["rows"]] == [False, True, False]
    assert body["rows"][1]["text"] == "chunk 2"


def test_context_refuses_escape_and_missing(client, cfg):
    assert client.get("/api/review/context?path=../config.json").status_code == 400
    assert client.get("/api/review/context?path=nope.jsonl").status_code == 404
    chunks = cfg["generated_dir"] / "c.jsonl"
    write_jsonl(chunks, [{"text": "only"}])
    assert client.get("/api/review/context?path=c.jsonl&row=9").status_code == 404


# ── spreadsheet round trip ───────────────────────────────────────────────

def test_sheet_export_and_merge(tmp_path):
    queue = tmp_path / "q.jsonl"
    write_jsonl(queue, QUEUE)
    xlsx = tmp_path / "q.xlsx"
    export_xlsx(QUEUE, xlsx)

    ws = openpyxl.load_workbook(xlsx)["review"]
    titles = [c.value for c in ws[1]]
    assert titles == ["no", "item id", "eval type", "category",
                      "source document", "question", "benchmark answer",
                      "source excerpt", "queued because", "verdict",
                      "fixed answer", "note"]
    assert ws.cell(2, 2).value == "a1"
    assert ws.cell(2, 10).value == "ok"

    # the expert fills the second row and the merge folds it back
    ws_wb = openpyxl.load_workbook(xlsx)
    ws2 = ws_wb["review"]
    ws2.cell(3, 10, "broken")
    ws2.cell(3, 11, "3 m")
    ws2.cell(3, 12, "wrong clause")
    ws_wb.save(xlsx)

    stats = merge_xlsx(queue, xlsx)
    assert stats == {"rows": 2, "judged_in_sheet": 2, "applied": 2}
    rows = [json.loads(ln) for ln in queue.read_text().splitlines()]
    assert rows[1]["verdict"] == "broken"
    assert rows[1]["fixed_answer"] == "3 m"
    assert queue.with_suffix(".jsonl.bak").exists()


def test_merge_rejects_unknown_verdict(tmp_path):
    queue = tmp_path / "q.jsonl"
    write_jsonl(queue, QUEUE)
    xlsx = tmp_path / "q.xlsx"
    export_xlsx(QUEUE, xlsx)
    wb = openpyxl.load_workbook(xlsx)
    wb["review"].cell(2, 10, "maybe")
    wb.save(xlsx)
    with pytest.raises(SystemExit):
        merge_xlsx(queue, xlsx)


def test_sheet_download(client, cfg):
    write_jsonl(cfg["out_dir"] / "review_queue.jsonl", QUEUE)
    r = client.get("/api/review/sheet?path=review_queue.jsonl")
    assert r.status_code == 200
    assert r.headers["Content-Disposition"].endswith('review_queue.xlsx')
    assert r.data[:2] == b"PK"  # a zip container, i.e. a real xlsx

    assert client.get("/api/review/sheet?path=../config.json").status_code == 400
    assert client.get("/api/review/sheet?path=nope.jsonl").status_code == 404


# ── the new routes stay read-only ────────────────────────────────────────

def test_review_routes_allowed_in_readonly(cfg):
    write_jsonl(cfg["out_dir"] / "review_queue.jsonl", QUEUE)
    app = create_app(cfg, Jobs(), readonly=True)
    app.testing = True
    ro = app.test_client()
    assert ro.get("/api/review/queues").status_code == 200
    assert ro.get("/api/review/sheet?path=review_queue.jsonl").status_code == 200
    # and the write path the tab uses stays closed
    assert ro.post("/api/record", json={}).status_code == 403
