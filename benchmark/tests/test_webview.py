"""
Endpoint tests for the browser view.

These pin the behaviour the page relies on - what each route returns, that a
crafted path cannot leave its root, that read-only closes every write - so a
change to webview.py that breaks one of them fails here instead of in a
browser session.

    python -m pytest tests/ -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

from kcbench.webview import Jobs, create_app, safe_path  # noqa: E402


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    dirs = {}
    for name in ("corpus", "generated", "metadata", "out"):
        d = tmp_path / name
        d.mkdir()
        dirs[f"{name}_dir"] = d
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    dirs["_config_path"] = str(config_path)
    return dirs


@pytest.fixture()
def client(cfg):
    app = create_app(cfg, Jobs())
    app.testing = True
    return app.test_client()


@pytest.fixture()
def ro_client(cfg):
    app = create_app(cfg, Jobs(), readonly=True)
    app.testing = True
    return app.test_client()


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


# ── state ────────────────────────────────────────────────────────────────

def test_state_lists_commands_and_roots(client):
    body = client.get("/api/state").get_json()
    assert "eval" in body["commands"]
    assert set(body["roots"]) >= {"corpus", "generated", "metadata", "out"}
    assert body["readonly"] is False
    assert body["job"] is None


def test_state_reports_readonly(ro_client):
    assert ro_client.get("/api/state").get_json()["readonly"] is True


# ── path confinement ─────────────────────────────────────────────────────

def test_safe_path_refuses_escape(cfg):
    with pytest.raises(ValueError):
        safe_path(cfg, "out", "../corpus/secret.txt")
    with pytest.raises(ValueError):
        safe_path(cfg, "nonsense", "x")


def test_ls_refuses_escape(client):
    r = client.get("/api/ls?root=out&path=../corpus")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_file_refuses_escape_and_unknown_root(client):
    assert client.get("/api/file?root=out&path=../config.json").status_code == 400
    assert client.get("/api/file?root=bogus&path=x").status_code == 400


def test_raw_refuses_escape(client):
    assert client.get("/api/raw?root=out&path=../config.json").status_code == 400


# ── reading ──────────────────────────────────────────────────────────────

def test_ls_lists_files(client, cfg):
    (cfg["out_dir"] / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (cfg["out_dir"] / "sub").mkdir()
    body = client.get("/api/ls?root=out").get_json()
    assert [d["name"] for d in body["dirs"]] == ["sub"]
    assert [f["name"] for f in body["files"]] == ["a.jsonl"]


def test_file_pages_jsonl_records(client, cfg):
    write_jsonl(cfg["out_dir"] / "t.jsonl", [{"n": i} for i in range(30)])
    body = client.get("/api/file?root=out&path=t.jsonl&offset=25&limit=10").get_json()
    assert body["kind"] == "records"
    assert body["total"] == 30
    assert [r["n"] for r in body["records"]] == [25, 26, 27, 28, 29]


def test_file_missing_is_404(client):
    assert client.get("/api/file?root=out&path=nope.jsonl").status_code == 404


def test_tracks_counts_items(client, cfg):
    write_jsonl(cfg["out_dir"] / "track2_sft.jsonl", [
        {"id": "a", "question_ko": "q1", "eval_type": "numeric", "split": "holdout"},
        {"id": "b", "question_ko": "q2", "eval_type": "numeric", "split": "holdout"},
    ])
    # provenance-style rows carry no question and must not be counted
    write_jsonl(cfg["out_dir"] / "provenance.jsonl", [{"id": "a", "eval_type": "numeric"}])
    rows = client.get("/api/tracks").get_json()["tracks"]
    assert [(r["file"], r["n"]) for r in rows] == [("track2_sft.jsonl", 2)]


# ── read-only closes every write ─────────────────────────────────────────

def test_readonly_refuses_writes(ro_client):
    assert ro_client.put("/api/config", json={"text": "{}"}).status_code == 403
    assert ro_client.post("/api/record", json={}).status_code == 403
    assert ro_client.post("/api/run", json={"command": "eval"}).status_code == 403
    assert ro_client.post("/api/stop", json={"job": "x"}).status_code == 403


# ── config save ──────────────────────────────────────────────────────────

def test_config_save_keeps_bak_and_rejects_bad_json(client, cfg):
    path = Path(cfg["_config_path"])
    assert client.put("/api/config", json={"text": "not json"}).status_code == 400

    body = client.put("/api/config", json={"text": "{\"a\": 1}"}).get_json()
    assert body["saved"] is True
    assert path.read_text(encoding="utf-8") == "{\"a\": 1}"
    assert path.with_suffix(".json.bak").read_text(encoding="utf-8") == "{}"


# ── record editing ───────────────────────────────────────────────────────

def test_record_save_append_delete(client, cfg):
    path = cfg["out_dir"] / "queue.jsonl"
    write_jsonl(path, [{"id": "a", "verdict": ""}, {"id": "b", "verdict": ""}])

    r = client.post("/api/record", json={
        "root": "out", "path": "queue.jsonl", "action": "save",
        "index": 1, "record": {"id": "b", "verdict": "ok"}})
    assert r.get_json()["saved"] is True
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert rows[1] == {"id": "b", "verdict": "ok"}
    assert path.with_suffix(".jsonl.bak").exists()

    r = client.post("/api/record", json={
        "root": "out", "path": "queue.jsonl", "action": "append",
        "record": {"id": "c"}})
    assert r.get_json()["total"] == 3

    r = client.post("/api/record", json={
        "root": "out", "path": "queue.jsonl", "action": "delete", "index": 0})
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert [x["id"] for x in rows] == ["b", "c"]


def test_record_refuses_bad_requests(client, cfg):
    path = cfg["out_dir"] / "queue.jsonl"
    write_jsonl(path, [{"id": "a"}])
    post = lambda body: client.post("/api/record", json=body).status_code  # noqa: E731
    assert post({"root": "out", "path": "../queue.jsonl", "action": "save",
                 "index": 0, "record": {}}) == 400          # escape
    assert post({"root": "out", "path": "missing.jsonl", "action": "append",
                 "record": {}}) == 400                       # not a file
    assert post({"root": "out", "path": "queue.jsonl", "action": "save",
                 "index": 5, "record": {}}) == 400           # no such record
    assert post({"root": "out", "path": "queue.jsonl", "action": "rename"}) == 400


# ── job control ──────────────────────────────────────────────────────────

def test_run_refuses_unknown_command(client):
    assert client.post("/api/run", json={"command": "rm"}).status_code == 400


def test_run_refuses_while_one_is_running(cfg):
    class Busy:
        command = "eval"

        def describe(self):
            return {"command": "eval"}

    jobs = Jobs()
    jobs.current = lambda: Busy()  # type: ignore[method-assign]
    app = create_app(cfg, jobs)
    app.testing = True
    r = app.test_client().post("/api/run", json={"command": "eval"})
    assert r.status_code == 409
    assert "one command at a time" in r.get_json()["error"]


def test_stop_unknown_job_is_404(client):
    assert client.post("/api/stop", json={"job": "zzz"}).status_code == 404
