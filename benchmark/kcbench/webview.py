#!/usr/bin/env python3
"""
Browse the benchmark in a browser.

    python cb.py webview
    python cb.py webview --port 8800 --no-browser
    python cb.py webview --config my.json

Serves a local page that runs the same `cb.py` commands you would type, streams
their log while they run, and renders what they wrote: mined items as records,
corpus text as text, renders and site photos as images, and a run file as a
scored dashboard.

The page is a reader and a launcher, not a second implementation. Every number
it draws is read back out of a run file that the ordinary command wrote, so
nothing here can disagree with what `cb.py` reports.

It writes in exactly two places, both of them explicit: starting a command, and
saving `config.json` (which keeps a `.bak` of what it replaced). Everything
else is read-only, and paths are confined to the directories the config names.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

from kcbench.common import (HERE, PROJECT, add_common_args, describe, log,
                            resolve_config)

LOG = log("webview")

FLASK_HINT = (
    "the webview needs Flask, which the rest of the benchmark does not:\n"
    "    pip install flask\n"
    "Everything else keeps working without it."
)

# One job at a time, deliberately. Scoring loads a model, `ppl` loads a second
# copy of one locally, and on a unified-memory box those add up against
# whatever else is resident -- which is how training runs on this project died.
# The queue is one deep so the page cannot start that by accident.
MAX_LOG_LINES = 4000

# What the left and right panels are allowed to show. Anything outside these
# roots is refused, so a crafted path cannot walk out of the benchmark.
def roots_for(cfg: Dict[str, Any]) -> Dict[str, Path]:
    return {
        "corpus": Path(cfg["corpus_dir"]),
        "generated": Path(cfg["generated_dir"]),
        "metadata": Path(cfg["metadata_dir"]),
        "out": Path(cfg["out_dir"]),
        "doc": PROJECT / "doc",
    }


INPUT_ROOTS = ("corpus", "generated", "metadata")
OUTPUT_ROOTS = ("out", "doc")

TEXT_SUFFIXES = {".md", ".txt", ".py", ".sh", ".cfg", ".ini", ".yaml", ".yml",
                 ".csv", ".log", ".toml"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}

# The form the top panel draws for each command. Fields are the flags worth
# clicking; anything rarer goes in the free-text box the page always offers, so
# this table never has to keep up with every argparse option.
COMMAND_SPECS: Dict[str, Dict[str, Any]] = {
    "build": {"group": "build", "fields": [
        {"flag": "--strict", "type": "flag",
         "help": "fail the build on a contaminated item instead of warning"}]},
    "holdout": {"group": "build", "fields": []},
    "tracks": {"group": "build", "fields": [
        {"flag": "--tracks", "type": "text", "default": "1,2,3"}]},
    "probe": {"group": "build", "fields": []},
    "usecases": {"group": "build", "fields": []},
    "split": {"group": "build", "fields": [
        {"flag": "--check", "type": "flag",
         "help": "report contamination and exit without writing"}]},
    "verify": {"group": "build", "fields": [
        {"flag": "--tracks", "type": "text", "default": "1,2,3"},
        {"flag": "--strict", "type": "flag"}]},

    "eval": {"group": "score", "fields": [
        {"flag": "-m", "type": "text", "required": True, "placeholder": "qwen3:8b",
         "label": "model"},
        {"flag": "--tag", "type": "text", "placeholder": "base"},
        {"flag": "--tracks", "type": "text", "default": "sft"},
        {"flag": "--closed-book", "type": "flag",
         "help": "withhold the passage, so the item tests what the weights hold"},
        {"flag": "--lang", "type": "select", "options": ["", "ko", "en", "mix"]},
        {"flag": "--think", "type": "select", "options": ["", "on", "off"]},
        {"flag": "--limit", "type": "number", "placeholder": "smoke test: 20"}]},
    "ppl": {"group": "score", "fields": [
        {"flag": "-m", "type": "text", "required": True,
         "placeholder": "Qwen/Qwen3-8B", "label": "model"},
        {"flag": "--tag", "type": "text"},
        {"flag": "--limit", "type": "number"}]},
    "rag": {"group": "score", "fields": [
        {"flag": "-m", "type": "text", "required": True, "label": "model"},
        {"flag": "--tag", "type": "text"},
        {"flag": "--tracks", "type": "text", "default": "sft"},
        {"flag": "--limit", "type": "number"}]},
    "ece": {"group": "score", "fields": [
        {"flag": "-m", "type": "text", "required": True, "label": "model"},
        {"flag": "--tag", "type": "text"},
        {"flag": "--tracks", "type": "text", "default": "sft"},
        {"flag": "--closed-book", "type": "flag"}]},
    "selfcheck": {"group": "score", "fields": [
        {"flag": "-m", "type": "text", "required": True, "label": "model"},
        {"flag": "--tag", "type": "text"},
        {"flag": "--tracks", "type": "text", "default": "sft"},
        {"flag": "--closed-book", "type": "flag"}]},
    "matrix": {"group": "score", "fields": [
        {"flag": "--models", "type": "text", "placeholder": "qwen3:8b,qwen3:14b"},
        {"flag": "--tracks", "type": "text", "default": "sft"},
        {"flag": "--book", "type": "select", "options": ["", "closed", "open", "both"]}]},
    "compare": {"group": "score", "fields": [
        {"flag": "--base", "type": "run", "required": True},
        {"flag": "--after", "type": "run", "required": True},
        {"flag": "--markdown", "type": "text", "placeholder": "report.md"}]},

    "triage": {"group": "review", "fields": []},
    "review": {"group": "review", "fields": []},
    "export": {"group": "review", "fields": []},
}


class Job:
    """One `cb.py` subprocess, with its output kept for the log panel."""

    def __init__(self, command: str, args: List[str], cwd: Path):
        self.id = uuid.uuid4().hex[:12]
        self.command = command
        self.args = args
        self.started = time.time()
        self.finished: float | None = None
        self.code: int | None = None
        self.lines: deque = deque(maxlen=MAX_LOG_LINES)
        self.dropped = 0
        self._lock = threading.Lock()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            [sys.executable, str(cwd / "cb.py"), command, *args],
            cwd=str(cwd), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._lock:
                if len(self.lines) == self.lines.maxlen:
                    self.dropped += 1
                self.lines.append(line.rstrip("\n"))
        self.code = self.proc.wait()
        self.finished = time.time()

    @property
    def running(self) -> bool:
        return self.finished is None

    def tail(self, offset: int) -> Dict[str, Any]:
        with self._lock:
            total = self.dropped + len(self.lines)
            start = max(offset, self.dropped)
            body = list(self.lines)[start - self.dropped:]
        return {"lines": body, "next": total, "dropped": self.dropped}

    def stop(self) -> None:
        if self.running:
            self.proc.terminate()

    def describe(self) -> Dict[str, Any]:
        return {
            "id": self.id, "command": self.command, "args": self.args,
            "running": self.running, "code": self.code,
            "started": self.started, "finished": self.finished,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
        }


class Jobs:
    def __init__(self) -> None:
        self.all: Dict[str, Job] = {}
        self.order: List[str] = []

    def current(self) -> Job | None:
        for jid in reversed(self.order):
            if self.all[jid].running:
                return self.all[jid]
        return None

    def add(self, job: Job) -> Job:
        self.all[job.id] = job
        self.order.append(job.id)
        return job

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        return [self.all[j].describe() for j in reversed(self.order[-n:])]


def safe_path(cfg: Dict[str, Any], root: str, rel: str) -> Path:
    """Resolve <root>/<rel>, refusing anything that escapes the root."""
    roots = roots_for(cfg)
    if root not in roots:
        raise ValueError(f"unknown root: {root}")
    base = roots[root].resolve()
    target = (base / (rel or "")).resolve()
    if target != base and base not in target.parents:
        raise ValueError("path escapes its root")
    return target


def listing(path: Path, limit: int = 2000) -> Dict[str, Any]:
    dirs, files, truncated = [], [], False
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except (OSError, PermissionError) as exc:
        return {"error": str(exc), "dirs": [], "files": []}
    for entry in entries:
        if entry.name.startswith(".") and entry.name != ".ckpt":
            continue
        if len(dirs) + len(files) >= limit:
            truncated = True
            break
        if entry.is_dir():
            dirs.append({"name": entry.name})
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            files.append({"name": entry.name, "size": size,
                          "suffix": entry.suffix.lower()})
    return {"dirs": dirs, "files": files, "truncated": truncated}


def is_run_file(payload: Any) -> bool:
    return (isinstance(payload, dict) and payload.get("benchmark") == "kcbench"
            and "tracks" in payload)


def read_file(path: Path, offset: int, limit: int) -> Dict[str, Any]:
    """Whatever the viewer needs to render this file, and nothing more."""
    suffix = path.suffix.lower()
    size = path.stat().st_size

    if suffix in IMAGE_SUFFIXES:
        return {"kind": "image", "size": size}

    if suffix == ".jsonl":
        records, total = [], 0
        with path.open(encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh):
                if not line.strip():
                    continue
                total += 1
                if offset <= total - 1 < offset + limit:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        records.append({"_unparsed": line[:400], "_error": str(exc)})
        return {"kind": "records", "size": size, "total": total,
                "offset": offset, "records": records}

    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"kind": "text", "size": size, "text": f"unreadable JSON: {exc}"}
        kind = "run" if is_run_file(payload) else "json"
        # holdout.json is 1 MB of chunk digests; the viewer only ever shows its
        # shape, so send a summary rather than the whole thing.
        if kind == "json" and size > 512_000:
            return {"kind": "json", "size": size, "truncated": True,
                    "json": summarise_json(payload)}
        return {"kind": kind, "size": size, "json": payload}

    if suffix in TEXT_SUFFIXES or suffix == "":
        text = path.read_text(encoding="utf-8", errors="replace")
        clipped = len(text) > 400_000
        return {"kind": "text", "size": size, "clipped": clipped,
                "text": text[:400_000]}

    return {"kind": "binary", "size": size}


def summarise_json(payload: Any, depth: int = 0) -> Any:
    """Shape, not contents: enough to see what is in a file too big to send."""
    if depth > 2:
        return "..."
    if isinstance(payload, dict):
        return {k: summarise_json(v, depth + 1) for k, v in list(payload.items())[:40]}
    if isinstance(payload, list):
        head = [summarise_json(v, depth + 1) for v in payload[:3]]
        return head + [f"... {len(payload)} item(s) total"] if len(payload) > 3 else head
    if isinstance(payload, str) and len(payload) > 200:
        return payload[:200] + "..."
    return payload


_TRACK_CACHE: Dict[str, Any] = {}


def track_summaries(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    The benchmark as the user thinks of it: one row per item set, with what a
    run against it will be scored on. Counted from the files themselves so the
    page cannot drift from what is on disk; cached on mtime because the sets
    are a megabyte each and this is asked for on every page load.
    """
    out_dir = Path(cfg["out_dir"])
    rows: List[Dict[str, Any]] = []
    if not out_dir.is_dir():
        return rows
    for path in sorted(out_dir.glob("*.jsonl")):
        key = str(path)
        mtime = path.stat().st_mtime
        cached = _TRACK_CACHE.get(key)
        if cached and cached["mtime"] == mtime:
            rows.append(cached["row"])
            continue
        n, kinds, splits, modes, track, usecase = 0, {}, {}, {}, None, None
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # An item asks something. provenance.jsonl carries the same
                # ids and eval types but no question, and is not a track.
                if not any(k.startswith(("question", "instruction")) for k in rec):
                    continue
                n += 1
                kinds[rec.get("eval_type", "?")] = kinds.get(rec.get("eval_type", "?"), 0) + 1
                sp = rec.get("split", "holdout")
                splits[sp] = splits.get(sp, 0) + 1
                mo = rec.get("match_mode", "exact")
                modes[mo] = modes.get(mo, 0) + 1
                track = track or rec.get("track")
                usecase = usecase or rec.get("usecase")
        if n == 0:
            continue
        row = {"file": path.name, "size": path.stat().st_size, "n": n,
               "track": track, "usecase": usecase, "eval_types": kinds,
               "splits": splits, "match_modes": modes,
               "contaminated": splits.get("train", 0)}
        _TRACK_CACHE[key] = {"mtime": mtime, "row": row}
        rows.append(row)
    return rows


def run_summaries(cfg: Dict[str, Any], folder: Path | None = None) -> List[Dict[str, Any]]:
    """
    Every run file in a folder, flattened to what the dashboard plots. The
    default folder is where cb.py writes runs; any other directory under an
    allowed root can be asked for, so selecting a folder of results draws them.
    """
    runs_dir = folder if folder is not None else Path(cfg["out_dir"]) / "runs"
    out: List[Dict[str, Any]] = []
    if not runs_dir.is_dir():
        return out
    for path in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                      # a half-written run is not fatal
            continue
        if path.name.startswith("compare_"):
            out.append({"file": path.name, "kind": "compare",
                        "base": payload.get("base"), "after": payload.get("after"),
                        "headline": payload.get("headline", {}),
                        "significance": payload.get("significance", {})})
            continue
        if not is_run_file(payload):
            continue
        metrics: List[Dict[str, Any]] = []
        for track, body in (payload.get("tracks") or {}).items():
            if body.get("perplexity") is not None:
                metrics.append({"track": track, "type": "chunks",
                                "metric": "perplexity",
                                "value": body["perplexity"], "n": body.get("items"),
                                "lower_better": True})
            for kind, m in (body.get("by_type") or {}).items():
                for name in ("correct", "f1", "key_f1", "value_accuracy",
                             "abstained", "ece", "brier", "mean_inconsistency"):
                    if isinstance(m.get(name), (int, float)):
                        metrics.append({
                            "track": track, "type": kind, "metric": name,
                            "value": m[name], "n": m.get("n"),
                            "ci95": m.get(f"{name}_ci95"),
                            "no_answer": m.get("no_answer"),
                            "lower_better": name in ("ece", "brier"),
                        })
        out.append({
            "file": path.name, "kind": "run", "tag": payload.get("tag"),
            "model": payload.get("model"), "book": payload.get("book"),
            "lang": payload.get("lang"), "think": payload.get("think"),
            "elapsed": payload.get("elapsed_sec"),
            "headline": payload.get("headline", {}),
            "meta": payload.get("meta", {}), "metrics": metrics,
        })
    return out


def create_app(cfg: Dict[str, Any], jobs: Jobs):
    try:
        from flask import Flask, jsonify, request, send_file, send_from_directory
    except ImportError as exc:                 # noqa: PLC0415
        raise SystemExit(FLASK_HINT) from exc

    static_dir = Path(__file__).resolve().parent / "webui"
    app = Flask(__name__, static_folder=None)

    @app.after_request
    def no_store(resp):
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/ui/<path:name>")
    def ui(name: str):
        return send_from_directory(static_dir, name)

    @app.get("/api/state")
    def state():
        roots = roots_for(cfg)
        current = jobs.current()
        return jsonify({
            "commands": COMMAND_SPECS,
            "roots": {k: {"path": str(v), "exists": v.exists()}
                      for k, v in roots.items()},
            "input_roots": list(INPUT_ROOTS),
            "output_roots": list(OUTPUT_ROOTS),
            "config_path": cfg.get("_config_path"),
            "project": str(PROJECT),
            "job": current.describe() if current else None,
            "recent": jobs.recent(),
        })

    @app.get("/api/config")
    def get_config():
        path = cfg.get("_config_path")
        if not path or not Path(path).is_file():
            return jsonify({"path": path, "text": "", "missing": True})
        return jsonify({"path": path,
                        "text": Path(path).read_text(encoding="utf-8")})

    @app.put("/api/config")
    def put_config():
        path = cfg.get("_config_path")
        if not path:
            return jsonify({"error": "this session has no config file to write"}), 400
        text = (request.json or {}).get("text", "")
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"not valid JSON: {exc}"}), 400
        target = Path(path)
        if target.is_file():
            shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
        target.write_text(text, encoding="utf-8")
        LOG.info("wrote %s (previous kept as .bak)", target)
        return jsonify({"saved": True, "path": str(target),
                        "note": "takes effect on the next command; "
                                "restart the webview to re-resolve its own paths"})

    @app.get("/api/ls")
    def ls():
        try:
            path = safe_path(cfg, request.args.get("root", ""),
                             request.args.get("path", ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not path.is_dir():
            return jsonify({"error": "not a directory", "dirs": [], "files": []})
        return jsonify(listing(path))

    @app.get("/api/file")
    def file_():
        try:
            path = safe_path(cfg, request.args.get("root", ""),
                             request.args.get("path", ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not path.is_file():
            return jsonify({"error": "not a file"}), 404
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(200, max(1, int(request.args.get("limit", 25))))
        try:
            payload = read_file(path, offset, limit)
        except Exception as exc:               # a broken file is not fatal
            return jsonify({"error": str(exc)}), 500
        payload["name"] = path.name
        return jsonify(payload)

    @app.get("/api/raw")
    def raw():
        try:
            path = safe_path(cfg, request.args.get("root", ""),
                             request.args.get("path", ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not path.is_file():
            return jsonify({"error": "not a file"}), 404
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return send_file(path, mimetype=mime)

    @app.get("/api/tracks")
    def tracks():
        return jsonify({"tracks": track_summaries(cfg)})

    @app.get("/api/runs")
    def runs():
        root, rel = request.args.get("root"), request.args.get("path", "")
        if not root:
            return jsonify({"runs": run_summaries(cfg), "folder": "runs"})
        try:
            folder = safe_path(cfg, root, rel)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"runs": run_summaries(cfg, folder),
                        "folder": f"{root}/{rel}" if rel else root})

    @app.post("/api/run")
    def run_command():
        body = request.json or {}
        command = body.get("command")
        if command not in COMMAND_SPECS:
            return jsonify({"error": f"unknown command: {command}"}), 400
        running = jobs.current()
        if running:
            return jsonify({
                "error": "one command at a time",
                "detail": f"'{running.command}' is still running. Scoring loads a "
                          "model and `ppl` loads a second copy locally; on a "
                          "unified-memory box running two at once is what kills "
                          "long runs. Stop it first.",
                "job": running.describe()}), 409
        args = [str(a) for a in body.get("args", []) if str(a).strip()]
        job = jobs.add(Job(command, args, HERE))
        LOG.info("started %s %s (job %s)", command, " ".join(args), job.id)
        return jsonify({"job": job.describe()})

    @app.get("/api/log")
    def log_():
        job = jobs.all.get(request.args.get("job", ""))
        if not job:
            return jsonify({"error": "no such job"}), 404
        payload = job.tail(max(0, int(request.args.get("from", 0))))
        payload["job"] = job.describe()
        return jsonify(payload)

    @app.post("/api/stop")
    def stop():
        job = jobs.all.get((request.json or {}).get("job", ""))
        if not job:
            return jsonify({"error": "no such job"}), 404
        job.stop()
        LOG.info("stop requested for job %s", job.id)
        return jsonify({"job": job.describe()})

    return app


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Serve the benchmark's browser view",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    add_common_args(ap)
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface to bind (default 127.0.0.1, local only)")
    ap.add_argument("--port", type=int, default=8799, help="port (default 8799)")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not open a browser window")
    ap.add_argument("--debug", action="store_true", help="Flask debug reloader")
    args = ap.parse_args(argv)

    cfg = resolve_config(args)
    describe(cfg)

    app = create_app(cfg, Jobs())
    url = f"http://{args.host}:{args.port}/"
    LOG.info("webview on %s", url)
    if args.host not in ("127.0.0.1", "localhost"):
        LOG.warning("bound to %s - this serves your corpus to the network, and "
                    "anyone who can reach it can start a command", args.host)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
