"""
Collect public IFC (BIM) models: buildings, infrastructure and facilities.

Sources are public GitHub repositories that publish IFC test and sample files
under open licences - buildingSMART's own conformance set, the IfcOpenShell
test corpus, and community sample collections.

Each model is classified by what it represents rather than by where it came
from, because a training set is organised by subject:

    01_buildings            houses, offices, clinics, apartments
    02_infrastructure       bridges, roads, rail, tunnels, ports
    03_facilities_and_mep   plants, MEP-heavy and industrial models
    04_structural           structural-only models
    05_site_and_civil       terrain, landscaping, earthworks
    06_reference_and_test   small conformance and unit-test files

The IFC header is parsed for schema, authoring tool and description, so the
catalogue records what each file actually is rather than trusting its name.

The repository list, category patterns and budget all come from config.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, project_root, resolve_dir  # noqa: E402
from collect import _safe_name, _write_json, _relative_to_root  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("collect_ifc")


@dataclass
class IfcRecord:
    file_name: str
    relative_path: str
    category: str
    size_bytes: int
    sha256: str
    source_url: str
    downloaded_utc: str
    repository: str = ""
    repo_path: str = ""
    licence_note: str = ""
    # Parsed from the IFC header
    schema: str = ""
    authoring_tool: str = ""
    description: str = ""
    file_name_header: str = ""
    timestamp: str = ""
    organisation: str = ""
    entity_count: Optional[int] = None


class IfcCollector:
    def __init__(self, root: Path, cfg: Dict[str, Any]) -> None:
        self.root = root
        self.cfg = cfg
        ifc = cfg["ifc"]
        http = cfg["http"]
        paths = cfg["paths"]

        self.github_api = ifc["github_api"].rstrip("/")
        self.repositories = ifc["repositories"]
        self.data_dir = resolve_dir(root, paths["data_dir"])
        if ifc.get("output_subdir"):
            self.data_dir = self.data_dir / ifc["output_subdir"]
        self.meta_dir = resolve_dir(root, paths["metadata_dir"])
        self.outputs = ifc["outputs"]

        self.limit_bytes = int(ifc["limit_gb"] * 1e9)
        self.delay = ifc["delay_seconds"]
        self.api_timeout = http["api_timeout_seconds"]
        self.download_timeout = http["download_timeout_seconds"]
        self.small_file_bytes = ifc["small_file_bytes"]
        self.categories = [
            (entry["name"], re.compile(entry["pattern"], re.IGNORECASE))
            for entry in ifc["categories"]
        ]
        self.fallback_category = ifc["fallback_category"]

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": http["user_agent"], "Accept": "application/vnd.github+json"}
        )
        token = ifc.get("github_token") or os.environ.get("GITHUB_TOKEN", "")
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
            logger.info("Using a GitHub token for the API rate limit.")

        self.records: List[IfcRecord] = []
        self.seen_hashes: Set[str] = set()
        self.total_bytes = 0
        self.skipped_duplicates = 0
        self.failures: List[Dict[str, str]] = []

    def _room_left(self) -> int:
        return self.limit_bytes - self.total_bytes

    # ── Discovery ─────────────────────────────────────────────────────────

    def list_ifc_files(self, repo: str) -> List[Dict[str, Any]]:
        """Every .ifc/.ifczip blob in a repository's default branch."""
        for branch in ("main", "master"):
            try:
                r = self.session.get(
                    f"{self.github_api}/repos/{repo}/git/trees/{branch}",
                    params={"recursive": "1"}, timeout=max(self.api_timeout, 60),
                )
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                tree = r.json().get("tree", [])
            except Exception as exc:
                self.failures.append({"repo": repo, "error": str(exc)})
                return []

            files = [
                {"path": t["path"], "size": t.get("size", 0), "branch": branch}
                for t in tree
                if t.get("type") == "blob"
                and t["path"].lower().endswith((".ifc", ".ifczip"))
            ]
            if files:
                return files
        return []

    # ── Download ──────────────────────────────────────────────────────────

    def fetch(self, repo: str, note: str, entry: Dict[str, Any]) -> None:
        if entry["size"] and entry["size"] > self._room_left():
            return

        url = (
            f"https://raw.githubusercontent.com/{repo}/"
            f"{entry['branch']}/{quote(entry['path'])}"
        )
        try:
            r = self.session.get(url, timeout=max(self.download_timeout, 180))
            r.raise_for_status()
            payload = r.content
        except Exception as exc:
            self.failures.append({"repo": repo, "path": entry["path"], "error": str(exc)})
            return

        if not payload or len(payload) > self._room_left():
            return

        digest = hashlib.sha256(payload).hexdigest()
        if digest in self.seen_hashes:
            self.skipped_duplicates += 1
            return
        self.seen_hashes.add(digest)

        header = parse_ifc_header(payload)
        category = self.classify_model(entry["path"], header, len(payload))

        target_dir = self.data_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        name = _safe_ifc_name(Path(entry["path"]).name)
        path = target_dir / name
        counter = 1
        while path.exists():
            path = target_dir / f"{Path(name).stem}__{counter}{Path(name).suffix}"
            counter += 1
        path.write_bytes(payload)

        self.total_bytes += len(payload)
        self.records.append(
            IfcRecord(
                file_name=path.name,
                relative_path=_relative_to_root(path, self.root),
                category=category,
                size_bytes=len(payload),
                sha256=digest,
                source_url=url,
                downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                repository=repo,
                repo_path=entry["path"],
                licence_note=note,
                **header,
            )
        )

    def classify_model(self, path: str, header: Dict[str, Any], size: int) -> str:
        haystack = " ".join([
            path, header.get("description", ""), header.get("file_name_header", ""),
        ]).lower()
        for name, pattern in self.categories:
            if pattern.search(haystack):
                return name
        return self.fallback_category

    def run(self) -> None:
        for source in self.repositories:
            if self._room_left() <= 0:
                logger.info("Size cap reached; stopping.")
                return
            repo, note = source["repo"], source.get("note", "")
            files = self.list_ifc_files(repo)
            if not files:
                logger.info("[%s] no IFC files found", repo)
                continue
            total_mb = sum(f["size"] for f in files) / 1e6
            logger.info(
                "[%s] %d IFC file(s), %.1f MB", repo, len(files), total_mb
            )
            for entry in files:
                if self._room_left() <= 0:
                    return
                self.fetch(repo, note, entry)
                time.sleep(self.delay)
            logger.info(
                "  collected so far: %d file(s), %.2f GB",
                len(self.records), self.total_bytes / 1e9,
            )

    # ── Output ────────────────────────────────────────────────────────────

    def write_metadata(self, started: datetime) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        by_category: Dict[str, Dict[str, Any]] = {}
        by_schema: Dict[str, int] = {}
        for rec in self.records:
            entry = by_category.setdefault(
                rec.category, {"file_count": 0, "size_bytes": 0}
            )
            entry["file_count"] += 1
            entry["size_bytes"] += rec.size_bytes
            by_schema[rec.schema or "unknown"] = by_schema.get(rec.schema or "unknown", 0) + 1

        catalog = {
            "dataset": "Public IFC (BIM) models for AI training",
            "schema_version": "1.0",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collection_started_utc": started.isoformat(timespec="seconds"),
            "sources": self.repositories,
            "totals": {
                "file_count": len(self.records),
                "size_bytes": self.total_bytes,
                "size_gb": round(self.total_bytes / 1e9, 3),
                "duplicates_skipped": self.skipped_duplicates,
                "failures": len(self.failures),
            },
            "categories": dict(sorted(by_category.items())),
            "ifc_schemas": dict(sorted(by_schema.items(), key=lambda kv: -kv[1])),
            "models": [asdict(r) for r in self.records],
        }
        _write_json(self.meta_dir / self.outputs["catalog"], catalog)
        with open(self.meta_dir / self.outputs["index"], "w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        if self.failures:
            _write_json(self.meta_dir / self.outputs["failures"], self.failures)
        logger.info("Metadata written to %s", self.meta_dir)


# ── IFC header parsing ────────────────────────────────────────────────────

_HEADER_RE = re.compile(rb"HEADER;(.{0,4000}?)ENDSEC;", re.S | re.I)


def parse_ifc_header(payload: bytes) -> Dict[str, Any]:
    """
    Pull schema, tool and description out of the STEP header.

    Only the first few kilobytes are read: the header is always at the top and
    these files run to hundreds of megabytes.
    """
    out = {
        "schema": "", "authoring_tool": "", "description": "",
        "file_name_header": "", "timestamp": "", "organisation": "",
        "entity_count": None,
    }
    head = payload[:8192]
    match = _HEADER_RE.search(head)
    if not match:
        return out
    text = match.group(1).decode("utf-8", errors="replace")

    schema = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", text, re.I)
    if schema:
        out["schema"] = schema.group(1)

    desc = re.search(r"FILE_DESCRIPTION\s*\(\s*\(\s*'([^']*)'", text, re.I)
    if desc:
        out["description"] = desc.group(1)[:300]

    name = re.search(r"FILE_NAME\s*\((.{0,1500}?)\)\s*;", text, re.S | re.I)
    if name:
        fields = re.findall(r"'([^']*)'", name.group(1))
        if fields:
            out["file_name_header"] = fields[0][:200]
        if len(fields) > 1:
            out["timestamp"] = fields[1][:40]
        # FILE_NAME order: name, time, author, organisation, preprocessor,
        # originating system, authorisation.
        if len(fields) > 3:
            out["organisation"] = fields[3][:120]
        if len(fields) > 5:
            out["authoring_tool"] = fields[5][:150]
        elif len(fields) > 4:
            out["authoring_tool"] = fields[4][:150]

    # Cheap proxy for model complexity; exact counts would need a full parse.
    out["entity_count"] = payload.count(b"=IFC") or None
    return out


def _safe_ifc_name(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|\r\n\t]", "_", name).strip()
    name = re.sub(r"\s+", " ", name)
    suffix = ".ifczip" if name.lower().endswith(".ifczip") else ".ifc"
    stem = name[: -len(suffix)] if name.lower().endswith(suffix) else name
    while len(stem.encode("utf-8")) > 180:
        stem = stem[:-1]
    return stem + suffix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect public IFC/BIM models for AI training. "
                    "Defaults come from config.json; the flags below override it."
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.json (default: <root>/config.json).")
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--limit-gb", type=float, default=None)
    parser.add_argument("--delay", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.root)
    ifc = cfg["ifc"]
    if args.limit_gb is not None:
        ifc["limit_gb"] = args.limit_gb
    if args.delay is not None:
        ifc["delay_seconds"] = args.delay

    started = datetime.now(timezone.utc)
    collector = IfcCollector(root=args.root, cfg=cfg)
    logger.info(
        "Collecting IFC models into %s (budget %.1f GB, config %s)",
        collector.data_dir, ifc["limit_gb"],
        cfg.get("_config_path", "built-in defaults"),
    )
    try:
        collector.run()
    except KeyboardInterrupt:
        logger.warning("Interrupted - writing metadata for what was collected.")
    finally:
        collector.write_metadata(started)

    logger.info(
        "Done: %d model(s), %.2f GB, %d duplicate(s), %d failure(s)",
        len(collector.records), collector.total_bytes / 1e9,
        collector.skipped_duplicates, len(collector.failures),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
