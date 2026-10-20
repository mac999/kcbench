"""
Collect construction research reports and technical documents from CODIL.

CODIL (건설기술정보시스템, codil.or.kr) is the Ministry of Land, Infrastructure
and Transport's construction technology information system, operated by KICT.
Its robots.txt allows crawling, and the documents are publicly funded research
output.

Unlike the law.go.kr corpus, which is thousands of small notices, these are
full research reports - typically tens of megabytes and hundreds of pages
each, which is what makes a large training corpus reachable.

Note on TLS: the server sends its leaf certificate without the intermediate,
so a default client cannot build a chain to the root. The intermediate is
fetched from the certificate's own AIA extension and appended to the system
bundle; verification stays on.

Budgets, sections, output folders and TLS behaviour all come from config.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, project_root, resolve_dir  # noqa: E402
from collect import (  # noqa: E402  - shared taxonomy and helpers
    Taxonomy, _safe_name, _decode_zip_name, _page_count, _write_json,
    _relative_to_root,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("collect_codil")

# The visible row is an anchor to the detail page; the fn_dtlSearch() calls
# that also carry the code sit inside HTML comments, so they yield no title.
_ROW_RE = re.compile(
    r"<a\s+href='[^']*viewDtlConRpt\.do[^']*pMetaCode=([A-Z0-9]+)[^']*'>([^<]{2,300})</a>",
    re.I,
)
_META_RE = re.compile(r"pMetaCode=([A-Z0-9]+)")


@dataclass
class ReportRecord:
    file_name: str
    relative_path: str
    category: str
    size_bytes: int
    sha256: str
    content_type: str
    source_url: str
    downloaded_utc: str
    doc_type: str = "codil_report"
    title: str = ""
    meta_code: str = ""
    section: str = ""
    page_count: Optional[int] = None
    extracted_from: str = ""


class CodilCollector:
    def __init__(self, root: Path, cfg: Dict[str, Any]) -> None:
        self.root = root
        self.cfg = cfg
        codil = cfg["codil"]
        http = cfg["http"]
        paths = cfg["paths"]

        self.base = codil["base_url"].rstrip("/")
        self.sections = codil["sections"]
        self.group_by_section = bool(codil.get("group_by_section", False))
        self.data_dir = resolve_dir(root, paths["data_dir"])
        if codil.get("output_subdir"):
            self.data_dir = self.data_dir / codil["output_subdir"]
        self.meta_dir = resolve_dir(root, paths["metadata_dir"])
        self.certs_dir = resolve_dir(root, paths["certs_dir"])
        self.outputs = codil["outputs"]

        self.limit_bytes = int(codil["limit_gb"] * 1e9)
        self.delay = codil["delay_seconds"]
        self.list_timeout = http["api_timeout_seconds"]
        self.download_timeout = http["download_timeout_seconds"]

        self.taxonomy = Taxonomy(cfg["taxonomy"])
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": http["user_agent"]})
        self.session.verify = str(_resolve_ca_bundle(self.certs_dir, codil["tls"]))

        self.records: List[ReportRecord] = []
        self.seen_hashes: Set[str] = set()
        self.seen_codes: Set[str] = set()
        self.total_bytes = 0
        self.skipped_irrelevant = 0
        self.skipped_duplicates = 0
        self.skipped_oversize = 0
        self.failures: List[Dict[str, str]] = []

    def _room_left(self) -> int:
        return self.limit_bytes - self.total_bytes

    # ── Listing ───────────────────────────────────────────────────────────

    def list_page(self, gubun: str, page: int) -> List[Dict[str, str]]:
        """One page of the section index: metadata code plus title."""
        try:
            r = self.session.get(
                f"{self.base}/viewConRpt.do",
                params={"gubun": gubun, "pageIndex": page},
                timeout=max(self.list_timeout, 60),
            )
            r.raise_for_status()
        except Exception as exc:
            self.failures.append(
                {"where": f"list {gubun} p{page}", "error": str(exc)}
            )
            return []

        html = r.text
        rows: List[Dict[str, str]] = []
        seen_here: Set[str] = set()
        for match in _ROW_RE.finditer(html):
            code, title = match.group(1), _clean(match.group(2))
            # Listings prefix an index number onto the title.
            title = re.sub(r"^\s*\d+\s*[.)]\s*", "", title)
            if code in seen_here or not title:
                continue
            seen_here.add(code)
            rows.append({"meta_code": code, "title": title})

        # Fall back to codes alone if the title capture missed the markup.
        if not rows:
            for code in dict.fromkeys(_META_RE.findall(html)):
                rows.append({"meta_code": code, "title": ""})
        return rows

    # ── Download ──────────────────────────────────────────────────────────

    def fetch(self, meta_code: str, title: str, section: str) -> None:
        if meta_code in self.seen_codes:
            return
        self.seen_codes.add(meta_code)

        if title and not self.taxonomy.is_relevant(title):
            self.skipped_irrelevant += 1
            return

        url = f"{self.base}/fileDown.do?pMetaCode={meta_code}"
        try:
            head = self.session.get(url, timeout=self.download_timeout, stream=True)
            head.raise_for_status()
            declared = int(head.headers.get("Content-Length") or 0)
            if declared and declared > self._room_left():
                self.skipped_oversize += 1
                head.close()
                return
            payload = head.content
        except Exception as exc:
            self.failures.append({"meta_code": meta_code, "error": str(exc)})
            return

        if not payload or len(payload) > self._room_left():
            self.skipped_oversize += 1
            return

        category = self.taxonomy.classify(title) if title else self.taxonomy.fallback
        if payload[:2] == b"PK":
            self._store_zip(payload, meta_code, title, category, section, url)
        elif payload.startswith(b"%PDF"):
            self._store_pdf(
                payload, f"{meta_code}.pdf", meta_code, title, category, section, url
            )

    def _store_zip(
        self, payload: bytes, meta_code: str, title: str,
        category: str, section: str, url: str,
    ) -> None:
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".pdf"):
                        continue
                    if info.file_size > self._room_left():
                        self.skipped_oversize += 1
                        break
                    inner = zf.read(info)
                    if not inner.startswith(b"%PDF"):
                        continue
                    name = _decode_zip_name(info.filename).lstrip("/")
                    # The archive names files by code; the title is far more
                    # useful on disk.
                    pretty = f"{title[:120]} [{meta_code}].pdf" if title else name
                    self._store_pdf(
                        inner, pretty, meta_code, title, category, section, url,
                        from_zip=f"{meta_code}.zip",
                    )
        except zipfile.BadZipFile:
            self.failures.append({"meta_code": meta_code, "error": "bad zip"})

    def _store_pdf(
        self, payload: bytes, name: str, meta_code: str, title: str,
        category: str, section: str, url: str, from_zip: str = "",
    ) -> None:
        digest = hashlib.sha256(payload).hexdigest()
        if digest in self.seen_hashes:
            self.skipped_duplicates += 1
            return

        target_dir = self.data_dir / section / category if self.group_by_section \
            else self.data_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / _safe_name(name)
        counter = 1
        while path.exists():
            path = target_dir / f"{_safe_name(name)[:-4]}__{counter}.pdf"
            counter += 1
        path.write_bytes(payload)

        self.seen_hashes.add(digest)
        self.total_bytes += len(payload)
        self.records.append(
            ReportRecord(
                file_name=path.name,
                relative_path=_relative_to_root(path, self.root),
                category=category,
                size_bytes=len(payload),
                sha256=digest,
                content_type="application/pdf",
                source_url=url,
                downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                title=title,
                meta_code=meta_code,
                section=section,
                page_count=_page_count(payload),
                extracted_from=from_zip,
            )
        )

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self, sections: List[str], max_pages: int) -> None:
        for gubun in sections:
            label = self.sections.get(gubun, gubun)
            for page in range(1, max_pages + 1):
                if self._room_left() <= 0:
                    logger.info("Size cap reached; stopping.")
                    return
                rows = self.list_page(gubun, page)
                if not rows:
                    logger.info("[%s] page %d empty; moving on.", label, page)
                    break
                logger.info(
                    "[%s] page %d: %d item(s)  (%.2f GB, %d file(s))",
                    label, page, len(rows), self.total_bytes / 1e9, len(self.records),
                )
                for row in rows:
                    if self._room_left() <= 0:
                        return
                    self.fetch(row["meta_code"], row["title"], label)
                    time.sleep(self.delay)

    def write_metadata(self, started: datetime) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        by_section: Dict[str, Dict[str, Any]] = {}
        for rec in self.records:
            key = f"{rec.section}/{rec.category}"
            entry = by_section.setdefault(key, {"file_count": 0, "size_bytes": 0})
            entry["file_count"] += 1
            entry["size_bytes"] += rec.size_bytes

        catalog = {
            "dataset": "Korean construction research reports for AI training",
            "schema_version": "1.0",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collection_started_utc": started.isoformat(timespec="seconds"),
            "source": {
                "name": "CODIL 건설기술정보시스템 (Construction Technology Information System)",
                "url": self.base,
                "operator": "한국건설기술연구원 (KICT) for 국토교통부 (MOLIT)",
                "note": (
                    "robots.txt allows crawling. Publicly funded construction "
                    "research reports and technical documents."
                ),
            },
            "totals": {
                "file_count": len(self.records),
                "size_bytes": self.total_bytes,
                "size_gb": round(self.total_bytes / 1e9, 3),
                "duplicates_skipped": self.skipped_duplicates,
                "off_topic_skipped": self.skipped_irrelevant,
                "oversize_skipped": self.skipped_oversize,
                "failures": len(self.failures),
            },
            "sections": dict(sorted(by_section.items())),
            "documents": [asdict(r) for r in self.records],
        }
        _write_json(self.meta_dir / self.outputs["catalog"], catalog)
        with open(self.meta_dir / self.outputs["index"], "w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        if self.failures:
            _write_json(self.meta_dir / self.outputs["failures"], self.failures)
        logger.info("Metadata written to %s", self.meta_dir)


# ── Helpers ───────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _resolve_ca_bundle(certs_dir: Path, tls: Dict[str, Any]) -> Path:
    """
    Build (or reuse) a CA bundle that can verify CODIL.

    The server omits its intermediate certificate. Rather than turn
    verification off, the intermediate named in the leaf's AIA extension is
    downloaded once and appended to the system roots. Point
    ``codil.tls.ca_bundle_path`` at your own bundle to skip that entirely.
    """
    import ssl
    import subprocess

    explicit = tls.get("ca_bundle_path")
    if explicit:
        return Path(explicit)

    bundle = certs_dir / "bundle.pem"
    if bundle.exists() and bundle.stat().st_size > 0:
        return bundle
    if not tls.get("build_ca_bundle", True):
        return Path(tls["system_ca_bundle"])

    certs_dir.mkdir(parents=True, exist_ok=True)
    system = Path(tls["system_ca_bundle"])
    leaf_pem = ssl.get_server_certificate((tls["host"], int(tls["port"])))
    (certs_dir / "codil_leaf.pem").write_text(leaf_pem)

    text = subprocess.run(
        ["openssl", "x509", "-noout", "-text"],
        input=leaf_pem.encode(), capture_output=True,
    ).stdout.decode(errors="replace")
    match = re.search(r"CA Issuers - URI:(http[^\s]+)", text)
    if not match:
        raise RuntimeError("Could not find the intermediate certificate URI")

    der = requests.get(match.group(1), timeout=60).content
    pem = subprocess.run(
        ["openssl", "x509", "-inform", "DER", "-outform", "PEM"],
        input=der, capture_output=True,
    ).stdout
    bundle.write_bytes(system.read_bytes() + b"\n" + pem)
    logger.info("Built CA bundle with the missing intermediate: %s", bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect construction research reports from CODIL. "
                    "Defaults come from config.json; the flags below override it."
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.json (default: <root>/config.json).")
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--limit-gb", type=float, default=None)
    parser.add_argument("--sections", nargs="+", default=None,
                        help="CODIL section codes, e.g. rpt tch bus.")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--delay", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.root)
    codil = cfg["codil"]
    if args.limit_gb is not None:
        codil["limit_gb"] = args.limit_gb
    if args.sections is not None:
        codil["enabled_sections"] = args.sections
    if args.max_pages is not None:
        codil["max_pages"] = args.max_pages
    if args.delay is not None:
        codil["delay_seconds"] = args.delay

    unknown = [s for s in codil["enabled_sections"] if s not in codil["sections"]]
    if unknown:
        parser.error(
            f"unknown section(s) {unknown}; known: {sorted(codil['sections'])}"
        )

    started = datetime.now(timezone.utc)
    collector = CodilCollector(root=args.root, cfg=cfg)
    logger.info(
        "Collecting CODIL into %s (budget %.1f GB, config %s)",
        collector.data_dir, codil["limit_gb"],
        cfg.get("_config_path", "built-in defaults"),
    )
    try:
        collector.run(codil["enabled_sections"], codil["max_pages"])
    except KeyboardInterrupt:
        logger.warning("Interrupted - writing metadata for what was collected.")
    finally:
        collector.write_metadata(started)

    logger.info(
        "Done: %d file(s), %.2f GB, %d duplicate(s), %d off-topic, "
        "%d oversize, %d failure(s)",
        len(collector.records), collector.total_bytes / 1e9,
        collector.skipped_duplicates, collector.skipped_irrelevant,
        collector.skipped_oversize, len(collector.failures),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
