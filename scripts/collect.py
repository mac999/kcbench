"""
Collect Korean construction regulation documents for AI training.

Source: 국가법령정보센터 (law.go.kr) DRF open API, whose robots.txt allows
crawling and whose documents are public government records.

Three document families are gathered:

    admrul  administrative rules - this is where the KDS design standards and
            KCS standard specifications actually live, published as ministry
            notices with the full standard attached
    law     acts, enforcement decrees and enforcement rules
    licbyl  the tables and forms attached to them

Only the attachments are downloaded; the API's own metadata is kept alongside
so every file can be traced back to the rule that published it.

Everything tunable - budgets, search terms, the category taxonomy, output
folders - comes from config.json; see scripts/config.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, project_root, resolve_dir  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("collect")

# Attachment kinds worth keeping.
PDF_RE = re.compile(r"\.pdf$", re.IGNORECASE)
ZIP_RE = re.compile(r"\.zip$", re.IGNORECASE)


# ── Classification ────────────────────────────────────────────────────────

class Taxonomy:
    """
    The relevance gate and the category folders, compiled from config.

    law.go.kr searches full text, so a query for 건설 also returns rules that
    merely mention it - military service notices, court deposit procedures,
    forestry plans. The title has to carry the subject itself for a document
    to enter the dataset.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        relevance = cfg.get("relevance", {})
        self.include = re.compile(relevance.get("include_pattern", "."), re.IGNORECASE)
        exclude = relevance.get("exclude_pattern") or ""
        self.exclude = re.compile(exclude, re.IGNORECASE) if exclude else None
        self.categories = [
            (entry["name"], re.compile(entry["pattern"], re.IGNORECASE))
            for entry in cfg.get("categories", [])
        ]
        self.fallback = cfg.get("fallback_category", "99_uncategorised")

    def is_relevant(self, title: str) -> bool:
        if not self.include.search(title):
            return False
        return not (self.exclude and self.exclude.search(title))

    def classify(self, title: str) -> str:
        for name, pattern in self.categories:
            if pattern.search(title):
                return name
        return self.fallback


@dataclass
class DocRecord:
    """One downloaded file plus the rule it came from."""
    file_name: str
    relative_path: str
    category: str
    size_bytes: int
    sha256: str
    content_type: str
    source_url: str
    downloaded_utc: str
    # Provenance from the API
    doc_type: str = ""              # admrul | law | licbyl
    title: str = ""
    rule_id: str = ""
    rule_seq: str = ""
    ministry: str = ""
    promulgation_no: str = ""
    promulgation_date: str = ""
    revision_type: str = ""
    related_law: str = ""
    attachment_name: str = ""
    extracted_from: str = ""        # set when the file came out of a zip
    page_count: Optional[int] = None


class Collector:
    def __init__(self, root: Path, cfg: Dict[str, Any]) -> None:
        self.root = root
        self.cfg = cfg
        law = cfg["law"]
        http = cfg["http"]
        paths = cfg["paths"]

        self.api = law["api_base"].rstrip("/")
        self.oc = law["oc"]
        self.data_dir = resolve_dir(root, paths["data_dir"])
        if law.get("output_subdir"):
            self.data_dir = self.data_dir / law["output_subdir"]
        self.meta_dir = resolve_dir(root, paths["metadata_dir"])
        self.outputs = law["outputs"]

        self.limit_bytes = int(law["limit_gb"] * 1e9)
        self.delay = law["delay_seconds"]
        self.api_timeout = http["api_timeout_seconds"]
        self.download_timeout = http["download_timeout_seconds"]
        self.max_retries = http["max_retries"]

        self.taxonomy = Taxonomy(cfg["taxonomy"])
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": http["user_agent"]})

        self.records: List[DocRecord] = []
        self.seen_hashes: Set[str] = set()
        self.seen_urls: Set[str] = set()
        self.total_bytes = 0
        self.skipped_duplicates = 0
        self.skipped_irrelevant = 0
        self.failures: List[Dict[str, str]] = []

    # ── API ───────────────────────────────────────────────────────────────

    def _get_json(self, path: str, params: Dict[str, Any]) -> Optional[Dict]:
        params = {"OC": self.oc, "type": "JSON", **params}
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(
                    f"{self.api}/{path}", params=params, timeout=self.api_timeout
                )
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    logger.warning("API %s failed: %s", path, exc)
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def search(self, target: str, query: str, max_items: int = 100) -> List[Dict]:
        """Paged search over one target."""
        out: List[Dict] = []
        page = 1
        while len(out) < max_items:
            data = self._get_json(
                "lawSearch.do",
                {"target": target, "query": query, "display": 100, "page": page},
            )
            if not data:
                break
            section = data.get(list(data)[0], {})
            items = section.get(target) or []
            if isinstance(items, dict):
                items = [items]
            if not items:
                break
            out.extend(items)
            total = int(section.get("totalCnt", 0) or 0)
            if page * 100 >= total:
                break
            page += 1
            time.sleep(self.delay)
        return out[:max_items]

    def attachments(self, target: str, rule_seq: str) -> List[tuple[str, str]]:
        """(name, url) for every attachment on one rule."""
        data = self._get_json(
            "lawService.do", {"target": target, "ID": rule_seq}
        )
        if not data:
            return []
        body = data.get(list(data)[0], {})
        node = body.get("첨부파일") or {}
        names = node.get("첨부파일명") or []
        links = node.get("첨부파일링크") or []
        if isinstance(names, str):
            names, links = [names], [links]
        return list(zip(names, links))

    # ── Download ──────────────────────────────────────────────────────────

    def _room_left(self) -> int:
        return self.limit_bytes - self.total_bytes

    def download(
        self, url: str, name: str, category: str, meta: Dict[str, str]
    ) -> Optional[DocRecord]:
        if url in self.seen_urls:
            self.skipped_duplicates += 1
            return None
        self.seen_urls.add(url)

        if self._room_left() <= 0:
            return None

        # law.go.kr serves attachments over http in the API payload; force https.
        url = url.replace("http://law.go.kr", "https://www.law.go.kr")

        try:
            r = self.session.get(url, timeout=self.download_timeout, stream=True)
            r.raise_for_status()
            payload = r.content
        except Exception as exc:
            self.failures.append({"url": url, "name": name, "error": str(exc)})
            return None

        if not payload:
            return None

        digest = hashlib.sha256(payload).hexdigest()
        if digest in self.seen_hashes:
            # The same standard is attached to several rules; keep one copy.
            self.skipped_duplicates += 1
            return None

        if len(payload) > self._room_left():
            logger.info("Size cap reached, skipping %s", name[:60])
            return None

        if ZIP_RE.search(name):
            return self._store_zip(payload, name, category, meta, url, digest)
        if not payload.startswith(b"%PDF"):
            return None
        return self._store_pdf(payload, name, category, meta, url, digest)

    def _store_pdf(
        self, payload: bytes, name: str, category: str,
        meta: Dict[str, str], url: str, digest: str, from_zip: str = "",
    ) -> DocRecord:
        target_dir = self.data_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        safe = _safe_name(name)
        path = target_dir / safe
        counter = 1
        while path.exists():
            path = target_dir / f"{Path(safe).stem}__{counter}.pdf"
            counter += 1
        path.write_bytes(payload)

        self.seen_hashes.add(digest)
        self.total_bytes += len(payload)

        record = DocRecord(
            file_name=path.name,
            relative_path=_relative_to_root(path, self.root),
            category=category,
            size_bytes=len(payload),
            sha256=digest,
            content_type="application/pdf",
            source_url=url,
            downloaded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            attachment_name=name,
            extracted_from=from_zip,
            page_count=_page_count(payload),
            **meta,
        )
        self.records.append(record)
        return record

    def _store_zip(
        self, payload: bytes, name: str, category: str,
        meta: Dict[str, str], url: str, digest: str,
    ) -> Optional[DocRecord]:
        """Pull the PDFs out of a zip; the archive itself is not kept."""
        import io

        self.seen_hashes.add(digest)
        first: Optional[DocRecord] = None
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                for info in zf.infolist():
                    if info.is_dir() or not PDF_RE.search(info.filename):
                        continue
                    if info.file_size > self._room_left():
                        break
                    inner = zf.read(info)
                    if not inner.startswith(b"%PDF"):
                        continue
                    inner_digest = hashlib.sha256(inner).hexdigest()
                    if inner_digest in self.seen_hashes:
                        self.skipped_duplicates += 1
                        continue
                    member = _decode_zip_name(info.filename)
                    rec = self._store_pdf(
                        inner, member, category, meta, url, inner_digest,
                        from_zip=name,
                    )
                    first = first or rec
        except zipfile.BadZipFile:
            self.failures.append({"url": url, "name": name, "error": "bad zip"})
        return first

    # ── Orchestration ─────────────────────────────────────────────────────

    def run(self, targets: Iterable[str], queries: List[str], per_query: int) -> None:
        for target in targets:
            for query in queries:
                if self._room_left() <= 0:
                    logger.info("Size cap reached; stopping.")
                    return
                items = self.search(target, query, max_items=per_query)
                if not items:
                    continue
                logger.info(
                    "[%s] '%s' -> %d rule(s)  (collected %.2f GB, %d file(s))",
                    target, query, len(items),
                    self.total_bytes / 1e9, len(self.records),
                )
                for item in items:
                    if self._room_left() <= 0:
                        return
                    self._handle_item(target, item)
                    time.sleep(self.delay)

    def _handle_item(self, target: str, item: Dict[str, Any]) -> None:
        title = (
            item.get("행정규칙명") or item.get("법령명한글")
            or item.get("별표명") or ""
        ).strip()
        if not title:
            return
        if not self.taxonomy.is_relevant(title):
            self.skipped_irrelevant += 1
            return
        category = self.taxonomy.classify(title)
        meta = {
            "doc_type": target,
            "title": title,
            "rule_id": str(item.get("행정규칙ID") or item.get("법령ID") or ""),
            "rule_seq": str(
                item.get("행정규칙일련번호") or item.get("법령일련번호") or ""
            ),
            "ministry": item.get("소관부처명") or "",
            "promulgation_no": str(
                item.get("발령번호") or item.get("공포번호") or ""
            ),
            "promulgation_date": str(
                item.get("발령일자") or item.get("공포일자") or ""
            ),
            "revision_type": item.get("제개정구분명") or "",
            "related_law": item.get("관련법령명") or "",
        }

        # licbyl exposes its PDF directly; the others need a detail lookup.
        direct = item.get("별표서식PDF파일링크")
        if direct:
            self.download(
                f"https://www.law.go.kr{direct}", f"{title}.pdf", category, meta
            )
            return

        seq = meta["rule_seq"]
        if not seq:
            return
        for name, url in self.attachments(target, seq):
            if PDF_RE.search(name) or ZIP_RE.search(name):
                self.download(url, name, category, meta)
                if self._room_left() <= 0:
                    return

    # ── Output ────────────────────────────────────────────────────────────

    def write_metadata(self, started: datetime) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        by_category: Dict[str, Dict[str, Any]] = {}
        for rec in self.records:
            entry = by_category.setdefault(
                rec.category, {"file_count": 0, "size_bytes": 0}
            )
            entry["file_count"] += 1
            entry["size_bytes"] += rec.size_bytes

        catalog = {
            "dataset": "Korean construction regulations for AI training",
            "schema_version": "1.0",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collection_started_utc": started.isoformat(timespec="seconds"),
            "source": {
                "name": "국가법령정보센터 (Korean Law Information Center)",
                "url": "https://www.law.go.kr",
                "api": "DRF open API",
                "note": (
                    "Public government records. robots.txt permits crawling. "
                    "KDS design standards and KCS specifications are published "
                    "as ministry administrative rules and collected as such."
                ),
            },
            "totals": {
                "file_count": len(self.records),
                "size_bytes": self.total_bytes,
                "size_gb": round(self.total_bytes / 1e9, 3),
                "duplicates_skipped": self.skipped_duplicates,
                "off_topic_rules_skipped": self.skipped_irrelevant,
                "failures": len(self.failures),
            },
            "categories": dict(sorted(by_category.items())),
            "documents": [asdict(r) for r in self.records],
        }
        _write_json(self.meta_dir / self.outputs["catalog"], catalog)

        # A flat index is easier to load in a training pipeline than the
        # full catalog.
        _write_json(
            self.meta_dir / self.outputs["stats"],
            {"categories": dict(sorted(by_category.items()))},
        )
        with open(self.meta_dir / self.outputs["index"], "w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        if self.failures:
            _write_json(self.meta_dir / self.outputs["failures"], self.failures)

        logger.info("Metadata written to %s", self.meta_dir)


# ── Helpers ───────────────────────────────────────────────────────────────

def _relative_to_root(path: Path, root: Path) -> str:
    """Paths outside the project root (custom data_dir) are kept absolute."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _safe_name(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|\r\n\t]", "_", name).strip()
    name = re.sub(r"\s+", " ", name)
    if not PDF_RE.search(name):
        name += ".pdf"
    # Keep well clear of the 255-byte filesystem limit once encoded as UTF-8.
    stem, suffix = name[:-4], name[-4:]
    while len(stem.encode("utf-8")) > 180:
        stem = stem[:-1]
    return stem + suffix


def _decode_zip_name(name: str) -> str:
    """Zip entries from Korean tools are usually CP949, not UTF-8."""
    try:
        return name.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def _page_count(payload: bytes) -> Optional[int]:
    matches = re.findall(rb"/Type\s*/Page[^s]", payload)
    return len(matches) or None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Korean construction regulation PDFs for AI training. "
                    "Defaults come from config.json; the flags below override it."
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.json (default: <root>/config.json).")
    parser.add_argument("--root", type=Path, default=project_root(),
                        help="Project root that data/ and metadata/ hang off.")
    parser.add_argument("--limit-gb", type=float, default=None,
                        help="Total download budget in GB.")
    parser.add_argument("--per-query", type=int, default=None,
                        help="Maximum rules pulled per search term.")
    parser.add_argument("--targets", nargs="+", default=None,
                        choices=["admrul", "law", "licbyl"])
    parser.add_argument("--delay", type=float, default=None,
                        help="Seconds between requests.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.root)
    law = cfg["law"]
    if args.limit_gb is not None:
        law["limit_gb"] = args.limit_gb
    if args.per_query is not None:
        law["per_query"] = args.per_query
    if args.targets is not None:
        law["targets"] = args.targets
    if args.delay is not None:
        law["delay_seconds"] = args.delay

    started = datetime.now(timezone.utc)
    collector = Collector(root=args.root, cfg=cfg)
    logger.info(
        "Collecting into %s (budget %.1f GB, config %s)",
        collector.data_dir, law["limit_gb"], cfg.get("_config_path", "built-in defaults"),
    )
    try:
        collector.run(law["targets"], law["queries"], law["per_query"])
    except KeyboardInterrupt:
        logger.warning("Interrupted - writing metadata for what was collected.")
    finally:
        collector.write_metadata(started)

    logger.info(
        "Done: %d file(s), %.2f GB, %d duplicate(s), %d off-topic rule(s), "
        "%d failure(s)",
        len(collector.records), collector.total_bytes / 1e9,
        collector.skipped_duplicates, collector.skipped_irrelevant,
        len(collector.failures),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
