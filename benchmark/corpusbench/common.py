"""
Shared plumbing for the benchmark pipeline.

Settings resolve in three layers, each overriding the one before: the defaults
below, then a config file, then the command line. So a user can keep one
config.json per dataset variant and still redirect a single run with a flag.

Everything is deterministic. Given the same corpus and seed, every builder
writes byte-identical output — which is the whole point when the number you
care about is the difference between two model checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

# Anchor for config.json and every relative path in it. The modules live in
# a package one level down, so this is the package's parent, not its own
# directory.
HERE = Path(__file__).resolve().parent.parent
PROJECT = HERE.parent

# Dataset identity, carried in every artefact and every item. A score is only
BENCHMARK_NAME = "kcbench"
BENCHMARK_VERSION = "v2"
SCHEMA_VERSION = "kcbench-item-v2"

DEFAULTS: Dict[str, Any] = {
    # Where the pipeline reads from. All four accept absolute paths, or paths
    # relative to this directory.
    "corpus_dir": "../data",              # source PDFs and IFC models
    "generated_dir": "../ai_ready_full",  # AI-ready training data being evaluated
    "metadata_dir": "../metadata",        # collector catalogues
    "pipeline_dir": "../gen_aec_syn_data",  # supplies the chunker, so chunking matches
    "out_dir": "data",                    # benchmark artefacts
    "holdout": {
        "chunk_fraction": 0.10,   # share of PDF chunks reserved, per category
        "ifc_fraction": 0.25,     # share of IFC models reserved for track 3
        "max_chunks_per_doc": 50,
        "seed": 20260814,
        # Parser regression fixtures — files named for the crash they reproduce
        "ifc_exclude_categories": ["06_reference_and_test"],
        "ifc_min_element_types": 2,
    },
    "track2_sft": {
        "target_items": 400,
        "max_items_per_doc": 4,
        # strict  — admit a threshold only when its unit and qualifier occur once
        "numeric_uniqueness": "strict",
        "min_subject_len": 4,
        "max_subject_len": 30,
        "nameset_min_items": 3,
        "nameset_max_item_len": 60,
        # An enumeration is only answerable if the text says what it enumerates.
        "nameset_require_lead_in": True,
    },
    "track3_vlm": {
        "whole_model_groups_only": True,
        "min_types_for_mapping": 3,
        # Counting is a fair question on a house and an unfair one on a model
        "mapping_max_total_elements": 60,
    },
    "probe": {
        # Items mined from the training side, to separate "did it learn what we
        # taught it" from track 2's "does it generalise to regulation it never
        # saw". Contamination is deliberate here and every row says so.
        "target_items": 400,
        "max_items_per_doc": 6,
        "nameset_share": 0.2,
        "seed": 20260814,
    },
    "export": {
        "lm_eval_harness": True,
        "dataset_card": True,
    },
    "eval": {
        "repeats": 3,
        "numeric_tolerance": 0.02,
        "ollama_base_url": "http://localhost:11434",
        "request_timeout": 600,
        # Without this the server sizes the KV cache from the model's maximum
        "num_ctx": 8192,
        # Generation budget. A reasoning model spends most of it thinking before
        "num_predict": 4096,
        "temperature": 0.0,
        # Models to score when run_matrix.py is given none on the command line.
        "models": [],
        "book": "both",
        # Prompt-language ratio for --lang mix. Korean is the operating language
        # of the agent; English keeps a regression check on instruction-following.
        "lang_mix": {"ko": 0.8, "en": 0.2},
    },
    "perplexity": {
        # Track 1 runs the weights directly, so these are the knobs that decide
        # whether it fits in memory and how long it takes.
        "max_length": 2048,
        "dtype": "bfloat16",
        "device": "cuda",
        "log_every": 200,
    },
    "compare": {
        "alpha": 0.05,
        "bootstrap_rounds": 10000,
        "bootstrap_seed": 20260814,
    },
    "triage": {
        "min_models": 2,
        "sample": 30,
        "sample_seed": 20260814,
        "no_answer_threshold": 0.5,
    },
}

PATH_KEYS = ("corpus_dir", "generated_dir", "metadata_dir", "pipeline_dir", "out_dir")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def log(name: str) -> logging.Logger:
    return logging.getLogger(name)


# CLI

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flags every stage of the pipeline understands."""
    g = parser.add_argument_group("paths (override config.json)")
    g.add_argument("-c", "--config", metavar="FILE",
                   help="settings file; defaults to ./config.json when present")
    g.add_argument("-i", "--generated-dir", metavar="DIR",
                   help="AI-ready training data to evaluate (default ../ai_ready_full)")
    g.add_argument("--corpus-dir", metavar="DIR",
                   help="source corpus of PDFs and IFC models (default ../data)")
    g.add_argument("--metadata-dir", metavar="DIR",
                   help="collector catalogues (default ../metadata)")
    g.add_argument("--pipeline-dir", metavar="DIR",
                   help="gen_aec_syn_data checkout, used for its chunker")
    g.add_argument("-o", "--out-dir", metavar="DIR",
                   help="where benchmark artefacts are written (default ./data)")
    parser.add_argument("--seed", type=int, help="override the holdout seed")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def resolve_config(args: argparse.Namespace | None = None) -> Dict[str, Any]:
    """Defaults, then the config file, then anything given on the command line."""
    cfg = copy.deepcopy(DEFAULTS)

    path = Path(args.config) if (args and args.config) else HERE / "config.json"
    if path.is_file():
        user = json.loads(path.read_text(encoding="utf-8"))
        cfg = _deep_merge(cfg, {k: v for k, v in user.items() if not k.startswith("_")})
        cfg["_config_path"] = str(path)
    elif args and args.config:
        raise SystemExit(f"config file not found: {path}")

    if args:
        for key in PATH_KEYS:
            val = getattr(args, key, None)
            if val:
                cfg[key] = val
        if getattr(args, "seed", None) is not None:
            cfg["holdout"]["seed"] = args.seed
        if getattr(args, "verbose", False):
            logging.getLogger().setLevel(logging.DEBUG)

    for key in PATH_KEYS:
        p = Path(cfg[key])
        cfg[key] = p.resolve() if p.is_absolute() else (HERE / p).resolve()
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    for key in ("corpus_dir", "generated_dir", "metadata_dir"):
        if not cfg[key].exists():
            log("config").warning("%s does not exist: %s", key, cfg[key])
    return cfg


def describe(cfg: Dict[str, Any]) -> None:
    lg = log("config")
    lg.info("config      %s", cfg.get("_config_path", "built-in defaults"))
    lg.info("corpus      %s", cfg["corpus_dir"])
    lg.info("generated   %s", cfg["generated_dir"])
    lg.info("out         %s", cfg["out_dir"])


# I/O

def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# identity, hashing, normalisation

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def item_id(track: str, *parts: Any) -> str:
    """
    A stable id derived from what the item *is*, not from its position.

    Sequential ids renumber the whole file when one item is added upstream, and
    two runs then disagree about what `t2-0007` refers to. Hashing the content
    keeps an id attached to its question for as long as that question exists.
    """
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{track}-{digest[:12]}"


_PUNCT = re.compile(r"[\s ]+")
_TRAILING = " \t.,;:·，、。()[]{}<>\"'“”‘’"


def normalise(text: str) -> str:
    """
    Comparison form for set-valued answers.

    Korean regulation is copied out of PDFs with full-width punctuation, hard
    spaces and inconsistent spacing around particles. Comparing raw strings
    scores a correct answer wrong for a trailing full stop, which measures
    transcription rather than knowledge.
    """
    s = unicodedata.normalize("NFKC", text)
    s = _PUNCT.sub(" ", s).strip(_TRAILING)
    return s.casefold()


# Units as they are written in Korean regulation, and the symbol an English
UNIT_EN = {
    "mm": "mm", "cm": "cm", "m": "m", "㎡": "m2", "㎥": "m3", "kg": "kg",
    "톤": "t", "kN": "kN", "MPa": "MPa", "%": "%", "일": "days", "개월": "months",
    "년": "years", "명": "persons", "회": "times", "배": "times", "층": "storeys",
    "시간": "hours", "분": "minutes", "원": "KRW",
}

QUALIFIER_EN = {"이상": "at least", "이하": "at most", "미만": "less than",
                "초과": "more than", "이내": "within"}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest_entry(path: Path) -> Dict[str, Any]:
    """Checksum and size for one artefact, so a build can be proven identical."""
    return {"file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "lines": sum(1 for _ in path.open(encoding="utf-8")) if path.suffix == ".jsonl" else None}


# Corpus access

def corpus_documents(cfg: Dict[str, Any]) -> List[Path]:
    """Every PDF in the corpus. Suffix match is case-insensitive."""
    return sorted(p for p in cfg["corpus_dir"].rglob("*")
                  if p.is_file() and p.suffix.lower() == ".pdf")


def source_metadata(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Collector metadata keyed by path relative to the project root."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in ("index.jsonl", "index_codil.jsonl"):
        path = cfg["metadata_dir"] / name
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            out[rec["relative_path"]] = rec
    return out


def rel(cfg: Dict[str, Any], path: Path | str) -> str:
    """Corpus-relative id, e.g. data/03_safety_and_disaster/rule.pdf."""
    p = Path(path).resolve()
    for base in (cfg["corpus_dir"].parent, PROJECT):
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    return str(p)


DAPT_FILE = "dapt_training_data.jsonl"
SFT_FILE = "sllm_training_data.jsonl"


def generated_documents(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    One record per document that produced training data.

    Chunk text is read from the generated DAPT file rather than re-extracted
    from the PDF. That is both far faster — no re-parsing, no OCR of scanned
    documents — and more accurate: the benchmark then describes exactly the text
    a model was trained on, rather than a second extraction that might differ.
    """
    docs: List[Dict[str, Any]] = []
    for path in sorted(cfg["generated_dir"].rglob(DAPT_FILE)):
        try:
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        except Exception as exc:
            log("corpus").warning("unreadable, skipping: %s (%s)", path, exc)
            continue
        # Row index and digest travel with the text. An evaluation item has to be
        chunk_rows = [{"row_index": i, "text": r["text"], "sha256": sha256_text(r["text"])}
                      for i, r in enumerate(rows) if r.get("text")]
        if not chunk_rows:
            continue
        folder = path.parent
        docs.append({
            "stem": folder.name,
            "category": folder.relative_to(cfg["generated_dir"]).parts[0],
            "source_name": rows[0].get("source_name", folder.name + ".pdf"),
            "source_org": rows[0].get("source_org", ""),
            "source_date": str(rows[0].get("source_date", "")),
            "generated_dir": str(folder.relative_to(cfg["generated_dir"])),
            "dataset_file": str(path.relative_to(cfg["generated_dir"])),
            "chunks": [c["text"] for c in chunk_rows],
            "chunk_rows": chunk_rows,
        })
    return docs


def corpus_path_for(cfg: Dict[str, Any], doc: Dict[str, Any]) -> str | None:
    """Locate the source PDF a generated document came from, if it is still there."""
    cand = cfg["corpus_dir"] / doc["generated_dir"]
    for p in (cand.with_suffix(".pdf"), cand.parent / doc["source_name"]):
        if p.is_file():
            return rel(cfg, p)
    hits = list((cfg["corpus_dir"] / doc["category"]).rglob(doc["stem"] + ".pdf"))
    return rel(cfg, hits[0]) if hits else None


def extractor(cfg: Dict[str, Any]):
    """The pipeline's own extractor, so chunking matches the training data."""
    sys.path.insert(0, str(cfg["pipeline_dir"]))
    from src.config import PipelineConfig       # noqa: PLC0415
    from src.pdf_extractor import PDFExtractor   # noqa: PLC0415
    return PDFExtractor(PipelineConfig.load_default(str(cfg["pipeline_dir"] / "config.json")))


def iter_chunks(cfg: Dict[str, Any], paths: Iterator[Path]):
    """(path, chunks) per document, skipping those that yield nothing."""
    ex = extractor(cfg)
    for p in paths:
        try:
            chunks = ex.extract_chunks(Path(p))
        except Exception as exc:          # a broken PDF is not fatal to a build
            log("corpus").warning("extract failed: %s (%s)", Path(p).name, exc)
            continue
        if chunks:
            yield Path(p), chunks
