"""
Configuration loading for the collectors.

Everything a user is expected to change - download budgets, search terms,
the category taxonomy, source lists, output folder names - lives in
``config.json`` at the project root. The defaults below are the fallback used
when no config file is present, and they are also what ``config.json`` is
generated from, so the two can never drift.

Resolution order for the config file:

    1. ``--config <path>`` on the command line
    2. ``$AI_DATASET_CONFIG``
    3. ``<project root>/config.json``
    4. the built-in defaults

Values found in the file are deep-merged onto the defaults, so a config file
only has to carry the keys it actually overrides.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_FILENAME = "config.json"
ENV_VAR = "AI_DATASET_CONFIG"

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULTS: Dict[str, Any] = {
    "paths": {
        # Relative to the project root unless given as an absolute path.
        "data_dir": "data",
        "metadata_dir": "metadata",
        "certs_dir": "certs",
    },
    "http": {
        "user_agent": "construction-ai-dataset/1.0 (research; contact you@example.com)",
        "api_timeout_seconds": 40,
        "download_timeout_seconds": 120,
        "max_retries": 3,
    },
    # Shared by the law.go.kr and CODIL collectors: what counts as a
    # construction document, and which folder it lands in.
    "taxonomy": {
        "relevance": {
            # A title must match include_pattern and must not match
            # exclude_pattern. Full-text search returns rules that merely
            # mention construction; the title has to carry the subject itself.
            "include_pattern": (
                "건설|시공|공사|토목|건축|구조물|설계기준|시방|표준품셈|적산|감리|건설사업관리|"
                "콘크리트|강구조|철골|지반|기초|말뚝|흙막이|굴착|사면|옹벽|"
                "교량|터널|댐|하천|항만|어항|철도|도로|포장공|상수도|하수도|수도관|관로|"
                "가설|비계|거푸집|조경|설비|전기설비|기계설비|승강기|소방시설|"
                "내진|안전점검|정밀안전진단|시설물|유지관리|해체|리모델링|주택|건물|"
                "BIM|건설정보|스마트건설|모듈러|건설기계|건설기술|엔지니어링|건설재료|골재|아스팔트"
            ),
            "exclude_pattern": (
                "구조견|항공대|병역|참사|포장재\\s*재활용|재활용의무|유해화학|검역|"
                "수의사|가축|축산|어선|선박검사|해기사|항공기|무인기|"
                "문화재수리|국가유산수리|산림|목재|농업기계|훈련교재|교육원|"
                "공무원|인사|보수규정|복무|승진|포상|표창"
            ),
        },
        # Checked in order; the first pattern matching the title wins.
        "categories": [
            {"name": "01_design_standards_kds", "pattern": "KDS|설계기준|구조기준|내진설계"},
            {"name": "02_specifications_kcs", "pattern": "KCS|시방서|시방기준"},
            {"name": "03_safety_and_disaster", "pattern": "안전|재해|재난|방재|위험|점검|진단"},
            {"name": "04_quality_and_inspection", "pattern": "품질|검사|시험|인증|성능평가"},
            {"name": "05_structures_and_facilities",
             "pattern": "교량|터널|댐|항만|하천|철도|도로|상수도|하수도|옹벽|사면"},
            {"name": "06_building_and_architecture",
             "pattern": "건축|건물|주택|리모델링|용도변경|피난|방화"},
            {"name": "07_environment_and_energy",
             "pattern": "환경|소음|진동|대기|폐기물|에너지|녹색|탄소"},
            {"name": "08_contract_and_cost",
             "pattern": "계약|입찰|적산|원가|대가|낙찰|하도급|공사비"},
            {"name": "09_bim_and_smart_construction",
             "pattern": "BIM|스마트|정보모델|디지털|건설정보|자동화"},
            {"name": "10_equipment_and_services",
             "pattern": "설비|승강기|기계|전기|배관|덕트|공조"},
            {"name": "11_general_construction",
             "pattern": "건설|시공|공사|토목|엔지니어링|감리|품셈|적산"},
        ],
        # Documents that passed the relevance gate but matched no theme above.
        "fallback_category": "12_other_construction",
    },
    # ── collect.py : 국가법령정보센터 (law.go.kr) ──────────────────────────
    "law": {
        "api_base": "https://www.law.go.kr/DRF",
        # law.go.kr DRF account id. "test" works for the open sandbox; register
        # your own id at law.go.kr/DRF for production-rate access.
        "oc": "test",
        "limit_gb": 10.0,
        "per_query": 100,
        "delay_seconds": 0.4,
        # admrul = administrative rules (KDS/KCS live here)
        # law    = acts, decrees, enforcement rules
        # licbyl = attached tables and forms
        "targets": ["admrul", "licbyl"],
        # Subfolder of data/ that the category folders are created under.
        # "" puts them straight in data/.
        "output_subdir": "",
        "queries": [
            "건설", "시공", "공사", "토목", "건축", "구조", "설계기준", "표준시방서",
            "시방서", "콘크리트", "강구조", "지반", "기초", "터널", "교량", "댐",
            "하천", "항만", "어항", "철도", "도로", "포장", "상수도", "하수도",
            "가설", "조경", "설비", "전기설비", "기계설비", "소방", "내진",
            "건설안전", "건설기계", "건설공사 품질", "감리", "건설사업관리",
            "건축물 유지관리", "시설물 안전", "정밀안전진단", "해체공사",
            "흙막이", "사면", "옹벽", "말뚝", "지하안전", "굴착",
            "적산", "표준품셈", "공사원가", "건설기술", "엔지니어링",
            "BIM", "건설정보모델", "스마트건설", "모듈러",
        ],
        "outputs": {
            "catalog": "catalog.json",
            "index": "index.jsonl",
            "stats": "index.jsonl_stats.json",
            "failures": "failures.json",
        },
    },
    # ── collect_codil.py : CODIL 건설기술정보시스템 ────────────────────────
    "codil": {
        "base_url": "https://www.codil.or.kr",
        "limit_gb": 9.0,
        "max_pages": 400,
        "delay_seconds": 1.0,
        # CODIL's own `gubun` section codes.
        "sections": {
            "rpt": "research_reports",
            "bus": "business_documents",
            "tch": "technical_documents",
        },
        "enabled_sections": ["rpt", "tch", "bus"],
        "output_subdir": "50_research_reports",
        # true  -> data/<output_subdir>/<section>/<category>
        # false -> data/<output_subdir>/<category>
        "group_by_section": False,
        # The server omits its intermediate certificate; it is fetched from the
        # leaf's AIA extension and appended to the system roots. Set
        # ca_bundle_path to use your own bundle instead.
        "tls": {
            "build_ca_bundle": True,
            "system_ca_bundle": "/etc/ssl/certs/ca-certificates.crt",
            "ca_bundle_path": "",
            "host": "www.codil.or.kr",
            "port": 443,
        },
        "outputs": {
            "catalog": "catalog_codil.json",
            "index": "index_codil.jsonl",
            "failures": "failures_codil.json",
        },
    },
    # ── collect_ifc.py : public IFC/BIM models from GitHub ─────────────────
    "ifc": {
        "github_api": "https://api.github.com",
        # Optional personal access token; raises the API rate limit from 60 to
        # 5000 requests/hour. Leave empty to read $GITHUB_TOKEN instead.
        "github_token": "",
        "limit_gb": 1.5,
        "delay_seconds": 0.3,
        "output_subdir": "40_bim_models_ifc",
        "small_file_bytes": 40960,
        "repositories": [
            {"repo": "buildingSMART/Sample-Test-Files",
             "note": "buildingSMART official sample and conformance files"},
            {"repo": "IfcOpenShell/files",
             "note": "IfcOpenShell public test files"},
            {"repo": "andrewisen/bim-whale-ifc-samples",
             "note": "BIM Whale IFC samples"},
            {"repo": "jul-ls/sample-ifc-files",
             "note": "Community IFC sample collection"},
            {"repo": "atomczak/Sample-BIM-Files",
             "note": "Sample BIM files (IFC, IDS, BCF, mvdXML)"},
            {"repo": "openBIMstandards/DataSetSchependomlaan",
             "note": "Schependomlaan open BIM dataset"},
            {"repo": "Rkastro07/bim-analysis-sample-data",
             "note": "BIM analysis sample data"},
        ],
        # Matched against the file path and the parsed IFC header, in order.
        "categories": [
            {"name": "02_infrastructure",
             "pattern": ("bridge|road|rail|tunnel|highway|viaduct|metro|port|harbou?r|"
                         "alignment|infra|street|overpass")},
            {"name": "03_facilities_and_mep",
             "pattern": ("plant|factory|industrial|mep|hvac|duct|pipe|electrical|plumbing|"
                         "mechanical|fire|sprinkler|substation|treatment")},
            {"name": "04_structural",
             "pattern": "structur|steel|concrete|frame|beam|column|foundation|rebar|truss"},
            {"name": "05_site_and_civil",
             "pattern": "site|terrain|landscap|topograph|earthwork|excavat|geotech|survey"},
            {"name": "01_buildings",
             "pattern": ("building|house|home|villa|office|clinic|hospital|school|apartment|"
                         "residential|tower|hotel|museum|duplex|smiley|institute|arch")},
        ],
        "fallback_category": "06_reference_and_test",
        "outputs": {
            "catalog": "catalog_ifc.json",
            "index": "index_ifc.jsonl",
            "failures": "failures_ifc.json",
        },
    },
}


# ── Loading ───────────────────────────────────────────────────────────────

def project_root() -> Path:
    """The repository root, i.e. the parent of scripts/."""
    return Path(__file__).resolve().parent.parent


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dicts recursively; lists and scalars are replaced wholesale."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_config_path(explicit: Optional[Path], root: Path) -> Optional[Path]:
    for candidate in (explicit, os.environ.get(ENV_VAR), root / CONFIG_FILENAME):
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    return None


def load_config(explicit: Optional[Path] = None,
                root: Optional[Path] = None) -> Dict[str, Any]:
    """Defaults, with config.json merged on top if one is found."""
    root = root or project_root()
    path = resolve_config_path(explicit, root)
    if path is None:
        return copy.deepcopy(DEFAULTS)
    with open(path, encoding="utf-8") as fh:
        user = json.load(fh)
    cfg = _deep_merge(DEFAULTS, user)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_dir(root: Path, value: str) -> Path:
    """Config paths are relative to the project root unless absolute."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def write_default_config(path: Path) -> Path:
    """Dump the built-in defaults so a user has a full file to edit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(DEFAULTS, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


if __name__ == "__main__":  # python scripts/config.py  -> writes config.json
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root() / CONFIG_FILENAME
    print(f"Wrote {write_default_config(target)}")
