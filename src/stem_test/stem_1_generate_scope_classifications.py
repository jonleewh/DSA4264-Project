import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COURSES_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "combined_courses_cleaned.pkl"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

NUS_OUT = DEFAULT_REFERENCE_DIR / "nus_stem_classification_v1.json"
NTU_OUT = DEFAULT_REFERENCE_DIR / "ntu_stem_classification_v1.json"
SUTD_OUT = DEFAULT_REFERENCE_DIR / "sutd_stem_classification_v1.json"


STEM_KEYWORDS = [
    "engineering",
    "engineer",
    "computer",
    "computing",
    "information systems",
    "analytics",
    "operations",
    "mathematics",
    "math",
    "statistics",
    "statistical",
    "physics",
    "chemistry",
    "biological",
    "biology",
    "biochemistry",
    "microbiology",
    "immunology",
    "anatomy",
    "physiology",
    "materials",
    "medicine",
    "medical",
    "dentistry",
    "pharmacy",
    "science",
    "public health",
    "architecture",
    "built environment",
]

NON_STEM_KEYWORDS = [
    "business",
    "accounting",
    "accountancy",
    "finance",
    "law",
    "arts",
    "social science",
    "humanities",
    "history",
    "philosophy",
    "language",
    "linguistics",
    "music",
    "conservatory",
    "sociology",
    "psychology",
    "political",
    "policy",
    "marketing",
    "communications",
    "communication",
    "media",
    "real estate",
    "management",
    "economics",
]


NUS_FACULTY_OVERRIDES = {
    "College of Design and Engineering": "clear_stem",
    "Computing": "clear_stem",
    "Dentistry": "clear_stem",
    "NUS-ISS": "clear_stem",
    "SSH School of Public Health": "clear_stem",
    "Science": "clear_stem",
    "Yong Loo Lin Sch of Medicine": "clear_stem",
    "Arts and Social Science": "clear_non_stem",
    "Law": "clear_non_stem",
    "YST Conservatory of Music": "clear_non_stem",
    "NUS Business School": "unclear_or_mixed",
    "Yale-NUS College": "unclear_or_mixed",
    "Residential College": "unclear_or_mixed",
}

NUS_DEPARTMENT_OVERRIDES = {
    "Economics": "clear_stem",
    "Analytics and Operations": "clear_stem",
    "Yale-NUS College": "unclear_or_mixed",
}

NTU_DEPARTMENT_OVERRIDES = {
    "Economics": "unclear_or_mixed",
    "Psychology": "unclear_or_mixed",
    "National Institute of Education": "unclear_or_mixed",
}


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_course_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned courses PKL not found: {path}")

    df = pd.read_pickle(path)
    required_cols = {"code", "department", "university"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required course columns in {path}: {missing}"
        )
    return [row for row in df.to_dict("records") if isinstance(row, dict)]


def _filter_by_university(rows: list[dict[str, Any]], university: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _norm_text(row.get("university")).upper() == university.upper()
    ]


def _extract_unique_values(rows: list[dict[str, Any]], key: str) -> list[str]:
    out: set[str] = set()
    for row in rows:
        value = row.get(key)
        if isinstance(value, str):
            cleaned = _norm_text(value)
            if cleaned:
                out.add(cleaned)
            continue
        if isinstance(value, list):
            for item in value:
                cleaned = _norm_text(item)
                if cleaned:
                    out.add(cleaned)
    return sorted(out)


def _score_keywords(label: str, keywords: list[str]) -> int:
    text = label.lower()
    return sum(1 for kw in keywords if kw in text)


def _bucket_from_keywords(label: str) -> str:
    stem_score = _score_keywords(label, STEM_KEYWORDS)
    non_stem_score = _score_keywords(label, NON_STEM_KEYWORDS)
    if stem_score > 0 and non_stem_score == 0:
        return "clear_stem"
    if non_stem_score > 0 and stem_score == 0:
        return "clear_non_stem"
    return "unclear_or_mixed"


def _classify_values(values: list[str], overrides: dict[str, str] | None = None) -> dict[str, list[str]]:
    buckets = {
        "clear_stem": [],
        "clear_non_stem": [],
        "unclear_or_mixed": [],
    }
    for value in values:
        if overrides and value in overrides:
            bucket = overrides[value]
        else:
            bucket = _bucket_from_keywords(value)
        buckets[bucket].append(value)
    for k in buckets:
        buckets[k] = sorted(set(buckets[k]))
    return buckets


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_nus(nus_rows: list[dict[str, Any]]) -> dict[str, Any]:
    departments = _extract_unique_values(nus_rows, "department")
    return {
        "version": "v1",
        "source_file": "data/cleaned_data/combined_courses_cleaned.pkl",
        "notes": [
            "Auto-generated by src/stem_test/stem_1_generate_scope_classifications.py.",
            "Built from notebook-cleaned combined course PKL rows filtered to NUS.",
            "NUS faculty buckets are left empty because the cleaned PKL does not preserve a faculty column.",
            "Classification uses keyword-based rules plus explicit overrides for known ambiguous departments.",
        ],
        "faculties": {
            "clear_stem": [],
            "clear_non_stem": [],
            "unclear_or_mixed": [],
        },
        "departments": _classify_values(departments, overrides=NUS_DEPARTMENT_OVERRIDES),
    }


def build_ntu(ntu_rows: list[dict[str, Any]]) -> dict[str, Any]:
    departments = _extract_unique_values(ntu_rows, "department")
    return {
        "version": "v1",
        "source_file": "data/cleaned_data/combined_courses_cleaned.pkl",
        "notes": [
            "Auto-generated by src/stem_test/stem_1_generate_scope_classifications.py.",
            "Built from notebook-cleaned combined course PKL rows filtered to NTU.",
            "Department-only classification for NTU.",
        ],
        "departments": _classify_values(departments, overrides=NTU_DEPARTMENT_OVERRIDES),
    }


def build_sutd(sutd_rows: list[dict[str, Any]]) -> dict[str, Any]:
    departments = _extract_unique_values(sutd_rows, "department")
    buckets = _classify_values(departments)
    if not departments:
        notes = [
            "Auto-generated by src/stem_test/stem_1_generate_scope_classifications.py.",
            "Built from notebook-cleaned combined course PKL rows filtered to SUTD.",
            "No non-empty department values found in data/cleaned_data/combined_courses_cleaned.pkl for SUTD.",
            "SUTD rows default to unclear_or_mixed when department is missing.",
        ]
    else:
        notes = [
            "Auto-generated by src/stem_test/stem_1_generate_scope_classifications.py.",
            "Built from notebook-cleaned combined course PKL rows filtered to SUTD.",
            "Department-only classification for SUTD.",
        ]

    return {
        "version": "v1",
        "source_file": "data/cleaned_data/combined_courses_cleaned.pkl",
        "notes": notes,
        "departments": buckets,
        "fallback_policy": {
            "if_department_missing": "unclear_or_mixed",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NUS/NTU/SUTD STEM scope classification reference JSON files.",
    )
    parser.add_argument("--courses-input", type=Path, default=DEFAULT_COURSES_INPUT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_dir = args.reference_dir
    all_rows = _load_course_rows(args.courses_input)

    nus_rows = _filter_by_university(all_rows, "NUS")
    ntu_rows = _filter_by_university(all_rows, "NTU")
    sutd_rows = _filter_by_university(all_rows, "SUTD")

    nus_obj = build_nus(nus_rows)
    ntu_obj = build_ntu(ntu_rows)
    sutd_obj = build_sutd(sutd_rows)

    nus_out = reference_dir / NUS_OUT.name
    ntu_out = reference_dir / NTU_OUT.name
    sutd_out = reference_dir / SUTD_OUT.name

    if args.dry_run:
        print(f"NUS faculties: {sum(len(v) for v in nus_obj['faculties'].values())}")
        print(f"NUS departments: {sum(len(v) for v in nus_obj['departments'].values())}")
        print(f"NTU departments: {sum(len(v) for v in ntu_obj['departments'].values())}")
        print(f"SUTD departments: {sum(len(v) for v in sutd_obj['departments'].values())}")
        return

    _write_json(nus_out, nus_obj)
    _write_json(ntu_out, ntu_obj)
    _write_json(sutd_out, sutd_obj)

    print(f"Generated {nus_out}")
    print(f"Generated {ntu_out}")
    print(f"Generated {sutd_out}")


if __name__ == "__main__":
    main()
