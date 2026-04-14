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


NUS_CLEAR_STEM_FACULTIES = {
    "College of Design and Engineering",
    "Computing",
    "Dentistry",
    "NUS-ISS",
    "Science",
    "Yong Loo Lin Sch of Medicine",
}

NUS_CLEAR_STEM_DEPARTMENTS = {
    "Accounting",
    "Alice Lee Center for Nursing Studies",
    "Analytics and Operations",
    "Anatomy",
    "Architecture",
    "Biochemistry",
    "Biological Sciences",
    "Biomedical Engineering",
    "Built Environment",
    "Chemical and Biomolecular Engineering",
    "Chemistry",
    "Civil and Environmental Engineering",
    "Computer Science",
    "Computing and Engineering Programme",
    "Electrical and Computer Engineering",
    "Engineering Science Programme",
    "EngrgDesignandInnovationCentre",
    "Food Science and Technology",
    "Industrial Design",
    "Industrial Systems Engineering and Management",
    "Information Systems and Analytics",
    "Materials Science and Engineering",
    "Mathematics",
    "Mechanical Engineering",
    "Microbiology and Immunology",
    "NUS-ISS",
    "Pathology",
    "Pharmacology",
    "PharmacyandPharmaceuticalScience",
    "Physics",
    "Physiology",
    "Statistics and Data Science",
}

NTU_CLEAR_STEM_DEPARTMENTS = {
    "Aerospace Engineering (MAE)",
    "Bachelor of Applied Computing in Finance (CE)",
    "Bioengineering (CBE)",
    "Biological Sciences (general)",
    "Biological Sciences Programme",
    "Biomedical Sciences (Biological Sciences)",
    "Chemistry (CBE)",
    "Chinese Medicine (Biological Sciences)",
    "Computer Science (Computer Engineering track, SCSE)",
    "Environmental Engineering (CEE)",
    "Information Engineering & Media (EEE)",
    "Physics (SPMS)",
    "Programme under Civil Engineering",
    "Renaissance Engineering Programme",
    "Robotics (MAE)",
    "School of Biological Sciences",
    "School of Chemistry, Chemical Engineering & Biotechnology",
    "School of Civil and Environmental Engineering",
    "School of Civil and Environmental Engineering (CEE)",
    "School of Computer Science and Engineering (SCSE)",
    "School of Electrical and Electronic Engineering",
    "School of Materials Science and Engineering (College of Engineering)",
    "School of Mechanical and Aerospace Engineering",
    "School of Physical & Mathematical Sciences",
    "School of Physical and Mathematical Sciences (College of Science)",
}

SUTD_UNCLEAR_DEPARTMENTS = {
    "Humanities, Arts and Social Sciences",
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


def _classify_values(
    values: list[str],
    clear_stem: set[str] | None = None,
    clear_non_stem: set[str] | None = None,
    unclear_or_mixed: set[str] | None = None,
    default_bucket: str = "unclear_or_mixed",
    include_clear_non_stem: bool = False,
) -> dict[str, list[str]]:
    buckets = {
        "clear_stem": [],
        "unclear_or_mixed": [],
    }
    if include_clear_non_stem:
        buckets["clear_non_stem"] = []
    clear_stem = clear_stem or set()
    clear_non_stem = clear_non_stem or set()
    unclear_or_mixed = unclear_or_mixed or set()

    for value in values:
        if value in clear_stem:
            bucket = "clear_stem"
        elif include_clear_non_stem and value in clear_non_stem:
            bucket = "clear_non_stem"
        elif value in unclear_or_mixed:
            bucket = "unclear_or_mixed"
        else:
            bucket = default_bucket
        buckets[bucket].append(value)
    for k in buckets:
        buckets[k] = sorted(set(buckets[k]))
    return buckets


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_nus(nus_rows: list[dict[str, Any]]) -> dict[str, Any]:
    faculties = _extract_unique_values(nus_rows, "faculty")
    departments = _extract_unique_values(nus_rows, "department")
    return {
        "version": "v1",
        "source_file": "data/cleaned_data/combined_courses_cleaned.pkl",
        "notes": [
            "Auto-generated by src/stem_test/stem_1_generate_scope_classifications.py.",
            "Built from notebook-cleaned combined course PKL rows filtered to NUS.",
            "NUS classification uses explicit rules.",
            "Faculties and departments listed in clear_stem are treated as STEM.",
            "All remaining faculties/departments default to unclear_or_mixed.",
        ],
        "faculties": _classify_values(
            faculties,
            clear_stem=NUS_CLEAR_STEM_FACULTIES,
            default_bucket="unclear_or_mixed",
        ),
        "departments": _classify_values(
            departments,
            clear_stem=NUS_CLEAR_STEM_DEPARTMENTS,
            default_bucket="unclear_or_mixed",
        ),
    }


def build_ntu(ntu_rows: list[dict[str, Any]]) -> dict[str, Any]:
    faculties = _extract_unique_values(ntu_rows, "faculty")
    departments = _extract_unique_values(ntu_rows, "department")
    return {
        "version": "v1",
        "source_file": "data/cleaned_data/combined_courses_cleaned.pkl",
        "notes": [
            "Auto-generated by src/stem_test/stem_1_generate_scope_classifications.py.",
            "Built from notebook-cleaned combined course PKL rows filtered to NTU.",
            "NTU classification uses explicit rules.",
            "Departments listed in clear_stem are treated as STEM.",
            "All remaining faculties/departments default to unclear_or_mixed.",
        ],
        "faculties": _classify_values(
            faculties,
            clear_stem=set(),
            default_bucket="unclear_or_mixed",
        ),
        "departments": _classify_values(
            departments,
            clear_stem=NTU_CLEAR_STEM_DEPARTMENTS,
            default_bucket="unclear_or_mixed",
        ),
    }


def build_sutd(sutd_rows: list[dict[str, Any]]) -> dict[str, Any]:
    departments = _extract_unique_values(sutd_rows, "department")
    clear_stem_departments = {d for d in departments if d not in SUTD_UNCLEAR_DEPARTMENTS}
    buckets = _classify_values(
        departments,
        clear_stem=clear_stem_departments,
        unclear_or_mixed=SUTD_UNCLEAR_DEPARTMENTS,
        default_bucket="unclear_or_mixed",
    )
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
            "SUTD classification uses explicit rules.",
            "All departments are treated as STEM except Humanities, Arts and Social Sciences.",
            "Humanities, Arts and Social Sciences is set to unclear_or_mixed.",
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
