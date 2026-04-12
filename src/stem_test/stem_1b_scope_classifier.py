import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DEFAULT_CLEANED_MODULE_INPUT = PROJECT_ROOT / "data" / "cleaned_module_rows.jsonl"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_STEM_OUTPUT = PROJECT_ROOT / "data" / "cleaned_module_rows_STEM.jsonl"

NUS_FILE = REFERENCE_DIR / "nus_stem_classification_v1.json"
NTU_FILE = REFERENCE_DIR / "ntu_stem_classification_v1.json"
SUTD_FILE = REFERENCE_DIR / "sutd_stem_classification_v1.json"


def _norm(value: str | None) -> str:
    return (value or "").strip()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Classification file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _to_set(values: list[str] | None) -> set[str]:
    return {_norm(v) for v in (values or []) if _norm(v)}


QUANT_STRONG_PATTERNS = [
    r"\bstatistic(s|al)?\b",
    r"\beconometric(s)?\b",
    r"\bregression\b",
    r"\bprobability\b",
    r"\bstochastic\b",
    r"\btime series\b",
    r"\boptimization|optimisation\b",
    r"\boperations research\b",
    r"\blinear algebra\b",
    r"\bcalculus\b",
    r"\bnumerical (method|methods)\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bneural network(s)?\b",
    r"\bdata mining\b",
    r"\bforecast(ing)?\b",
    r"\bsimulation\b",
    r"\balgorithm(s|ic)?\b",
    r"\bpython\b",
    r"\br programming\b|\bprogramming in r\b|\blanguage r\b",
    r"\bsql\b",
    r"\bprogramming\b",
]

QUANT_WEAK_PATTERNS = [
    r"\bquantitative\b",
    r"\banalytics?\b",
    r"\bmathematical model(l)?ing\b",
    r"\bcomputational\b",
    r"\bempirical\b",
    r"\bdata analysis\b",
]

QUANT_BLOCKLIST_PATTERNS = [
    r"\bliterary analysis\b",
    r"\bfilm analysis\b",
    r"\bcritical analysis\b",
    r"\btextual analysis\b",
    r"\bhistorical analysis\b",
    r"\bphilosophical analysis\b",
]

EXCLUDED_MODULE_PATTERNS: list[tuple[str, str]] = [
    (
        "excluded_future_ready_graduates_module",
        r"\bcent(re|er) for future[- ]ready graduates\b|\bcfrg\b",
    ),
]


def _exclusion_reason(
    title: str | None,
    description: str | None,
    department: str | None = None,
    faculty: str | None = None,
) -> str | None:
    text = " ".join(
        [
            _norm(title).lower(),
            _norm(description).lower(),
            _norm(department).lower(),
            _norm(faculty).lower(),
        ]
    )
    if not text.strip():
        return None

    for reason, pattern in EXCLUDED_MODULE_PATTERNS:
        if re.search(pattern, text):
            return reason
    return None


def _quant_signal_score(title: str | None, description: str | None) -> int:
    text = f"{_norm(title).lower()} {_norm(description).lower()}"
    if not text.strip():
        return 0
    if any(re.search(pat, text) for pat in QUANT_BLOCKLIST_PATTERNS):
        return 0

    score = 0
    for pat in QUANT_STRONG_PATTERNS:
        if re.search(pat, text):
            score += 2
    for pat in QUANT_WEAK_PATTERNS:
        if re.search(pat, text):
            score += 1
    return score


def is_quantitative_module(title: str | None, description: str | None, min_score: int = 3) -> bool:
    return _quant_signal_score(title, description) >= min_score


class StemScopeClassifier:
    """Shared classifier for NUS/NTU/SUTD scope bucketing.

    Buckets:
    - clear_stem
    - clear_non_stem
    - unclear_or_mixed
    """

    def __init__(self):
        self.nus = _load_json(NUS_FILE)
        self.ntu = _load_json(NTU_FILE)
        self.sutd = _load_json(SUTD_FILE)

        self.nus_fac = {
            "clear_stem": _to_set(self.nus.get("faculties", {}).get("clear_stem")),
            "clear_non_stem": _to_set(self.nus.get("faculties", {}).get("clear_non_stem")),
            "unclear_or_mixed": _to_set(self.nus.get("faculties", {}).get("unclear_or_mixed")),
        }
        self.nus_dept = {
            "clear_stem": _to_set(self.nus.get("departments", {}).get("clear_stem")),
            "clear_non_stem": _to_set(self.nus.get("departments", {}).get("clear_non_stem")),
            "unclear_or_mixed": _to_set(self.nus.get("departments", {}).get("unclear_or_mixed")),
        }

        self.ntu_dept = {
            "clear_stem": _to_set(self.ntu.get("departments", {}).get("clear_stem")),
            "clear_non_stem": _to_set(self.ntu.get("departments", {}).get("clear_non_stem")),
            "unclear_or_mixed": _to_set(self.ntu.get("departments", {}).get("unclear_or_mixed")),
        }

        self.sutd_dept = {
            "clear_stem": _to_set(self.sutd.get("departments", {}).get("clear_stem")),
            "clear_non_stem": _to_set(self.sutd.get("departments", {}).get("clear_non_stem")),
            "unclear_or_mixed": _to_set(self.sutd.get("departments", {}).get("unclear_or_mixed")),
        }

    def classify_module_scope(
        self,
        source: str | None,
        department: str | None = None,
        faculty: str | None = None,
        title: str | None = None,
        description: str | None = None,
        quant_min_score: int = 3,
    ) -> dict[str, str]:
        source_n = _norm(source).upper()
        dept_n = _norm(department)
        fac_n = _norm(faculty)

        excluded_reason = _exclusion_reason(
            title=title,
            description=description,
            department=department,
            faculty=faculty,
        )
        if excluded_reason:
            return {"scope_bucket": "clear_non_stem", "scope_reason": excluded_reason}

        base_bucket = "unclear_or_mixed"
        base_reason = "unknown_source"

        if source_n == "NUS":
            # Priority: explicit STEM match > explicit non-STEM match > unclear.
            if dept_n in self.nus_dept["clear_stem"] or fac_n in self.nus_fac["clear_stem"]:
                base_bucket, base_reason = "clear_stem", "nus_department_or_faculty_clear_stem"
            elif dept_n in self.nus_dept["clear_non_stem"] or fac_n in self.nus_fac["clear_non_stem"]:
                base_bucket, base_reason = "clear_non_stem", "nus_department_or_faculty_clear_non_stem"
            else:
                base_bucket, base_reason = "unclear_or_mixed", "nus_unlisted_or_mixed"

        elif source_n == "NTU":
            if dept_n in self.ntu_dept["clear_stem"]:
                base_bucket, base_reason = "clear_stem", "ntu_department_clear_stem"
            elif dept_n in self.ntu_dept["clear_non_stem"]:
                base_bucket, base_reason = "clear_non_stem", "ntu_department_clear_non_stem"
            else:
                base_bucket, base_reason = "unclear_or_mixed", "ntu_unlisted_or_mixed"

        elif source_n == "SUTD":
            if not dept_n:
                base_bucket, base_reason = "unclear_or_mixed", "sutd_department_missing"
            elif dept_n in self.sutd_dept["clear_stem"]:
                base_bucket, base_reason = "clear_stem", "sutd_department_clear_stem"
            elif dept_n in self.sutd_dept["clear_non_stem"]:
                base_bucket, base_reason = "clear_non_stem", "sutd_department_clear_non_stem"
            else:
                base_bucket, base_reason = "unclear_or_mixed", "sutd_unlisted_or_mixed"

        if base_bucket != "clear_stem" and is_quantitative_module(title, description, min_score=quant_min_score):
            if base_bucket == "clear_non_stem":
                return {"scope_bucket": "quant_non_stem", "scope_reason": "quant_semantic_override_non_stem"}
            return {"scope_bucket": "quant_non_stem", "scope_reason": "quant_semantic_override_unclear"}

        return {"scope_bucket": base_bucket, "scope_reason": base_reason}


def load_stem_scope_classifier() -> StemScopeClassifier:
    return StemScopeClassifier()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported JSON shape in {path}. Expected a top-level list or JSONL rows.")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_module_meta_lookup(processed_dir: Path):
    lookup = {}

    nus_path = processed_dir / "nus_cleaned.json"
    if nus_path.exists():
        for record in _load_rows(nus_path):
            code = _norm(record.get("moduleCode"))
            if not code:
                continue
            lookup[f"NUS::{code}"] = {
                "department": record.get("department"),
                "faculty": record.get("faculty"),
            }

    ntu_path = processed_dir / "ntu_cleaned.json"
    if ntu_path.exists():
        for record in _load_rows(ntu_path):
            code = _norm(record.get("code"))
            if not code:
                continue
            lookup[f"NTU::{code}"] = {
                "department": record.get("department"),
                "faculty": record.get("faculty"),
            }

    sutd_path = processed_dir / "sutd_cleaned.json"
    if sutd_path.exists():
        for record in _load_rows(sutd_path):
            code = _norm(record.get("code"))
            if not code:
                continue
            lookup[f"SUTD::{code}"] = {
                "department": record.get("department"),
                "faculty": record.get("faculty"),
            }

    return lookup


def build_stem_rows(rows: list[dict[str, Any]], processed_dir: Path, quant_min_score: int = 3):
    clf = load_stem_scope_classifier()
    module_meta = build_module_meta_lookup(processed_dir)

    stem_rows: list[dict[str, Any]] = []
    for row in rows:
        meta = module_meta.get(_norm(row.get("id")), {})
        out = clf.classify_module_scope(
            source=row.get("source"),
            department=meta.get("department"),
            faculty=meta.get("faculty"),
            title=row.get("title"),
            description=row.get("description"),
            quant_min_score=quant_min_score,
        )
        if out["scope_bucket"] in {"clear_stem", "quant_non_stem"}:
            stem_rows.append(row)
    return stem_rows


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Classify module rows into STEM scope buckets using shared classification JSON files.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input module file (.json list or .jsonl rows).")
    parser.add_argument("--source", type=str, default=None, choices=["NUS", "NTU", "SUTD"])
    parser.add_argument("--department-key", type=str, default="department")
    parser.add_argument("--faculty-key", type=str, default="faculty")
    parser.add_argument("--summary", action="store_true", help="Print bucket/reason counts.")
    parser.add_argument("--show-samples", type=int, default=0, help="Show N sample classified rows.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for classified rows (.jsonl).")
    parser.add_argument("--quant-min-score", type=int, default=3, help="Minimum semantic quantitative score for non-STEM override.")
    parser.add_argument(
        "--build-stem-rows",
        action="store_true",
        help="Build STEM-only rows from cleaned_module_rows.jsonl and save cleaned_module_rows_STEM.jsonl.",
    )
    parser.add_argument("--cleaned-input", type=Path, default=DEFAULT_CLEANED_MODULE_INPUT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--stem-output", type=Path, default=DEFAULT_STEM_OUTPUT)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_build_mode = args.build_stem_rows or (args.input is None and args.source is None)
    if run_build_mode:
        if not args.cleaned_input.exists():
            raise FileNotFoundError(f"Cleaned module input file not found: {args.cleaned_input}")
        rows = _load_rows(args.cleaned_input)
        stem_rows = build_stem_rows(rows, processed_dir=args.processed_dir, quant_min_score=args.quant_min_score)
        _write_jsonl(args.stem_output, stem_rows)
        print(f"Input rows: {len(rows)}")
        print(f"Saved STEM cleaned module rows: {len(stem_rows)} -> {args.stem_output}")
        return

    if args.input is None or args.source is None:
        raise SystemExit("Provide both --input and --source, or use --build-stem-rows.")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = _load_rows(args.input)
    clf = load_stem_scope_classifier()

    classified = []
    bucket_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        out = clf.classify_module_scope(
            source=args.source,
            department=row.get(args.department_key),
            faculty=row.get(args.faculty_key),
            title=row.get("title"),
            description=row.get("description"),
            quant_min_score=args.quant_min_score,
        )
        merged = {
            **row,
            "scope_bucket": out["scope_bucket"],
            "scope_reason": out["scope_reason"],
        }
        classified.append(merged)
        bucket_counts[out["scope_bucket"]] = bucket_counts.get(out["scope_bucket"], 0) + 1
        reason_counts[out["scope_reason"]] = reason_counts.get(out["scope_reason"], 0) + 1

    if args.summary or not args.output:
        print(f"Input rows: {len(rows)}")
        print("Bucket counts:")
        for k in sorted(bucket_counts):
            print(f"  {k}: {bucket_counts[k]}")
        print("Reason counts:")
        for k in sorted(reason_counts):
            print(f"  {k}: {reason_counts[k]}")

    if args.show_samples > 0:
        print(f"Samples (first {args.show_samples}):")
        for row in classified[: args.show_samples]:
            row_id = row.get("id") or row.get("code") or row.get("moduleCode") or ""
            title = row.get("title") or ""
            print(f"  {row_id} | {title} | {row['scope_bucket']} | {row['scope_reason']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for row in classified:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved classified rows: {len(classified)} -> {args.output}")


if __name__ == "__main__":
    main()
