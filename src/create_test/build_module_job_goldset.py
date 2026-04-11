import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = PROJECT_ROOT / "data" / "test"

DEFAULT_BENCHMARK = TEST_DIR / "module_skill_benchmark.jsonl"
DEFAULT_ANDRE_ALIGNMENT = TEST_DIR / "module_job_alignment_andre.json"
DEFAULT_INDEP_ALIGNMENT = TEST_DIR / "module_job_alignment_independent.json"
DEFAULT_JSONL_OUT = TEST_DIR / "module_job_goldset.jsonl"
DEFAULT_CSV_OUT = TEST_DIR / "module_job_goldset.csv"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def top_match_summary(row: dict, top_n: int = 3) -> str:
    matches = row.get("top_matches") or []
    parts = []
    for match in matches[:top_n]:
        parts.append(
            f"{match.get('ssoc_code')} {match.get('ssoc_title')} ({match.get('alignment_score')})"
        )
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Build a gold-set template for module-to-job evaluation."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--andre-alignment", type=Path, default=DEFAULT_ANDRE_ALIGNMENT)
    parser.add_argument("--independent-alignment", type=Path, default=DEFAULT_INDEP_ALIGNMENT)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL_OUT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    args = parser.parse_args()

    benchmark_rows = load_jsonl(args.benchmark)
    andre_alignment = {
        row["module_id"]: row
        for row in json.loads(args.andre_alignment.read_text(encoding="utf-8"))
    }
    indep_alignment = {
        row["module_id"]: row
        for row in json.loads(args.independent_alignment.read_text(encoding="utf-8"))
    }

    out_rows = []
    for row in benchmark_rows:
        module_id = row["id"]
        andre_row = andre_alignment.get(module_id, {})
        indep_row = indep_alignment.get(module_id, {})

        out_rows.append(
            {
                "id": module_id,
                "source": row.get("source"),
                "title": row.get("title"),
                "domain_bucket": row.get("domain_bucket"),
                "description_snippet": row.get("description_snippet"),
                "keybert_skills": row.get("keybert_skills", []),
                "andre_skills": row.get("andre_skills", []),
                "independent_skills": row.get("independent_skills", []),
                "andre_top_matches": top_match_summary(andre_row),
                "independent_top_matches": top_match_summary(indep_row),
                "expected_skills": "",
                "expected_job_families": "",
                "best_extraction_method": "",
                "best_alignment_method": "",
                "is_module_job_match_plausible": "",
                "review_notes": "",
            }
        )

    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl_out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with args.csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "source",
                "title",
                "domain_bucket",
                "description_snippet",
                "keybert_skills",
                "andre_skills",
                "independent_skills",
                "andre_top_matches",
                "independent_top_matches",
                "expected_skills",
                "expected_job_families",
                "best_extraction_method",
                "best_alignment_method",
                "is_module_job_match_plausible",
                "review_notes",
            ],
        )
        writer.writeheader()
        for row in out_rows:
            writer.writerow(
                {
                    **row,
                    "keybert_skills": " | ".join(row["keybert_skills"]),
                    "andre_skills": " | ".join(row["andre_skills"]),
                    "independent_skills": " | ".join(row["independent_skills"]),
                }
            )

    print(f"Saved gold-set JSONL: {len(out_rows)} -> {args.jsonl_out}")
    print(f"Saved gold-set CSV: {len(out_rows)} -> {args.csv_out}")


if __name__ == "__main__":
    main()
