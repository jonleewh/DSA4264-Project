import argparse
import csv
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_BASE = TEST_DIR / "module_descriptions_test.jsonl"
DEFAULT_KEYBERT = TEST_DIR / "module_descriptions_test_with_skills_keybert.jsonl"
DEFAULT_INDEPENDENT = TEST_DIR / "module_descriptions_test_with_skills_independent.jsonl"
DEFAULT_ANDRE = TEST_DIR / "module_descriptions_test_with_skills_andre_compare.jsonl"
DEFAULT_JSONL_OUT = TEST_DIR / "module_skill_benchmark.jsonl"
DEFAULT_CSV_OUT = TEST_DIR / "module_skill_benchmark.csv"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def shorten(text: str | None, limit: int = 420) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_domain_bucket(row: dict) -> str:
    text = " ".join(
        str(row.get(key, "")) for key in ["title", "description"]
    ).lower()
    rules = [
        ("law_policy", ["law", "arbitration", "regulation", "tax", "policy", "governance"]),
        ("computing_data", ["python", "machine learning", "algorithm", "software", "database", "analytics", "data"]),
        ("business_finance", ["finance", "accounting", "banking", "marketing", "investment", "business"]),
        ("engineering_design", ["engineering", "design", "prototype", "architecture", "systems", "materials"]),
        ("health_biomed", ["health", "medical", "clinical", "anatomy", "biology", "disease", "pharma", "nursing"]),
        ("humanities_social", ["literature", "history", "culture", "philosophy", "language", "society", "politics", "religion"]),
    ]
    for name, patterns in rules:
        if any(pattern in text for pattern in patterns):
            return name
    return "other"


def select_rows(base_rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in base_rows:
        buckets.setdefault(build_domain_bucket(row), []).append(row)

    rng = random.Random(seed)
    selected = []

    bucket_names = sorted(buckets)
    per_bucket = max(1, sample_size // max(1, len(bucket_names)))

    for name in bucket_names:
        rows = buckets[name][:]
        rng.shuffle(rows)
        selected.extend(rows[:per_bucket])

    if len(selected) < sample_size:
        selected_ids = {row["id"] for row in selected}
        remaining = [row for row in base_rows if row["id"] not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])

    selected = selected[:sample_size]
    selected.sort(key=lambda row: (row["source"], row["id"]))
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Build a side-by-side benchmark for module skill extraction review."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--keybert", type=Path, default=DEFAULT_KEYBERT)
    parser.add_argument("--independent", type=Path, default=DEFAULT_INDEPENDENT)
    parser.add_argument("--andre", type=Path, default=DEFAULT_ANDRE)
    parser.add_argument("--sample-size", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL_OUT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    args = parser.parse_args()

    base_rows = load_jsonl(args.base)
    keybert_rows = {row["id"]: row for row in load_jsonl(args.keybert)}
    independent_rows = {row["id"]: row for row in load_jsonl(args.independent)}
    andre_rows = {row["id"]: row for row in load_jsonl(args.andre)}

    selected = select_rows(base_rows, args.sample_size, args.seed)

    benchmark_rows = []
    for row in selected:
        module_id = row["id"]
        benchmark_rows.append(
            {
                "id": module_id,
                "source": row.get("source"),
                "title": row.get("title"),
                "description_snippet": shorten(row.get("description")),
                "domain_bucket": build_domain_bucket(row),
                "keybert_skills": keybert_rows.get(module_id, {}).get("skills", []),
                "andre_skills": andre_rows.get(module_id, {}).get("skills", []),
                "independent_skills": independent_rows.get(module_id, {}).get("skills", []),
                "review_notes": "",
                "preferred_method": "",
                "expected_skills": "",
            }
        )

    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl_out.open("w", encoding="utf-8") as f:
        for row in benchmark_rows:
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
                "preferred_method",
                "expected_skills",
                "review_notes",
            ],
        )
        writer.writeheader()
        for row in benchmark_rows:
            writer.writerow(
                {
                    **row,
                    "keybert_skills": " | ".join(row["keybert_skills"]),
                    "andre_skills": " | ".join(row["andre_skills"]),
                    "independent_skills": " | ".join(row["independent_skills"]),
                }
            )

    print(f"Saved benchmark JSONL: {len(benchmark_rows)} -> {args.jsonl_out}")
    print(f"Saved benchmark CSV: {len(benchmark_rows)} -> {args.csv_out}")


if __name__ == "__main__":
    main()
