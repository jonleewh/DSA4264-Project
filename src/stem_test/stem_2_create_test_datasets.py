import argparse
import html
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CLEANED_MODULE_INPUT = PROJECT_ROOT / "data" / "cleaned_module_rows_STEM.jsonl"


def clean_html(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_module_rows(min_description_length: int):
    if DEFAULT_CLEANED_MODULE_INPUT.exists():
        rows = []
        for record in load_jsonl(DEFAULT_CLEANED_MODULE_INPUT):
            desc = (record.get("description") or "").strip()
            if len(desc) < min_description_length:
                continue
            rows.append(record)
        print(f"Loaded cleaned module rows from: {DEFAULT_CLEANED_MODULE_INPUT}")
        return rows

    rows = []

    cleaned_sources = [
        ("NUS", DEFAULT_PROCESSED_DIR / "nus_cleaned.json", "moduleCode"),
        ("NTU", DEFAULT_PROCESSED_DIR / "ntu_cleaned.json", "code"),
        ("SUTD", DEFAULT_PROCESSED_DIR / "sutd_cleaned.json", "code"),
    ]

    for source_name, path, code_key in cleaned_sources:
        if not path.exists():
            continue

        for record in load_json(path):
            desc = (record.get("description") or "").strip()
            if len(desc) < min_description_length:
                continue

            code = record.get(code_key)
            if code is None:
                continue

            rows.append(
                {
                    "id": f"{source_name}::{code}",
                    "source": source_name,
                    "title": record.get("title"),
                    "description": desc,
                }
            )

    return rows


def build_job_rows(min_description_length: int):
    rows = []
    jobs_dir = PROJECT_ROOT / "data" / "data"
    if not jobs_dir.exists():
        return rows

    for path in jobs_dir.glob("*.json"):
        record = load_json(path)
        desc = clean_html(record.get("description"))
        if len(desc) < min_description_length:
            continue

        metadata = record.get("metadata") or {}
        job_id = metadata.get("jobPostId") or record.get("uuid") or path.stem

        rows.append(
            {
                "id": job_id,
                "source": record.get("sourceCode"),
                "title": record.get("title"),
                "description": desc,
            }
        )

    return rows


def write_json(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create test datasets for module and job descriptions."
    )
    parser.add_argument("--module-size", type=int, default=500)
    parser.add_argument("--job-size", type=int, default=1000)
    parser.add_argument("--min-description-length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    random.seed(args.seed)

    module_rows = build_module_rows(args.min_description_length)
    job_rows = build_job_rows(args.min_description_length)

    random.shuffle(module_rows)
    random.shuffle(job_rows)

    module_test = module_rows[: args.module_size]
    job_test = job_rows[: args.job_size]

    module_out = args.output_dir / "module_descriptions_test_STEM.json"
    job_out = args.output_dir / "job_descriptions_test_STEM.json"

    write_json(module_out, module_test)
    write_json(job_out, job_test)

    print(f"Saved module test set: {len(module_test)} -> {module_out}")
    print(f"Saved job test set: {len(job_test)} -> {job_out}")


if __name__ == "__main__":
    main()
