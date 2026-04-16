import argparse
import html
import json
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CLEANED_MODULE_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "cleaned_module_rows_STEM.jsonl"
DEFAULT_JOBS_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "jobs_cleaned.pkl"


def clean_html(text: Optional[str]) -> str:
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
    if not DEFAULT_CLEANED_MODULE_INPUT.exists():
        raise FileNotFoundError(
            "STEM module rows not found. Run "
            "src/stem_test/stem_1_scope_classifier.py first to create "
            f"{DEFAULT_CLEANED_MODULE_INPUT}."
        )

    rows = []
    for record in load_jsonl(DEFAULT_CLEANED_MODULE_INPUT):
        desc = (record.get("description") or "").strip()
        if len(desc) < min_description_length:
            continue
        rows.append(record)
    print(f"Loaded STEM module rows from: {DEFAULT_CLEANED_MODULE_INPUT}")
    return rows


def build_job_rows(min_description_length: int):
    rows = []
    if DEFAULT_JOBS_INPUT.exists():
        df = pd.read_pickle(DEFAULT_JOBS_INPUT)
        for record in df.to_dict("records"):
            desc = clean_html(record.get("description"))
            if len(desc) < min_description_length:
                continue

            job_id = record.get("uuid")
            if not job_id:
                continue

            rows.append(
                {
                    "id": job_id,
                    "source": record.get("source") or "MCF",
                    "title": record.get("title"),
                    "description": desc,
                    "is_good_job": int(record.get("is_good_job", 0) or 0),
                }
            )
        return rows

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
                "is_good_job": int(record.get("is_good_job", 0) or 0),
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
