from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_COURSES_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "combined_courses_cleaned.pkl"
DEFAULT_JOBS_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "jobs_cleaned.pkl"


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_min_description(text: str | None, min_length: int) -> bool:
    return len(normalize_text(text)) >= min_length


def write_json(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_module_rows(courses_input: Path, min_description_length: int) -> list[dict]:
    if not courses_input.exists():
        raise FileNotFoundError(f"Cleaned courses input not found: {courses_input}")

    df = pd.read_pickle(courses_input)
    required_cols = {"code", "title", "description", "department", "university"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required course columns in {courses_input}: {missing}"
        )

    rows = []
    for record in df.to_dict("records"):
        code = normalize_text(record.get("code"))
        university = normalize_text(record.get("university"))
        description = normalize_text(record.get("description"))
        if not code or not university or not has_min_description(description, min_description_length):
            continue

        skills = record.get("skills_embedding")
        if not isinstance(skills, list):
            skills = []
        skills = [
            normalize_text(skill).lower()
            for skill in skills
            if normalize_text(skill)
        ]

        hard_skills = record.get("hard_skills")
        if not isinstance(hard_skills, list):
            hard_skills = []
        hard_skills = [
            normalize_text(skill).lower()
            for skill in hard_skills
            if normalize_text(skill)
        ]

        soft_skills = record.get("soft_skills")
        if not isinstance(soft_skills, list):
            soft_skills = []
        soft_skills = [
            normalize_text(skill).lower()
            for skill in soft_skills
            if normalize_text(skill)
        ]

        rows.append(
            {
                "id": f"{university}::{code}",
                "source": university,
                "code": code,
                "title": normalize_text(record.get("title")),
                "department": normalize_text(record.get("department")),
                "university": university,
                "description": description,
                "skills": skills,
                "hard_skills": hard_skills,
                "soft_skills": soft_skills,
                "num_skills": record.get("num_skills"),
            }
        )

    return rows


def build_job_rows(jobs_input: Path, min_description_length: int) -> list[dict]:
    if not jobs_input.exists():
        raise FileNotFoundError(f"Cleaned jobs input not found: {jobs_input}")

    df = pd.read_pickle(jobs_input)
    required_cols = {"uuid", "title", "description"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required job columns in {jobs_input}: {missing}"
        )

    rows = []
    for record in df.to_dict("records"):
        job_id = normalize_text(record.get("uuid"))
        description = normalize_text(record.get("description"))
        if not job_id or not has_min_description(description, min_description_length):
            continue

        skills_clean = record.get("skills_clean")
        if not isinstance(skills_clean, list):
            skills_clean = []

        skills_clean = [
            normalize_text(skill).lower()
            for skill in skills_clean
            if normalize_text(skill)
        ]

        rows.append(
            {
                "id": job_id,
                "source": normalize_text(record.get("source")) or "MCF",
                "title": normalize_text(record.get("title")),
                "description": description,
                "ssoc_code": normalize_text(record.get("ssoc_code")),
                "ssoc_3d": normalize_text(record.get("ssoc_3d")),
                "contract_type": normalize_text(record.get("contract_type")),
                "work_type": normalize_text(record.get("work_type")),
                "avg_salary": record.get("avg_salary"),
                "skills": skills_clean,
                "all_relevant_skills": skills_clean,
                "is_freshgrad": True,
                "is_good_job": int(record.get("is_good_job", 0) or 0),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create downstream test datasets from notebook-cleaned course and job PKL files."
        )
    )
    parser.add_argument("--courses-input", type=Path, default=DEFAULT_COURSES_INPUT)
    parser.add_argument("--jobs-input", type=Path, default=DEFAULT_JOBS_INPUT)
    parser.add_argument(
        "--module-size",
        type=int,
        default=500,
        help="Sample size for module rows in default development mode.",
    )
    parser.add_argument(
        "--job-size",
        type=int,
        default=1000,
        help="Sample size for job rows in default development mode.",
    )
    parser.add_argument("--min-description-length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Export all eligible rows instead of development samples.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    module_rows = build_module_rows(args.courses_input, args.min_description_length)
    job_rows = build_job_rows(args.jobs_input, args.min_description_length)

    random.shuffle(module_rows)
    random.shuffle(job_rows)

    if args.full_dataset:
        module_test = module_rows
        job_test = job_rows
    else:
        module_test = module_rows[: args.module_size]
        job_test = job_rows[: args.job_size]

    module_json = args.output_dir / "module_descriptions_test.json"
    module_jsonl = args.output_dir / "module_descriptions_test.jsonl"
    job_json = args.output_dir / "job_descriptions_test.json"
    job_jsonl = args.output_dir / "job_descriptions_test.jsonl"

    write_json(module_json, module_test)
    write_jsonl(module_jsonl, module_test)
    write_json(job_json, job_test)
    write_jsonl(job_jsonl, job_test)

    print(f"Loaded cleaned courses: {len(module_rows)} from {args.courses_input}")
    print(f"Loaded cleaned jobs: {len(job_rows)} from {args.jobs_input}")
    print(
        "Export mode: full dataset"
        if args.full_dataset
        else f"Export mode: sampled ({len(module_test)} modules, {len(job_test)} jobs)"
    )
    print(f"Saved module dataset: {len(module_test)} -> {module_json}")
    print(f"Saved module dataset JSONL: {len(module_test)} -> {module_jsonl}")
    print(f"Saved job dataset: {len(job_test)} -> {job_json}")
    print(f"Saved job dataset JSONL: {len(job_test)} -> {job_jsonl}")


if __name__ == "__main__":
    main()
