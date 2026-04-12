import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURSES_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "combined_courses_cleaned.pkl"
JOBS_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "jobs_cleaned.pkl"
MODULE_OUTPUT = PROJECT_ROOT / "data" / "cleaned_module_rows.jsonl"
JOBS_OUTPUT = PROJECT_ROOT / "data" / "cleaned_data" / "jobs_cleaned_portable.jsonl"
MIN_WORDS = 10


def has_min_words(text):
    return len((text or "").split()) >= MIN_WORDS


def build_module_rows():
    if not COURSES_INPUT.exists():
        raise FileNotFoundError(f"Missing cleaned courses input: {COURSES_INPUT}")

    df = pd.read_pickle(COURSES_INPUT)
    rows = []
    seen = set()
    for row in df.to_dict(orient="records"):
        code = str(row.get("code") or "").strip()
        source = str(row.get("university") or "").strip()
        title = str(row.get("title") or "").strip()
        description = str(row.get("description") or "").strip()
        department = str(row.get("department") or "").strip()
        if not code or not source or not title or not has_min_words(description):
            continue
        key = (source, code, title, description)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": f"{source}::{code}",
                "source": source,
                "title": title,
                "description": description,
                "department": department,
                "faculty": "",
            }
        )
    return rows


def build_job_rows():
    if not JOBS_INPUT.exists():
        raise FileNotFoundError(f"Missing cleaned jobs input: {JOBS_INPUT}")

    df = pd.read_pickle(JOBS_INPUT)
    rows = []
    for row in df.to_dict(orient="records"):
        uuid = str(row.get("uuid") or "").strip()
        ssoc_code = str(row.get("ssoc_code") or "").strip()
        if not uuid or not ssoc_code:
            continue
        rows.append(
            {
                "id": uuid,
                "uuid": uuid,
                "title": str(row.get("title") or "").strip(),
                "description": str(row.get("description") or "").strip(),
                "ssoc_code": ssoc_code,
                "skills": row.get("skills_clean") or row.get("skills") or [],
                "is_freshgrad": True,
            }
        )
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    module_rows = build_module_rows()
    job_rows = build_job_rows()
    write_jsonl(MODULE_OUTPUT, module_rows)
    write_jsonl(JOBS_OUTPUT, job_rows)
    print(f"Saved cleaned module rows: {len(module_rows)} -> {MODULE_OUTPUT}")
    print(f"Saved portable cleaned jobs: {len(job_rows)} -> {JOBS_OUTPUT}")


if __name__ == "__main__":
    main()
