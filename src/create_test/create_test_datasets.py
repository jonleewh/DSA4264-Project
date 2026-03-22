import argparse
import html
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "test"


def clean_html(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_module_rows(min_description_length: int):
    rows = []

    nus_path = PROJECT_ROOT / "data" / "NUSModsInfo.json"
    if nus_path.exists():
        for record in load_json(nus_path):
            desc = (record.get("description") or "").strip()
            if len(desc) < min_description_length:
                continue
            rows.append(
                {
                    "id": f"NUS::{record.get('moduleCode')}",
                    "source": "NUS",
                    "title": record.get("title"),
                    "description": desc,
                }
            )

    ntu_path = PROJECT_ROOT / "data" / "ntuCourseInfo.json"
    if ntu_path.exists():
        for record in load_json(ntu_path):
            desc = (record.get("description") or "").strip()
            if len(desc) < min_description_length:
                continue
            rows.append(
                {
                    "id": f"NTU::{record.get('code')}",
                    "source": "NTU",
                    "title": record.get("title"),
                    "description": desc,
                }
            )

    sutd_path = PROJECT_ROOT / "data" / "sutdCourseDescriptions.json"
    if sutd_path.exists():
        for record in load_json(sutd_path):
            desc = (record.get("description") or "").strip()
            if len(desc) < min_description_length:
                continue
            rows.append(
                {
                    "id": f"SUTD::{record.get('code')}",
                    "source": "SUTD",
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


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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

    module_out = args.output_dir / "module_descriptions_test.jsonl"
    job_out = args.output_dir / "job_descriptions_test.jsonl"

    write_jsonl(module_out, module_test)
    write_jsonl(job_out, job_test)

    print(f"Saved module test set: {len(module_test)} -> {module_out}")
    print(f"Saved job test set: {len(job_test)} -> {job_out}")


if __name__ == "__main__":
    main()
