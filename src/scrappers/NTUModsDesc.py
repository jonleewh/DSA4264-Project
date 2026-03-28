import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "ntuCourseList.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ntuCourseInfo.json"
DETAIL_URL_TEMPLATE = "https://www.ntumods.org/mods/{code}"
MAX_WORKERS = 12


def load_course_codes(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row["code"] for row in rows if row.get("code")]


def parse_course_page(html: str):
    soup = BeautifulSoup(html, "html.parser")

    title = None
    description = None

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        text = script.string
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "Course":
            title = data.get("name")
            description = data.get("description")
            break

    details = {}
    for row in soup.select("table tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        key = " ".join(cells[0].stripped_strings).strip()
        value = " ".join(cells[1].stripped_strings).strip()
        if not key or key.isdigit():
            continue
        if key in {"Index", "Type", "Group", "Day", "Time", "Venue", "Remark"}:
            continue
        if key:
            details[key] = value

    return title, description, details


def fetch_course_info(code: str):
    url = DETAIL_URL_TEMPLATE.format(code=code)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    title, description, details = parse_course_page(response.text)

    return {
        "code": code,
        "title": title,
        "description": description,
        "academicUnits": details.get("Academic Units"),
        "examSchedule": details.get("Exam Schedule"),
        "gradeType": details.get("Grade Type"),
        "departmentMaintaining": details.get("Department Maintaining"),
        "prerequisites": details.get("Prerequisites"),
        "mutuallyExclusiveWith": details.get("Mutually Exclusive with"),
        "notAvailableToProgramme": details.get("Not Available to Programme"),
        "url": url,
        "details": details,
    }



def main(limit: Optional[int]):
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}. Run src/NTUMods.py first.")

    codes = load_course_codes(INPUT_PATH)
    if limit is not None:
        codes = codes[:limit]

    print(f"Fetching course details for {len(codes)} modules...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_course_info, code): code for code in codes}
        for idx, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "code": code,
                        "url": DETAIL_URL_TEMPLATE.format(code=code),
                        "error": str(exc),
                    }
                )

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(codes)}")

    results.sort(key=lambda x: x.get("code") or "")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(results)} records to: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Fetch only first N modules")
    args = parser.parse_args()
    main(args.limit)
