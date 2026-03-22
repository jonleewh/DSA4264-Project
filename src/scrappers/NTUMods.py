import json
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "ntuCourseList.json"
BASE_URL = "https://backend.ntumods.org/courses/"


def fetch_all_courses():
    all_courses = []
    seen_codes = set()
    page = 1
    total_pages = None

    while True:
        response = requests.get(BASE_URL, params={"page": page}, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if total_pages is None:
            total_pages = payload.get("total_pages")
            print(f"Total pages reported: {total_pages}")

        results = payload.get("results", [])
        if not results:
            break

        for course in results:
            code = course.get("code")
            if code and code not in seen_codes:
                seen_codes.add(code)
                all_courses.append(course)

        print(f"Fetched page {page} ({len(results)} items)")
        if not payload.get("next"):
            break
        page += 1

    all_courses.sort(key=lambda x: x.get("code") or "")
    return all_courses


def main():
    courses = fetch_all_courses()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(courses, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(courses)} NTU courses to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
