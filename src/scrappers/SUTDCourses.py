import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.sutd.edu.sg/education/undergraduate/courses/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "sutdCourseList.json"

CODE_RE = re.compile(r"^(?P<code>\d{2}\.\d{3}[A-Z]{0,2})\s+(?P<title>.+)$")
TERM_RE = re.compile(r"Term\(s\)\s*(.+)")
CREDITS_RE = re.compile(r"(\d+)\s*credits", re.IGNORECASE)

COURSE_TYPES = {
    "Capstone",
    "Core",
    "Core elective",
    "Elective / Technical Elective",
    "Freshmore core",
    "Freshmore elective",
    "Track core",
}

def fetch_page(page: int) -> str:
    response = requests.get(BASE_URL, params={"paged": page}, timeout=30)
    response.raise_for_status()
    return response.text

def parse_courses(html: str):
    soup = BeautifulSoup(html, "html.parser")
    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]

    results = []
    i = 0
    while i < len(lines):
        match = CODE_RE.match(lines[i])
        if not match:
            i += 1
            continue

        code = match.group("code")
        title = match.group("title")
        course_type = lines[i - 1] if i > 0 and lines[i - 1] in COURSE_TYPES else None

        term = None
        credits = None
        j = i + 1
        while j < len(lines):
            if CODE_RE.match(lines[j]):
                break

            term_match = TERM_RE.search(lines[j])
            if term_match:
                term = term_match.group(1).strip()

            credits_match = CREDITS_RE.search(lines[j])
            if credits_match:
                credits = int(credits_match.group(1))

            j += 1

        results.append(
            {
                "code": code,
                "title": title,
                "course_type": course_type,
                "terms": term,
                "credits": credits,
            }
        )
        i = j

    return results

def fetch_all_courses(max_pages: int = 100):
    all_courses = []
    seen_codes = set()

    for page in range(1, max_pages + 1):
        html = fetch_page(page)
        courses = parse_courses(html)
        new_courses = [course for course in courses if course["code"] not in seen_codes]

        if not new_courses:
            break

        for course in new_courses:
            seen_codes.add(course["code"])
            all_courses.append(course)

    return all_courses

if __name__ == "__main__":
    courses = fetch_all_courses()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(courses, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(courses)} courses to: {OUT_PATH}")
