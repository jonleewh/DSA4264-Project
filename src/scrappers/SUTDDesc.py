import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "sutdCourseList.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "sutdCourseDescriptions.json"

COURSE_LIST_BASE = "https://www.sutd.edu.sg/education/undergraduate/courses/"
COURSE_PAGE_BASE = "https://www.sutd.edu.sg/course/"
CODE_RE = re.compile(r"^(?P<code>\d{2}\.\d{3}[A-Z]{0,2})\s+")

def load_courses(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def slugify_title(title: str) -> str:
    text = title.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text


def default_course_url(code: str, title: str) -> str:
    code_slug = code.lower().replace(".", "-")
    return urljoin(COURSE_PAGE_BASE, f"{code_slug}-{slugify_title(title)}")


def crawl_course_urls(session: requests.Session, max_pages: int = 30):
    by_code = {}
    seen_urls = set()

    for page in range(1, max_pages + 1):
        response = session.get(COURSE_LIST_BASE, params={"paged": page}, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        page_urls = set()
        for a in soup.select("a[href*='/course/']"):
            href = (a.get("href") or "").strip()
            if not href:
                continue

            course_url = href if href.startswith("http") else urljoin(COURSE_LIST_BASE, href)
            page_urls.add(course_url)

            code_match = re.search(r"/course/(\d{2}-\d{3}[a-z]{0,2})-", course_url, re.IGNORECASE)
            if code_match:
                code_slug = code_match.group(1)
                code = f"{code_slug[:2]}.{code_slug[3:]}".upper()
                by_code[code] = course_url

        new_urls = page_urls - seen_urls
        if not new_urls and page > 1:
            break
        seen_urls.update(page_urls)

    return by_code


def extract_description(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    if not h1:
        return None

    parts = []
    for node in h1.find_all_next():
        if node.name and re.match(r"h[1-6]", node.name, re.IGNORECASE):
            heading_text = " ".join(node.stripped_strings)
            if "prerequisite" in heading_text.lower():
                break

        if node.name in {"p", "div"}:
            text = " ".join(node.stripped_strings).strip()
            if text:
                parts.append(text)

    cleaned = []
    seen = set()
    for text in parts:
        if text in seen:
            continue
        if text.lower().startswith("number of credits"):
            continue
        if text.lower() == "tags":
            continue
        seen.add(text)
        cleaned.append(text)

    return " ".join(cleaned).strip() or None


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    courses = load_courses(INPUT_PATH)
    output_rows = []

    with requests.Session() as session:
        discovered_urls = crawl_course_urls(session)
        print(f"Discovered {len(discovered_urls)} course URLs from listing pages.")

        for idx, course in enumerate(courses, start=1):
            code = course.get("code")
            title = course.get("title") or ""

            course_url = discovered_urls.get(code) or default_course_url(code, title)
            description = None
            error = None

            try:
                response = session.get(course_url, timeout=30)
                response.raise_for_status()
                description = extract_description(response.text)
            except Exception as exc:
                error = str(exc)

            row = {
                **course,
                "url": course_url,
                "description": description,
                "description_found": bool(description),
            }
            if error:
                row["error"] = error

            output_rows.append(row)

            if idx % 25 == 0:
                print(f"Processed {idx}/{len(courses)} courses...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    found_count = sum(1 for row in output_rows if row.get("description_found"))
    print(f"Saved {len(output_rows)} records to: {OUTPUT_PATH}")
    print(f"Descriptions found: {found_count}/{len(output_rows)}")


if __name__ == "__main__":
    main()
