import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned_module_rows.jsonl"
MIN_WORDS = 10
NUS_POSTGRAD_FACULTIES = {
    "Temasek Defence Sys. Institute",
    "Cont and Lifelong Education",
    "Duke-NUS Medical School",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def has_min_words(text: str | None, min_words: int = MIN_WORDS) -> bool:
    return len((text or "").split()) >= min_words


def is_undergrad_code(code: str | None) -> bool:
    if not code:
        return False
    code = code.strip().upper()
    prefix = []
    digits = []
    for ch in code:
        if ch.isalpha() and not digits:
            prefix.append(ch)
        elif ch.isdigit():
            digits.append(ch)
            break
        elif prefix:
            break
    return bool(prefix and digits and digits[0] in "01234")


def build_nus_rows():
    rows = []
    path = PROJECT_ROOT / "data" / "NUSModsInfo.json"
    if not path.exists():
        return rows

    for record in load_json(path):
        module_code = record.get("moduleCode")
        faculty = record.get("faculty")
        description = (record.get("description") or "").strip()
        if not is_undergrad_code(module_code):
            continue
        if faculty in NUS_POSTGRAD_FACULTIES:
            continue
        if not has_min_words(description):
            continue
        rows.append(
            {
                "id": f"NUS::{module_code}",
                "source": "NUS",
                "title": record.get("title"),
                "description": description,
            }
        )
    return rows


def build_ntu_rows():
    rows = []
    path = PROJECT_ROOT / "data" / "ntuCourseInfo.json"
    if not path.exists():
        return rows

    for record in load_json(path):
        code = record.get("code")
        description = (record.get("description") or "").strip()
        if not is_undergrad_code(code):
            continue
        if not has_min_words(description):
            continue
        rows.append(
            {
                "id": f"NTU::{code}",
                "source": "NTU",
                "title": record.get("title"),
                "description": description,
            }
        )
    return rows


def build_sutd_rows():
    rows = []
    path = PROJECT_ROOT / "data" / "sutdCourseDescriptions.json"
    if not path.exists():
        return rows

    for record in load_json(path):
        code = record.get("code")
        description = (record.get("description") or "").strip()
        if not has_min_words(description):
            continue
        rows.append(
            {
                "id": f"SUTD::{code}",
                "source": "SUTD",
                "title": record.get("title"),
                "description": description,
            }
        )
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rows = build_nus_rows() + build_ntu_rows() + build_sutd_rows()
    write_jsonl(OUTPUT_PATH, rows)
    print(f"Saved cleaned module rows: {len(rows)} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
