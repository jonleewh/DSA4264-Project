import argparse
import html
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOBS_DIR = PROJECT_ROOT / "data" / "data"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "data" / "processed" / "jobs_cleaned.json"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "data" / "processed" / "jobs_cleaned.jsonl"

ENTRY_LEVEL_TEXT_PATTERNS = [
    r"\bfresh grads?\b",
    r"\bfresh graduates?\b",
    r"\bentry\s*-?\s*level\b",
    r"\bno experience\b",
    r"\btraining (?:will be )?provided\b",
    r"\bwelcome fresh\b",
]

SOFT_SKILL_HINTS = [
    "communication",
    "interpersonal",
    "teamwork",
    "team player",
    "leadership",
    "problem solving",
    "adaptability",
    "time management",
    "stakeholder",
    "presentation",
    "critical thinking",
    "self-motivated",
    "work independently",
    "collaboration",
    "customer service",
    "attention to detail",
]

DROP_SKILL_PATTERNS = [
    r"^etc$",
    r"^others?$",
    r"^n/?a$",
    r"^skills?$",
    r"^knowledge$",
]

GENERIC_SOFT_SKILLS = {
    "communication",
    "communications",
    "interpersonal",
    "teamwork",
    "leadership",
    "problem solving",
    "analytical skills",
}


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    cleaned = html.unescape(cleaned)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_skill(skill: str | None) -> str:
    if not skill:
        return ""
    skill = clean_text(skill)
    skill = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9+#/]+$", "", skill)
    return skill.strip()


def should_drop_skill(skill: str) -> bool:
    s = skill.lower().strip()
    if not s:
        return True
    if len(s) < 2 or len(s) > 60:
        return True
    return any(re.search(pattern, s) for pattern in DROP_SKILL_PATTERNS)


def split_hard_soft_skills(skills: list[str]) -> tuple[list[str], list[str]]:
    hard, soft = [], []
    seen_hard, seen_soft = set(), set()

    for skill in skills:
        s = skill.lower()
        is_soft = any(hint in s for hint in SOFT_SKILL_HINTS) or s in GENERIC_SOFT_SKILLS
        if is_soft:
            if s not in seen_soft:
                soft.append(skill)
                seen_soft.add(s)
        else:
            if s not in seen_hard:
                hard.append(skill)
                seen_hard.add(s)
    return hard, soft


def extract_experience_mentions(text: str) -> list[str]:
    if not text:
        return []
    patterns = [
        r"(?:minimum|min\.?|at least|more than|over)\s+\d+\+?\s*(?:years?|yrs?)",
        r"\d+\+?\s*(?:years?|yrs?)\s+of\s+experience",
        r"\d+\s*[-/to]+\s*\d+\s*(?:years?|yrs?)\s+of\s+experience",
        r"entry-level candidates welcome",
        r"no experience (?:needed|required)",
    ]
    matches = []
    seen = set()
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = m.group(0).strip()
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            matches.append(phrase)
    return matches


def parse_min_exp(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def exp_bucket(min_exp: float | None) -> str:
    if min_exp is None:
        return "unknown"
    if min_exp <= 1:
        return "0-1"
    if min_exp <= 2:
        return "1-2"
    return "2+"


def build_row(record: dict, path: Path, fresh_exp_threshold: float) -> dict:
    title = clean_text(record.get("title"))
    description_clean = clean_text(record.get("description"))
    other_requirements_clean = clean_text(record.get("otherRequirements"))
    full_text = " ".join([title, description_clean, other_requirements_clean]).strip()
    full_text_lower = full_text.lower()

    position_levels = [
        (p.get("position") if isinstance(p, dict) else str(p)).strip()
        for p in (record.get("positionLevels") or [])
        if (p.get("position") if isinstance(p, dict) else str(p)).strip()
    ]

    min_exp = parse_min_exp(record.get("minimumYearsExperience"))
    signal_by_position = any("fresh/entry level" in p.lower() for p in position_levels)
    signal_by_exp = min_exp is not None and min_exp <= fresh_exp_threshold
    signal_by_text = any(
        re.search(pattern, full_text_lower, flags=re.IGNORECASE)
        for pattern in ENTRY_LEVEL_TEXT_PATTERNS
    )

    is_freshgrad = signal_by_position or signal_by_exp or signal_by_text
    entry_conflict = is_freshgrad and min_exp is not None and min_exp >= 3

    normalized_skills = []
    for item in record.get("skills") or []:
        skill_name = item.get("skill") if isinstance(item, dict) else str(item)
        skill_name = normalize_skill(skill_name)
        if should_drop_skill(skill_name):
            continue
        normalized_skills.append(
            {
                "name": skill_name,
                "is_key_skill": bool(item.get("isKeySkill")) if isinstance(item, dict) else False,
            }
        )

    deduped = []
    seen = set()
    for s in normalized_skills:
        key = s["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        phrase_in_text = key in full_text_lower
        keep = phrase_in_text or s["is_key_skill"] or len(key.split()) >= 2
        if keep:
            deduped.append(s["name"])

    hard_skills, soft_skills = split_hard_soft_skills(deduped)
    metadata = record.get("metadata") or {}
    job_id = metadata.get("jobPostId") or record.get("uuid") or path.stem

    return {
        "id": job_id,
        "source": record.get("sourceCode"),
        "title": title,
        "is_freshgrad": is_freshgrad,
        "freshgrad_signals": {
            "position_level": signal_by_position,
            "minimum_years_experience": signal_by_exp,
            "description_text": signal_by_text,
        },
        "entry_conflict": entry_conflict,
        "minimum_years_experience": min_exp,
        "experience_bucket": exp_bucket(min_exp),
        "experience_mentions": extract_experience_mentions(full_text),
        "position_levels": position_levels,
        "hard_skills": hard_skills,
        "soft_skills": soft_skills,
        "all_relevant_skills": deduped,
        "ssoc_code": record.get("ssocCode"),
        "occupation_id": record.get("occupationId"),
        "description_clean": description_clean,
        "other_requirements_clean": other_requirements_clean,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean MCF job JSONs and extract relevant hard/soft skills and experience, "
            "focused on fresh-grad job postings."
        )
    )
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--fresh-exp-threshold", type=float, default=1.0)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    if not args.jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {args.jobs_dir}")

    paths = sorted(args.jobs_dir.glob("*.json"))
    if args.max_files is not None:
        paths = paths[: args.max_files]

    cleaned_rows = []
    fresh_count = 0
    conflict_count = 0
    for i, path in enumerate(paths, start=1):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = build_row(record, path, args.fresh_exp_threshold)
        if not row["description_clean"]:
            continue
        if row["is_freshgrad"]:
            fresh_count += 1
            if row["entry_conflict"]:
                conflict_count += 1
        cleaned_rows.append(row)

        if i % 2000 == 0 or i == len(paths):
            print(f"Processed {i}/{len(paths)} files...", flush=True)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(cleaned_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Total cleaned rows: {len(cleaned_rows)}")
    print(f"Fresh-grad candidates: {fresh_count}")
    print(f"Fresh-grad conflicts (exp >= 3): {conflict_count}")
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
