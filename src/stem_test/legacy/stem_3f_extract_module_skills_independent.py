import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Optional

import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from module_skill_rules import (
    ACADEMIC_LABEL_BLOCKLIST,
    CANONICAL_MODULE_SKILLS,
    EDU_BLOCKLIST_SUBSTRINGS,
    EXPLICIT_PHRASE_BLOCKLIST,
    MODULE_SKILL_RULES,
    PRACTICAL_ANCHORS,
    STRICT_CANONICAL_EVIDENCE,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FULL_DIR = PROJECT_ROOT / "data" / "stem_full"
DEFAULT_MODULE_INPUT = PROJECT_ROOT / "data" / "cleaned_module_rows_STEM.jsonl"
DEFAULT_OUTPUT = FULL_DIR / "module_descriptions_STEM_with_skills_independent.jsonl"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_skill(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9+#/]+$", "", text)
    return text


def contains_phrase(haystack: str, needle: str) -> bool:
    pattern = r"\b" + re.escape(needle.lower()) + r"\b"
    return re.search(pattern, haystack.lower()) is not None


def is_skill_like(skill: str) -> bool:
    s = skill.lower().strip()
    if not s:
        return False
    if len(s) < 2 or len(s) > 80:
        return False
    if s in ACADEMIC_LABEL_BLOCKLIST:
        return False
    return not any(bad in s for bad in EDU_BLOCKLIST_SUBSTRINGS)


def load_module_rows(path: Path, max_rows: Optional[int]):
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_no}: {exc}"
                    ) from exc
                if max_rows is not None and len(rows) >= max_rows:
                    break
        return rows

    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file at {path}: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(
                f"Expected top-level JSON array in {path}, got {type(payload).__name__}."
            )
        if max_rows is not None:
            return payload[:max_rows]
        return payload

    raise ValueError(
        f"Unsupported module input format for {path}. Use .jsonl or .json."
    )


def extract_candidate_phrases(kw_model: KeyBERT, description: str):
    text = normalize_text(description)
    if not text:
        return []
    ranked = kw_model.extract_keywords(
        text[:2500],
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=30,
        use_mmr=True,
        diversity=0.8,
    )
    phrases = [normalize_skill(phrase) for phrase, _ in ranked if normalize_skill(phrase)]
    return list(dict.fromkeys(phrases))


def find_rule_based_skills(description: str, phrases: list[str]):
    haystack = " ".join([description.lower(), " ".join(phrases).lower()])
    matched = []
    for rule in MODULE_SKILL_RULES:
        if any(contains_phrase(haystack, pattern) for pattern in rule["patterns"]):
            matched.extend(rule["skills"])
    return [skill for skill in dict.fromkeys(matched) if is_skill_like(skill)]


def find_explicit_practical_phrases(description: str, phrases: list[str]):
    haystack = description.lower()
    chosen = []
    for phrase in phrases:
        phrase_l = phrase.lower()
        if phrase_l in ACADEMIC_LABEL_BLOCKLIST:
            continue
        if not is_skill_like(phrase):
            continue
        if phrase_l in EXPLICIT_PHRASE_BLOCKLIST:
            continue
        if len(phrase_l.split()) < 2:
            continue
        if len(phrase_l.split()) == 1 and phrase_l.endswith("ed"):
            continue
        if not any(contains_phrase(phrase_l, anchor) for anchor in PRACTICAL_ANCHORS):
            continue
        if not contains_phrase(haystack, phrase_l):
            continue
        chosen.append(phrase)
    return list(dict.fromkeys(chosen))


def semantic_normalize_phrases(
    kw_model: KeyBERT,
    description: str,
    phrases: list[str],
    canonical_skills: list[str],
    canonical_embeddings: torch.Tensor,
    min_score: float,
    max_items: int,
):
    usable_phrases = [
        phrase
        for phrase in phrases
        if is_skill_like(phrase) and phrase.lower() not in EXPLICIT_PHRASE_BLOCKLIST
    ]
    if not usable_phrases:
        return []

    phrase_embeddings = kw_model.model.embed(usable_phrases)
    phrase_embeddings = torch.tensor(phrase_embeddings)
    phrase_embeddings = torch.nn.functional.normalize(phrase_embeddings, dim=1)
    sims = torch.matmul(phrase_embeddings, canonical_embeddings.T)
    values, indices = torch.max(sims, dim=1)

    chosen = []
    seen = set()
    ranked = sorted(
        zip(usable_phrases, values.tolist(), indices.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    for _, score, idx in ranked:
        if score < min_score:
            continue
        mapped = canonical_skills[idx]
        evidence_terms = STRICT_CANONICAL_EVIDENCE.get(mapped)
        if evidence_terms:
            haystack = " ".join([description.lower(), " ".join(usable_phrases).lower()])
            if not any(contains_phrase(haystack, term) for term in evidence_terms):
                continue
        key = mapped.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(mapped)
        if len(chosen) >= max_items:
            break
    return chosen


def merge_skills(rule_skills: list[str], normalized_skills: list[str], phrase_skills: list[str], top_k: int):
    chosen = []
    seen = set()
    for skill in rule_skills + normalized_skills + phrase_skills:
        normalized = normalize_skill(skill)
        key = normalized.lower()
        if not normalized or key in seen or not is_skill_like(normalized):
            continue
        seen.add(key)
        chosen.append(normalized)
        if len(chosen) >= top_k:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Extract module-side skills independently of the job dataset."
    )
    parser.add_argument("--module-input", type=Path, default=DEFAULT_MODULE_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--semantic-min-score", type=float, default=0.62)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    if not args.module_input.exists():
        raise FileNotFoundError(f"Module input file not found: {args.module_input}")

    module_rows = load_module_rows(args.module_input, args.max_rows)

    try:
        st_model = SentenceTransformer(args.model, local_files_only=True)
        print(f"Loaded model from local cache: {args.model}")
    except Exception:
        st_model = SentenceTransformer(args.model)
        print(f"Downloaded model: {args.model}")

    kw_model = KeyBERT(model=st_model)
    canonical_skills = list(dict.fromkeys(CANONICAL_MODULE_SKILLS))
    canonical_embeddings = kw_model.model.embed(canonical_skills)
    canonical_embeddings = torch.tensor(canonical_embeddings)
    canonical_embeddings = torch.nn.functional.normalize(canonical_embeddings, dim=1)

    output_rows = []
    total = len(module_rows)
    for i, row in enumerate(module_rows, start=1):
        description = row.get("description") or ""
        phrases = extract_candidate_phrases(kw_model, description)
        rule_skills = find_rule_based_skills(description, phrases)
        normalized_skills = semantic_normalize_phrases(
            kw_model=kw_model,
            description=description,
            phrases=phrases,
            canonical_skills=canonical_skills,
            canonical_embeddings=canonical_embeddings,
            min_score=args.semantic_min_score,
            max_items=3,
        )
        phrase_skills = find_explicit_practical_phrases(description, phrases)
        row["skills"] = merge_skills(rule_skills, normalized_skills, phrase_skills, args.top_k)
        output_rows.append(row)
        if i % 50 == 0 or i == total:
            print(f"Processed {i}/{total} rows...", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Rule count: {len(MODULE_SKILL_RULES)}")
    print(f"Canonical skill count: {len(canonical_skills)}")
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
