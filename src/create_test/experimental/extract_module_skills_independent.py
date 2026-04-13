import argparse
import json
import re
import sys
from pathlib import Path

import torch
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from module_skill_rules import (
    ACADEMIC_LABEL_BLOCKLIST,
    CANONICAL_MODULE_SKILLS,
    EDU_BLOCKLIST_SUBSTRINGS,
    EXPLICIT_PHRASE_BLOCKLIST,
    MODULE_SKILL_RULES,
    PRACTICAL_ANCHORS,
    STRICT_CANONICAL_EVIDENCE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODULE_INPUT = PROJECT_ROOT / "data" / "test" / "module_descriptions_test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test" / "module_descriptions_test_with_skills_independent.jsonl"
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


def load_module_rows(path: Path, max_rows: int | None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def extract_candidate_phrases(model: SentenceTransformer, description: str, top_n: int = 30):
    text = normalize_text(description)
    if not text:
        return []
    text = text[:2500]

    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    try:
        vectorizer.fit([text])
    except ValueError:
        return []

    candidates = [normalize_skill(phrase) for phrase in vectorizer.get_feature_names_out()]
    candidates = [phrase for phrase in candidates if phrase]
    if not candidates:
        return []

    desc_embedding = model.encode(
        [text],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    phrase_embeddings = model.encode(
        candidates,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = torch.matmul(desc_embedding, phrase_embeddings.T)[0]
    values, indices = torch.topk(scores, k=min(top_n, len(candidates)))

    ranked = []
    seen = set()
    for score, idx in zip(values.tolist(), indices.tolist()):
        phrase = candidates[idx]
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        ranked.append((phrase, score))
    return [phrase for phrase, _ in ranked]


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
    model: SentenceTransformer,
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

    phrase_embeddings = model.encode(
        usable_phrases,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
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

    canonical_skills = list(dict.fromkeys(CANONICAL_MODULE_SKILLS))
    canonical_embeddings = st_model.encode(
        canonical_skills,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    output_rows = []
    total = len(module_rows)
    for i, row in enumerate(module_rows, start=1):
        description = row.get("description") or ""
        phrases = extract_candidate_phrases(st_model, description)
        rule_skills = find_rule_based_skills(description, phrases)
        normalized_skills = semantic_normalize_phrases(
            model=st_model,
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
