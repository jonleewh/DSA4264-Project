import argparse
import json
import random
import re
from pathlib import Path

import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEANED_JOBS_INPUT = PROJECT_ROOT / "data" / "processed" / "job_freshgrad_cleaned.json"
DEFAULT_TEST_OUTPUT = PROJECT_ROOT / "data" / "test" / "job_descriptions_test.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test" / "job_descriptions_test_with_skills_keybert.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_CLEANED_FIELDS = ("id", "source", "title")

BLOCKLIST_SUBSTRINGS = [
    "course",
    "module",
    "assessment",
    "student",
    "learning",
    "assignment",
    "lecturer",
    "classroom",
    "prerequisite",
    "credits",
]


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9+#/]+$", "", text)
    return text


def is_valid_skill(skill: str) -> bool:
    s = skill.lower().strip()
    if not s:
        return False
    if len(s) < 2 or len(s) > 60:
        return False
    return not any(bad in s for bad in BLOCKLIST_SUBSTRINGS)


def load_cleaned_rows(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(rows).__name__}.")
    for i, row in enumerate(rows, start=1):
        missing = [k for k in REQUIRED_CLEANED_FIELDS if k not in row]
        if missing:
            raise ValueError(
                f"Invalid cleaned dataset row {i} in {path}. Missing fields: {missing}."
            )
    return rows


def build_test_rows(
    cleaned_rows: list[dict],
    test_size: int,
    min_description_length: int,
    seed: int,
    fresh_only: bool,
):
    candidates = []
    for row in cleaned_rows:
        if fresh_only and not row.get("is_freshgrad"):
            continue
        description = normalize_text(row.get("description_clean") or row.get("description") or "")
        if len(description) < min_description_length:
            continue
        candidates.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "title": row.get("title"),
                "description": description,
            }
        )
    random.Random(seed).shuffle(candidates)
    return candidates[: min(test_size, len(candidates))]


def load_skill_vocab(cleaned_rows: list[dict], fresh_only: bool):
    vocab = set()
    for row in cleaned_rows:
        if fresh_only and not row.get("is_freshgrad"):
            continue

        for key in ("all_relevant_skills", "hard_skills", "soft_skills"):
            for item in row.get(key) or []:
                skill = normalize_text(str(item))
                if is_valid_skill(skill):
                    vocab.add(skill)
    return sorted(vocab)


def extract_skills_for_job(
    kw_model: KeyBERT,
    description: str,
    skill_vocab: list[str],
    skill_embeddings: torch.Tensor,
    top_k: int,
    map_min_score: float,
    desc_min_score: float,
):
    text = (description or "").strip()
    if not text:
        return []

    ranked_phrases = kw_model.extract_keywords(
        text[:3000],
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=40,
        use_mmr=True,
        diversity=0.8,
    )
    phrases = [normalize_text(k) for k, _ in ranked_phrases if normalize_text(k)]
    if not phrases:
        return []

    phrase_emb = kw_model.model.embed(phrases)
    phrase_emb = torch.tensor(phrase_emb)
    phrase_emb = torch.nn.functional.normalize(phrase_emb, dim=1)

    desc_emb = kw_model.model.embed([text[:3000]])
    desc_emb = torch.tensor(desc_emb)
    desc_emb = torch.nn.functional.normalize(desc_emb, dim=1)
    desc_scores = torch.matmul(desc_emb, skill_embeddings.T)[0]
    top_direct = set(torch.topk(desc_scores, k=min(300, len(skill_vocab))).indices.tolist())

    sims = torch.matmul(phrase_emb, skill_embeddings.T)
    values, indices = torch.max(sims, dim=1)

    chosen = []
    seen = set()
    for _, map_score, idx in sorted(
        zip(phrases, values.tolist(), indices.tolist()),
        key=lambda x: x[1],
        reverse=True,
    ):
        if map_score < map_min_score:
            continue
        if idx not in top_direct:
            continue
        if desc_scores[idx].item() < desc_min_score:
            continue

        skill = skill_vocab[idx]
        if not is_valid_skill(skill):
            continue

        key = skill.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(skill)
        if len(chosen) >= top_k:
            break

    return chosen


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate fresh-grad job test dataset from cleaned jobs and extract skills with KeyBERT."
        )
    )
    parser.add_argument("--cleaned-jobs-input", type=Path, default=DEFAULT_CLEANED_JOBS_INPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--min-description-length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-non-fresh", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--map-min-score", type=float, default=0.46)
    parser.add_argument("--desc-min-score", type=float, default=0.36)
    args = parser.parse_args()

    if not args.cleaned_jobs_input.exists():
        raise FileNotFoundError(f"Cleaned input file not found: {args.cleaned_jobs_input}")

    fresh_only = not args.include_non_fresh
    cleaned_rows = load_cleaned_rows(args.cleaned_jobs_input)
    job_rows = build_test_rows(
        cleaned_rows=cleaned_rows,
        test_size=args.test_size,
        min_description_length=args.min_description_length,
        seed=args.seed,
        fresh_only=fresh_only,
    )
    if not job_rows:
        raise RuntimeError("No candidate rows available to build job test dataset.")

    skill_vocab = load_skill_vocab(cleaned_rows, fresh_only=fresh_only)
    if not skill_vocab:
        raise RuntimeError("No skill vocabulary found in cleaned job file.")

    st_model = SentenceTransformer(args.model, local_files_only=True)
    kw_model = KeyBERT(model=st_model)
    skill_embeddings = kw_model.model.embed(skill_vocab)
    skill_embeddings = torch.tensor(skill_embeddings)
    skill_embeddings = torch.nn.functional.normalize(skill_embeddings, dim=1)

    output_rows = []
    total = len(job_rows)
    for i, row in enumerate(job_rows, start=1):
        row["skills"] = extract_skills_for_job(
            kw_model=kw_model,
            description=row.get("description") or "",
            skill_vocab=skill_vocab,
            skill_embeddings=skill_embeddings,
            top_k=args.top_k,
            map_min_score=args.map_min_score,
            desc_min_score=args.desc_min_score,
        )
        output_rows.append(row)
        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total} rows...", flush=True)

    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.write_text(
        json.dumps(job_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Generated job test rows: {len(job_rows)}")
    print(f"Saved base test set to: {args.test_output}")
    print(f"Skill vocabulary size: {len(skill_vocab)}")
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
