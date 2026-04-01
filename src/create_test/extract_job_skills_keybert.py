import argparse
import json
import re
from pathlib import Path

import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB_INPUT = PROJECT_ROOT / "data" / "test" / "job_descriptions_test.jsonl"
DEFAULT_JOBS_DIR = PROJECT_ROOT / "data" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test" / "job_descriptions_test_with_skills_keybert.jsonl"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_skill_vocab(jobs_dir: Path):
    vocab = set()
    for path in jobs_dir.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in row.get("skills") or []:
            if isinstance(item, dict):
                skill = normalize_text(item.get("skill"))
            else:
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
        description="Extract industry-relevant skills from job descriptions using KeyBERT."
    )
    parser.add_argument("--job-input", type=Path, default=DEFAULT_JOB_INPUT)
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--map-min-score", type=float, default=0.46)
    parser.add_argument("--desc-min-score", type=float, default=0.36)
    args = parser.parse_args()

    if not args.job_input.exists():
        raise FileNotFoundError(f"Input file not found: {args.job_input}")
    if not args.jobs_dir.exists():
        raise FileNotFoundError(f"Jobs dir not found: {args.jobs_dir}")

    job_rows = load_jsonl(args.job_input)
    skill_vocab = load_skill_vocab(args.jobs_dir)
    if not skill_vocab:
        raise RuntimeError("No skill vocabulary found from original job files.")

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Skill vocabulary size: {len(skill_vocab)}")
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
