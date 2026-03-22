import argparse
import json
import re
from pathlib import Path

import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODULE_INPUT = PROJECT_ROOT / "data" / "test" / "module_descriptions_test.jsonl"
DEFAULT_JOBS_DIR = PROJECT_ROOT / "data" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test" / "module_descriptions_test_with_skills_keybert.jsonl"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EDU_BLOCKLIST_SUBSTRINGS = [
    "course",
    "module",
    "student",
    "assessment",
    "assignment",
    "lecturer",
    "classroom",
    "prerequisite",
    "seminar",
    "credits",
    "term ",
    "workload",
]


def normalize_skill(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    text = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9+#/]+$", "", text)
    return text


def is_industry_relevant(skill: str) -> bool:
    s = skill.lower().strip()
    if not s:
        return False
    if len(s) < 2 or len(s) > 60:
        return False
    for bad in EDU_BLOCKLIST_SUBSTRINGS:
        if bad in s:
            return False
    return True


def load_module_rows(path: Path, max_rows: int | None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def load_job_skill_vocab(jobs_dir: Path):
    vocab = set()
    for path in jobs_dir.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in row.get("skills") or []:
            skill = item.get("skill") if isinstance(item, dict) else str(item)
            skill = normalize_skill(skill)
            if is_industry_relevant(skill):
                vocab.add(skill)
    return sorted(vocab)


def extract_skills_for_description(
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

    ranked = kw_model.extract_keywords(
        text[:2500],
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=30,
        use_mmr=True,
        diversity=0.8,
    )

    phrases = [normalize_skill(k) for k, _ in ranked if normalize_skill(k)]
    if not phrases:
        return []

    phrase_embeddings = kw_model.model.embed(phrases)
    phrase_embeddings = torch.tensor(phrase_embeddings)
    phrase_embeddings = torch.nn.functional.normalize(phrase_embeddings, dim=1)

    desc_embedding = kw_model.model.embed([text[:2500]])
    desc_embedding = torch.tensor(desc_embedding)
    desc_embedding = torch.nn.functional.normalize(desc_embedding, dim=1)
    desc_scores = torch.matmul(desc_embedding, skill_embeddings.T)[0]
    top_direct = set(torch.topk(desc_scores, k=min(200, len(skill_vocab))).indices.tolist())

    chosen = []
    seen = set()
    sims = torch.matmul(phrase_embeddings, skill_embeddings.T)
    values, indices = torch.max(sims, dim=1)

    for phrase, score, idx in zip(phrases, values.tolist(), indices.tolist()):
        if score < map_min_score:
            continue
        if idx not in top_direct:
            continue
        if desc_scores[idx].item() < desc_min_score:
            continue
        skill = skill_vocab[idx]
        if not is_industry_relevant(skill):
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
        description="Extract industry-relevant skills from module descriptions using KeyBERT."
    )
    parser.add_argument("--module-input", type=Path, default=DEFAULT_MODULE_INPUT)
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--map-min-score", type=float, default=0.42)
    parser.add_argument("--desc-min-score", type=float, default=0.34)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    if not args.module_input.exists():
        raise FileNotFoundError(f"Module input file not found: {args.module_input}")
    if not args.jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {args.jobs_dir}")

    module_rows = load_module_rows(args.module_input, args.max_rows)
    skill_vocab = load_job_skill_vocab(args.jobs_dir)
    if not skill_vocab:
        raise RuntimeError(f"No usable skills found under: {args.jobs_dir}")

    try:
        st_model = SentenceTransformer(args.model, local_files_only=True)
        print(f"Loaded model from local cache: {args.model}")
    except Exception:
        st_model = SentenceTransformer(args.model)
        print(f"Downloaded model: {args.model}")
    kw_model = KeyBERT(model=st_model)
    skill_embeddings = kw_model.model.embed(skill_vocab)
    skill_embeddings = torch.tensor(skill_embeddings)
    skill_embeddings = torch.nn.functional.normalize(skill_embeddings, dim=1)

    output_rows = []
    total = len(module_rows)
    for i, row in enumerate(module_rows, start=1):
        row["skills"] = extract_skills_for_description(
            kw_model=kw_model,
            description=row.get("description") or "",
            skill_vocab=skill_vocab,
            skill_embeddings=skill_embeddings,
            top_k=args.top_k,
            map_min_score=args.map_min_score,
            desc_min_score=args.desc_min_score,
        )
        output_rows.append(row)
        if i % 50 == 0 or i == total:
            print(f"Processed {i}/{total} rows...", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Skill vocabulary size: {len(skill_vocab)}")
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
