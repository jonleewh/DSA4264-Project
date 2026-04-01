import argparse
import json
import re
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODULE_INPUT = PROJECT_ROOT / "data" / "test" / "module_descriptions_test.jsonl"
DEFAULT_JOBS_DIR = PROJECT_ROOT / "data" / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "test" / "module_descriptions_test_with_skills.jsonl"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GENERIC_SKILLS = {
    "management",
    "communications",
    "communication",
    "research",
    "analysis",
    "design",
}


def normalize_skill(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    text = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9+#/]+$", "", text)
    return text


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
            skill = normalize_skill(item.get("skill") if isinstance(item, dict) else str(item))
            if not skill:
                continue
            if len(skill) < 2 or len(skill) > 60:
                continue
            if skill.lower() in GENERIC_SKILLS:
                continue
            vocab.add(skill)
    return sorted(vocab)


def rank_skills_for_description(
    model: SentenceTransformer,
    description: str,
    skill_vocab: list[str],
    skill_embeddings: torch.Tensor,
    top_k: int,
    min_score: float,
):
    desc_text = (description or "").strip()[:2500]
    if not desc_text:
        return []

    desc_embedding = model.encode(
        [desc_text],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    scores = util.cos_sim(desc_embedding, skill_embeddings)[0]

    top_count = min(max(top_k * 4, top_k), len(skill_vocab))
    values, indices = torch.topk(scores, k=top_count)

    selected = []
    seen = set()
    for score, idx in zip(values.tolist(), indices.tolist()):
        if score < min_score:
            continue
        skill = skill_vocab[idx]
        key = skill.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(skill)
        if len(selected) >= top_k:
            break

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Extract relevant skills for module descriptions using transformer semantic matching."
    )
    parser.add_argument("--module-input", type=Path, default=DEFAULT_MODULE_INPUT)
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.33)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    if not args.module_input.exists():
        raise FileNotFoundError(f"Module input file not found: {args.module_input}")
    if not args.jobs_dir.exists():
        raise FileNotFoundError(f"Jobs directory not found: {args.jobs_dir}")

    module_rows = load_module_rows(args.module_input, args.max_rows)
    skill_vocab = load_job_skill_vocab(args.jobs_dir)
    if not skill_vocab:
        raise RuntimeError(f"No skills found in job files under: {args.jobs_dir}")

    model = SentenceTransformer(args.model)
    skill_embeddings = model.encode(
        skill_vocab,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    output_rows = []
    total = len(module_rows)
    for i, row in enumerate(module_rows, start=1):
        row["skills"] = rank_skills_for_description(
            model=model,
            description=row.get("description") or "",
            skill_vocab=skill_vocab,
            skill_embeddings=skill_embeddings,
            top_k=args.top_k,
            min_score=args.min_score,
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
