import argparse
import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMEWORK = PROJECT_ROOT / "data" / "reference" / "canonical_skill_framework_v4.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^a-z0-9+#/]+|[^a-z0-9+#/]+$", "", text)
    return text


def load_framework(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class CanonicalSkillMapper:
    def __init__(self, framework_path: Path, model_name: str, threshold: float = 0.72):
        self.framework_path = framework_path
        self.model_name = model_name
        self.threshold = threshold
        self.framework = load_framework(framework_path)
        self.skills = self.framework["skills"]
        self.excluded_phrases = {normalize_text(x) for x in (self.framework.get("excluded_phrases") or []) if normalize_text(x)}
        self.alias_to_skill = {}
        self.semantic_aliases = []
        self.semantic_targets = []
        self.model = None
        self.alias_embeddings = None
        self.skill_lookup = {row["canonical_skill"]: row for row in self.skills}
        self.phrase_cache = {}
        self._prepare_alias_maps()

    def _prepare_alias_maps(self):
        for row in self.skills:
            canonical_skill = row["canonical_skill"]
            aliases = row.get("aliases") or []
            candidates = [canonical_skill, *aliases]
            for alias in candidates:
                norm = normalize_text(alias)
                if not norm:
                    continue
                self.alias_to_skill[norm] = canonical_skill
                self.semantic_aliases.append(norm)
                self.semantic_targets.append(canonical_skill)

    def _ensure_model(self):
        if self.model is not None and self.alias_embeddings is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for semantic fallback") from exc

        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name)

        self.alias_embeddings = self.model.encode(
            self.semantic_aliases,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def map_phrase(self, phrase: str) -> dict:
        normalized = normalize_text(phrase)
        cached = self.phrase_cache.get(normalized)
        if cached is not None:
            return {
                **cached,
                "raw_phrase": phrase,
                "normalized_phrase": normalized,
            }

        if not normalized:
            result = {
                "raw_phrase": phrase,
                "normalized_phrase": normalized,
                "canonical_skill": "",
                "skill_type": "",
                "match_type": "empty",
                "score": 0.0,
            }
            self.phrase_cache[normalized] = {
                "canonical_skill": result["canonical_skill"],
                "skill_type": result["skill_type"],
                "match_type": result["match_type"],
                "score": result["score"],
            }
            return result

        if normalized in self.excluded_phrases:
            result = {
                "raw_phrase": phrase,
                "normalized_phrase": normalized,
                "canonical_skill": "",
                "skill_type": "",
                "match_type": "excluded",
                "score": 1.0,
            }
            self.phrase_cache[normalized] = {
                "canonical_skill": result["canonical_skill"],
                "skill_type": result["skill_type"],
                "match_type": result["match_type"],
                "score": result["score"],
            }
            return result

        exact = self.alias_to_skill.get(normalized)
        if exact:
            skill_row = self.skill_lookup[exact]
            result = {
                "raw_phrase": phrase,
                "normalized_phrase": normalized,
                "canonical_skill": exact,
                "skill_type": skill_row.get("skill_type", ""),
                "match_type": "exact",
                "score": 1.0,
            }
            self.phrase_cache[normalized] = {
                "canonical_skill": result["canonical_skill"],
                "skill_type": result["skill_type"],
                "match_type": result["match_type"],
                "score": result["score"],
            }
            return result

        self._ensure_model()
        phrase_embedding = self.model.encode(
            [normalized],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        sims = np.dot(self.alias_embeddings, phrase_embedding)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < self.threshold:
            result = {
                "raw_phrase": phrase,
                "normalized_phrase": normalized,
                "canonical_skill": normalized,
                "skill_type": "",
                "match_type": "unmapped",
                "score": round(best_score, 6),
            }
            self.phrase_cache[normalized] = {
                "canonical_skill": result["canonical_skill"],
                "skill_type": result["skill_type"],
                "match_type": result["match_type"],
                "score": result["score"],
            }
            return result

        mapped = self.semantic_targets[best_idx]
        skill_row = self.skill_lookup[mapped]
        result = {
            "raw_phrase": phrase,
            "normalized_phrase": normalized,
            "canonical_skill": mapped,
            "skill_type": skill_row.get("skill_type", ""),
            "match_type": "semantic",
            "score": round(best_score, 6),
        }
        self.phrase_cache[normalized] = {
            "canonical_skill": result["canonical_skill"],
            "skill_type": result["skill_type"],
            "match_type": result["match_type"],
            "score": result["score"],
        }
        return result

    def map_phrases(self, phrases: list[str]) -> list[dict]:
        return [self.map_phrase(phrase) for phrase in phrases]

    def warm_cache(self, phrases: list[str], batch_size: int = 256):
        normalized_phrases = [normalize_text(phrase) for phrase in phrases if normalize_text(phrase)]
        unique_phrases = list(dict.fromkeys(normalized_phrases))
        pending = []

        for normalized in unique_phrases:
            if normalized in self.phrase_cache:
                continue
            exact = self.alias_to_skill.get(normalized)
            if exact:
                skill_row = self.skill_lookup[exact]
                self.phrase_cache[normalized] = {
                    "canonical_skill": exact,
                    "skill_type": skill_row.get("skill_type", ""),
                    "match_type": "exact",
                    "score": 1.0,
                }
            else:
                pending.append(normalized)

        if not pending:
            return

        self._ensure_model()
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            sims = np.matmul(batch_embeddings, self.alias_embeddings.T)
            best_indices = np.argmax(sims, axis=1)
            best_scores = np.max(sims, axis=1)

            for normalized, best_idx, best_score in zip(batch, best_indices.tolist(), best_scores.tolist()):
                if float(best_score) < self.threshold:
                    self.phrase_cache[normalized] = {
                        "canonical_skill": normalized,
                        "skill_type": "",
                        "match_type": "unmapped",
                        "score": round(float(best_score), 6),
                    }
                else:
                    mapped = self.semantic_targets[best_idx]
                    skill_row = self.skill_lookup[mapped]
                    self.phrase_cache[normalized] = {
                        "canonical_skill": mapped,
                        "skill_type": skill_row.get("skill_type", ""),
                        "match_type": "semantic",
                        "score": round(float(best_score), 6),
                    }


def parse_args():
    parser = argparse.ArgumentParser(description="Map raw phrases into a canonical skill framework.")
    parser.add_argument("--framework", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument(
        "--phrases",
        type=str,
        default="",
        help="Comma-separated phrases to map for quick testing.",
    )
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--input-field", type=str, default="skills")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-field", type=str, default="canonical_skills")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    mapper = CanonicalSkillMapper(
        framework_path=args.framework,
        model_name=args.model,
        threshold=args.threshold,
    )

    if args.phrases:
        phrases = [part.strip() for part in args.phrases.split(",") if part.strip()]
        print(json.dumps(mapper.map_phrases(phrases), indent=2, ensure_ascii=False))
        return

    if args.input_jsonl and args.output_jsonl:
        rows = []
        all_phrases = []
        with args.input_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                phrases = row.get(args.input_field) or []
                 # collect phrases first so unique unresolved items can be cached in batches
                all_phrases.extend(phrases)
                rows.append(row)

        mapper.warm_cache(all_phrases, batch_size=args.batch_size)

        mapped_rows = []
        for row in rows:
            phrases = row.get(args.input_field) or []
            mapped = mapper.map_phrases(phrases)
            row[args.output_field] = sorted(
                {item["canonical_skill"] for item in mapped if item["canonical_skill"]},
            )
            row[f"{args.output_field}_details"] = mapped
            mapped_rows.append(row)

        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for row in mapped_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Mapped {len(mapped_rows)} rows using framework: {args.framework}")
        print(f"Saved output to: {args.output_jsonl}")
        return

    raise SystemExit("Provide either --phrases or both --input-jsonl and --output-jsonl.")


if __name__ == "__main__":
    main()
