from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS_PATH = PROJECT_ROOT / "data" / "cleaned_data" / "jobs_cleaned.pkl"
STEM_FULL_DIR = PROJECT_ROOT / "data" / "stem_full"
STEM_MODULE_CANONICAL_PATH = STEM_FULL_DIR / "module_skills_canonical_stem.jsonl"
STEM_JOB_CANONICAL_PATH = STEM_FULL_DIR / "job_skills_canonical_stem.jsonl"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "course",
    "courses",
    "entry",
    "find",
    "for",
    "from",
    "full",
    "i",
    "in",
    "into",
    "is",
    "job",
    "jobs",
    "level",
    "looking",
    "me",
    "module",
    "modules",
    "most",
    "of",
    "on",
    "or",
    "part",
    "please",
    "relevant",
    "role",
    "roles",
    "show",
    "that",
    "the",
    "time",
    "to",
    "use",
    "used",
    "uses",
    "using",
    "want",
    "what",
    "which",
    "with",
}

EXPERIENCE_HINTS = {
    "entry level",
    "entry-level",
    "fresh grad",
    "fresh grads",
    "fresh graduate",
    "fresh graduates",
    "junior",
    "no experience",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: object) -> list[str]:
    return [
        token
        for token in normalize_text(value).split()
        if len(token) > 1 and token not in STOPWORDS
    ]


def parse_money(value: str) -> int | None:
    cleaned = value.lower().replace(",", "").replace("$", "").strip()
    if not cleaned:
        return None
    multiplier = 1000 if cleaned.endswith("k") else 1
    if cleaned.endswith("k"):
        cleaned = cleaned[:-1]
    try:
        return int(float(cleaned) * multiplier)
    except ValueError:
        return None


def safe_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()]


def normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    dot = sum(a[key] * b[key] for key in common)
    na = math.sqrt(sum(value * value for value in a.values()))
    nb = math.sqrt(sum(value * value for value in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a).union(b)
    if not keys:
        return 0.0
    num = sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    den = sum(max(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    return num / den if den else 0.0


def topk_coverage(module_counter: Counter[str], job_counter: Counter[str], top_k: int) -> float:
    top_job = [skill for skill, _ in job_counter.most_common(top_k)]
    if not top_job:
        return 0.0
    module_skills = set(module_counter)
    covered = sum(1 for skill in top_job if skill in module_skills)
    return covered / len(top_job)


def gap_score(module_counter: Counter[str], job_counter: Counter[str], top_k: int) -> float:
    top_job = job_counter.most_common(top_k)
    if not top_job:
        return 0.0
    total_weight = sum(weight for _, weight in top_job)
    if total_weight <= 0:
        return 0.0
    module_skills = set(module_counter)
    missing_weight = sum(weight for skill, weight in top_job if skill not in module_skills)
    return missing_weight / total_weight


def alignment_score(coverage: float, wj: float, cos: float, gap: float) -> float:
    score = 0.4 * coverage + 0.25 * wj + 0.2 * cos + 0.15 * (1.0 - gap)
    return max(0.0, min(1.0, score))


@dataclass
class QueryIntent:
    raw_query: str
    normalized_query: str
    tokens: list[str]
    non_skill_tokens: list[str]
    skills: set[str]
    work_type: str | None
    salary_min: int | None
    salary_max: int | None
    experience_max: int | None


class LocalJobCourseBot:
    def __init__(
        self,
        jobs_path: Path = JOBS_PATH,
        stem_module_canonical_path: Path = STEM_MODULE_CANONICAL_PATH,
        stem_job_canonical_path: Path = STEM_JOB_CANONICAL_PATH,
    ):
        self.jobs_df = pd.read_pickle(jobs_path)
        self.jobs = self._prepare_jobs(self.jobs_df)

        self.stem_module_canonical_path = stem_module_canonical_path
        self.stem_job_canonical_path = stem_job_canonical_path
        self.stem_data_status = self._load_stem_data()
        self.skill_vocabulary = self._build_skill_vocabulary()

    def _load_stem_data(self) -> dict[str, Any]:
        missing = [
            str(path.relative_to(PROJECT_ROOT))
            for path in [self.stem_module_canonical_path, self.stem_job_canonical_path]
            if not path.exists()
        ]
        if missing:
            self.stem_modules = []
            self.stem_jobs_canonical = []
            self.stem_jobs_by_post_id = {}
            return {
                "available": False,
                "message": (
                    "STEM module recommendations are disabled because the stem_test pipeline "
                    "outputs are missing."
                ),
                "missing_paths": missing,
            }

        module_rows = load_jsonl(self.stem_module_canonical_path)
        job_rows = load_jsonl(self.stem_job_canonical_path)

        self.stem_modules = []
        for row in module_rows:
            skills = sorted(
                {
                    normalize_text(skill)
                    for skill in safe_list(row.get("canonical_skills"))
                    if normalize_text(skill)
                }
            )
            self.stem_modules.append(
                {
                    "id": str(row.get("id") or row.get("module_id") or "").strip(),
                    "source": str(row.get("source") or "").strip(),
                    "title": str(row.get("title") or "").strip(),
                    "canonical_skills": skills,
                    "skill_counter": Counter(skills),
                    "skill_weights": normalize_counter(Counter(skills)),
                }
            )

        self.stem_jobs_canonical = []
        self.stem_jobs_by_post_id = {}
        for row in job_rows:
            post_id = str(row.get("job_post_id") or "").strip()
            skills = sorted(
                {
                    normalize_text(skill)
                    for skill in safe_list(row.get("canonical_skills"))
                    if normalize_text(skill)
                }
            )
            job_row = {
                "job_post_id": post_id,
                "ssoc_3d_code": str(row.get("ssoc_3d_code") or "").strip(),
                "ssoc_3d_title": str(row.get("ssoc_3d_title") or "").strip(),
                "canonical_skills": skills,
            }
            self.stem_jobs_canonical.append(job_row)
            if post_id:
                self.stem_jobs_by_post_id[post_id] = job_row

        return {
            "available": True,
            "message": (
                "Using stem_test canonical skill outputs for STEM-only module recommendations."
            ),
            "missing_paths": [],
        }

    def _build_skill_vocabulary(self) -> list[str]:
        vocab = set()
        for job in self.jobs:
            vocab.update(job["skills"])
        if self.stem_data_status["available"]:
            for module in self.stem_modules:
                vocab.update(module["canonical_skills"])
            for job in self.stem_jobs_canonical:
                vocab.update(job["canonical_skills"])
        return sorted(skill for skill in vocab if skill)

    def _prepare_jobs(self, jobs_df: pd.DataFrame) -> list[dict]:
        jobs = []
        for row in jobs_df.fillna("").to_dict(orient="records"):
            title = str(row.get("title", "")).strip()
            description = str(row.get("description", "")).strip()
            categories = safe_list(row.get("categories"))
            skills = [normalize_text(skill) for skill in safe_list(row.get("skills_clean"))]
            title_tokens = set(tokenize(title))
            description_tokens = set(tokenize(description))
            category_tokens = set(tokenize(" ".join(categories)))
            jobs.append(
                {
                    "uuid": str(row.get("uuid") or "").strip(),
                    "title": title,
                    "description": description,
                    "skills": [skill for skill in skills if skill],
                    "skill_set": set(skill for skill in skills if skill),
                    "categories": categories,
                    "title_norm": normalize_text(title),
                    "description_norm": normalize_text(description),
                    "title_tokens": title_tokens,
                    "description_tokens": description_tokens,
                    "category_tokens": category_tokens,
                    "work_type": str(row.get("work_type") or "").strip(),
                    "contract_type": str(row.get("contract_type") or "").strip(),
                    "salary_minimum": float(row.get("salary_minimum") or 0),
                    "salary_maximum": float(row.get("salary_maximum") or 0),
                    "avg_salary": float(row.get("avg_salary") or 0),
                    "minimum_years_experience": int(row.get("minimum_years_experience") or 0),
                }
            )
        return jobs

    def interpret_query(self, query: str) -> QueryIntent:
        normalized = normalize_text(query)
        tokens = tokenize(query)
        skills = {
            skill
            for skill in self.skill_vocabulary
            if skill and f" {skill} " in f" {normalized} "
        }
        skill_tokens = {token for skill in skills for token in tokenize(skill)}
        non_skill_tokens = [token for token in tokens if token not in skill_tokens]

        work_type = None
        if "part time" in normalized or "part-time" in normalized:
            work_type = "Part Time"
        elif "full time" in normalized or "full-time" in normalized:
            work_type = "Full Time"

        salary_min = None
        salary_max = None
        lower_match = re.search(
            r"(?:above|over|at least|min(?:imum)?|more than)\s*\$?\s*([0-9]+(?:\.[0-9]+)?k?)",
            normalized,
        )
        if lower_match:
            salary_min = parse_money(lower_match.group(1))

        upper_match = re.search(
            r"(?:below|under|up to|less than|max(?:imum)?)\s*\$?\s*([0-9]+(?:\.[0-9]+)?k?)",
            normalized,
        )
        if upper_match:
            salary_max = parse_money(upper_match.group(1))

        experience_max = 1 if any(hint in normalized for hint in EXPERIENCE_HINTS) else None

        return QueryIntent(
            raw_query=query,
            normalized_query=normalized,
            tokens=tokens,
            non_skill_tokens=non_skill_tokens,
            skills=skills,
            work_type=work_type,
            salary_min=salary_min,
            salary_max=salary_max,
            experience_max=experience_max,
        )

    def _score_job(self, job: dict, intent: QueryIntent) -> tuple[float, dict]:
        token_set = set(intent.tokens)
        non_skill_token_set = set(intent.non_skill_tokens)
        title_hits = len(non_skill_token_set & job["title_tokens"])
        desc_hits = len(non_skill_token_set & job["description_tokens"])
        category_hits = len(non_skill_token_set & job["category_tokens"])
        skill_hits = len(intent.skills & job["skill_set"])

        score = 0.0
        score += title_hits * 3.2
        score += desc_hits * 0.45
        score += category_hits * 1.2
        score += skill_hits * 3.0

        if intent.normalized_query and intent.normalized_query in job["title_norm"]:
            score += 5.0

        if intent.work_type:
            score += 1.5 if job["work_type"] == intent.work_type else -1.5

        if intent.experience_max is not None:
            score += 1.0 if job["minimum_years_experience"] <= intent.experience_max else -2.0

        if intent.salary_min is not None:
            score += 1.0 if job["salary_maximum"] >= intent.salary_min else -2.5

        if intent.salary_max is not None:
            score += 0.6 if job["salary_minimum"] <= intent.salary_max else -1.5

        matched_tokens = sorted(
            (non_skill_token_set & job["title_tokens"])
            | (non_skill_token_set & job["category_tokens"])
            | (token_set & job["description_tokens"])
        )
        matched_skills = sorted(intent.skills & job["skill_set"])

        return score, {
            "matched_tokens": matched_tokens,
            "matched_skills": matched_skills,
            "title_hits": title_hits,
            "desc_hits": desc_hits,
            "category_hits": category_hits,
            "skill_hits": skill_hits,
        }

    def search_jobs(self, intent: QueryIntent, limit: int = 5) -> tuple[list[dict], list[dict]]:
        scored = []
        for job in self.jobs:
            score, debug = self._score_job(job, intent)
            core_signal = (
                debug["title_hits"]
                + debug["skill_hits"]
                + debug["category_hits"]
                + (1 if debug["desc_hits"] >= 2 else 0)
            )
            if score <= 0 or core_signal == 0:
                continue
            if len(intent.skills) >= 2 and debug["title_hits"] == 0 and debug["category_hits"] == 0:
                if debug["skill_hits"] < min(2, len(intent.skills)):
                    continue
            scored.append(
                {
                    **job,
                    "search_score": round(score, 3),
                    **debug,
                }
            )

        scored.sort(
            key=lambda row: (
                row["search_score"],
                len(row["matched_skills"]),
                row["avg_salary"],
            ),
            reverse=True,
        )
        return scored[:limit], scored

    def _build_skill_profile(
        self,
        ranked_jobs: list[dict],
        top_k_jobs: int = 10,
        top_k_skills: int = 12,
    ) -> list[tuple[str, float]]:
        counter: Counter[str] = Counter()
        for rank, job in enumerate(ranked_jobs[:top_k_jobs], start=1):
            weight = max(job["search_score"], 0.1) / math.sqrt(rank)
            for skill in job["skills"]:
                counter[skill] += weight
        return counter.most_common(top_k_skills)

    def _build_canonical_job_profile(
        self,
        ranked_jobs: list[dict],
        top_k_jobs: int = 25,
    ) -> tuple[Counter[str], dict[str, float], list[str]]:
        counter: Counter[str] = Counter()
        matched_post_ids = []
        for rank, job in enumerate(ranked_jobs[:top_k_jobs], start=1):
            canonical_row = self.stem_jobs_by_post_id.get(job["uuid"])
            if not canonical_row:
                continue
            matched_post_ids.append(job["uuid"])
            weight = max(job["search_score"], 0.1) / math.sqrt(rank)
            for skill in canonical_row["canonical_skills"]:
                counter[skill] += weight
        return counter, normalize_counter(counter), matched_post_ids

    def recommend_modules(
        self,
        ranked_jobs: list[dict],
        limit: int = 5,
        top_k_job_skills: int = 10,
    ) -> tuple[list[dict], dict[str, Any]]:
        if not self.stem_data_status["available"]:
            return [], {
                "available": False,
                "message": self.stem_data_status["message"],
                "missing_paths": self.stem_data_status["missing_paths"],
                "canonical_job_match_count": 0,
            }

        job_counter, job_weights, matched_post_ids = self._build_canonical_job_profile(ranked_jobs)
        if not job_counter:
            return [], {
                "available": False,
                "message": (
                    "STEM outputs are present, but none of the matched job ads could be linked "
                    "to canonical stem_test job rows."
                ),
                "missing_paths": [],
                "canonical_job_match_count": 0,
            }

        recommendations = []
        for module in self.stem_modules:
            module_counter = module["skill_counter"]
            module_weights = module["skill_weights"]
            coverage = topk_coverage(module_counter, job_counter, top_k_job_skills)
            wj = weighted_jaccard(module_weights, job_weights)
            cos = cosine_similarity(module_weights, job_weights)
            gap = gap_score(module_counter, job_counter, top_k_job_skills)
            score = alignment_score(coverage, wj, cos, gap)
            overlap = sorted(set(module_counter) & set(job_counter))
            if score <= 0 or not overlap:
                continue
            recommendations.append(
                {
                    "id": module["id"],
                    "source": module["source"],
                    "title": module["title"],
                    "alignment_score": round(score, 4),
                    "coverage_top_k": round(coverage, 4),
                    "weighted_jaccard": round(wj, 4),
                    "cosine_similarity": round(cos, 4),
                    "gap_score": round(gap, 4),
                    "matched_skills": overlap[:6],
                    "missing_skills": [
                        skill for skill, _ in job_counter.most_common(top_k_job_skills) if skill not in module_counter
                    ][:5],
                }
            )

        recommendations.sort(
            key=lambda row: (
                row["alignment_score"],
                row["coverage_top_k"],
                row["weighted_jaccard"],
            ),
            reverse=True,
        )
        return recommendations[:limit], {
            "available": True,
            "message": self.stem_data_status["message"],
            "missing_paths": [],
            "canonical_job_match_count": len(matched_post_ids),
        }

    def _build_summary(
        self,
        intent: QueryIntent,
        all_ranked_jobs: list[dict],
        top_jobs: list[dict],
        top_modules: list[dict],
        stem_status: dict[str, Any],
    ) -> str:
        if not top_jobs:
            return (
                "I could not find strong matches for that query in the cleaned job dataset yet. "
                "Try adding a job family, skill, work type, or salary range."
            )

        top_skill_profile = self._build_skill_profile(all_ranked_jobs, top_k_jobs=8, top_k_skills=6)
        top_skill_names = ", ".join(skill for skill, _ in top_skill_profile[:5]) or "no clear shared skill cluster"
        avg_salary = int(
            sum(job["avg_salary"] for job in top_jobs if job["avg_salary"]) / max(len(top_jobs), 1)
        )
        work_types = Counter(job["work_type"] for job in top_jobs if job["work_type"])
        dominant_work_type = work_types.most_common(1)[0][0] if work_types else "mixed"

        sentence = (
            f'I interpreted your query as "{intent.raw_query.strip()}". '
            f"I found {len(all_ranked_jobs)} relevant job ads, with the strongest matches clustering around "
            f"{top_skill_names}. The top results average about SGD {avg_salary:,} and are mostly {dominant_work_type.lower()} roles."
        )

        if top_modules:
            best_module = top_modules[0]
            covered = ", ".join(best_module["matched_skills"][:4]) or "shared skills"
            sentence += (
                f" The top STEM module recommendation is {best_module['id']} {best_module['title']} "
                f"because its canonical skill overlap includes {covered}."
            )
        elif not stem_status["available"]:
            sentence += f" {stem_status['message']}"
        return sentence

    def run_query(self, query: str, top_job_limit: int = 5, top_module_limit: int = 5) -> dict:
        intent = self.interpret_query(query)
        top_jobs, all_ranked_jobs = self.search_jobs(intent, limit=top_job_limit)
        top_modules, stem_status = self.recommend_modules(all_ranked_jobs, limit=top_module_limit)
        top_skill_profile = self._build_skill_profile(all_ranked_jobs, top_k_jobs=8, top_k_skills=10)

        return {
            "query": query,
            "interpreted_query": {
                "tokens": intent.tokens,
                "skills": sorted(intent.skills),
                "work_type": intent.work_type,
                "salary_min": intent.salary_min,
                "salary_max": intent.salary_max,
                "experience_max": intent.experience_max,
            },
            "summary": self._build_summary(intent, all_ranked_jobs, top_jobs, top_modules, stem_status),
            "stats": {
                "matched_job_count": len(all_ranked_jobs),
                "returned_job_count": len(top_jobs),
                "returned_module_count": len(top_modules),
                "top_skill_profile": [
                    {"skill": skill, "weight": round(weight, 3)}
                    for skill, weight in top_skill_profile
                ],
            },
            "stem_status": stem_status,
            "jobs": top_jobs,
            "modules": top_modules,
        }
