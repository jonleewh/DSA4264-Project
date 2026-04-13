import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = PROJECT_ROOT / "data" / "test"
DEFAULT_MODULE = TEST_DIR / "module_skills_canonical_stem.jsonl"
DEFAULT_JOB = TEST_DIR / "job_skills_canonical_stem.jsonl"
DEFAULT_OUTPUT = TEST_DIR / "module_job_alignment_STEM.jsonl"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalize_counter(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a).union(b)
    if not keys:
        return 0.0
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return num / den if den else 0.0


def topk_coverage(module_counter: Counter, job_counter: Counter, top_k: int) -> float:
    top_job = [s for s, _ in job_counter.most_common(top_k)]
    if not top_job:
        return 0.0
    module_skills = set(module_counter)
    covered = sum(1 for s in top_job if s in module_skills)
    return covered / len(top_job)


def gap_score(module_counter: Counter, job_counter: Counter, top_k: int) -> float:
    top_job = job_counter.most_common(top_k)
    if not top_job:
        return 0.0
    total_weight = sum(w for _, w in top_job)
    if total_weight <= 0:
        return 0.0
    module_skills = set(module_counter)
    missing_weight = sum(w for s, w in top_job if s not in module_skills)
    return missing_weight / total_weight


def alignment_score(coverage: float, wj: float, cos: float, gap: float) -> float:
    score = 0.4 * coverage + 0.25 * wj + 0.2 * cos + 0.15 * (1.0 - gap)
    return max(0.0, min(1.0, score))


def build_job_profiles(job_rows: list[dict], ssoc_level: str):
    code_key = f"ssoc_{ssoc_level}_code"
    title_key = f"ssoc_{ssoc_level}_title"
    buckets = defaultdict(list)
    titles = {}
    counts = Counter()

    for row in job_rows:
        code = row.get(code_key)
        if not code:
            continue
        skills = row.get("canonical_skills") or []
        skills = [s for s in skills if s]
        if not skills:
            continue
        buckets[code].extend(skills)
        titles[code] = row.get(title_key, "")
        counts[code] += 1

    profiles = {}
    for code, skills in buckets.items():
        counter = Counter(skills)
        profiles[code] = {
            "ssoc_code": code,
            "ssoc_title": titles.get(code, ""),
            "job_count": counts[code],
            "skill_counter": counter,
            "skill_weights": normalize_counter(counter),
        }
    return profiles


def score_module_against_profiles(module_skills: list[str], profiles: dict[str, dict], top_k_job_skills: int, top_n_matches: int):
    module_counter = Counter([s for s in module_skills if s])
    module_weights = normalize_counter(module_counter)
    matches = []

    for code, profile in profiles.items():
        job_counter = profile["skill_counter"]
        job_weights = profile["skill_weights"]
        coverage = topk_coverage(module_counter, job_counter, top_k_job_skills)
        wj = weighted_jaccard(module_weights, job_weights)
        cos = cosine_similarity(module_weights, job_weights)
        gap = gap_score(module_counter, job_counter, top_k_job_skills)
        score = alignment_score(coverage, wj, cos, gap)
        overlap = sorted(set(module_counter) & set(job_counter))
        matches.append(
            {
                "ssoc_code": code,
                "ssoc_title": profile["ssoc_title"],
                "job_count": profile["job_count"],
                "strict_overlap_count": len(overlap),
                "strict_overlap": overlap,
                "coverage_top_k": round(coverage, 4),
                "weighted_jaccard": round(wj, 4),
                "cosine_similarity": round(cos, 4),
                "gap_score": round(gap, 4),
                "alignment_score": round(score, 4),
            }
        )

    matches.sort(
        key=lambda item: (
            item["alignment_score"],
            item["strict_overlap_count"],
            item["job_count"],
        ),
        reverse=True,
    )
    return matches[:top_n_matches]


def main():
    parser = argparse.ArgumentParser(description="Align modules and jobs in canonical skill space.")
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--ssoc-level", choices=["3d", "4d", "5d"], default="3d")
    parser.add_argument("--top-k-job-skills", type=int, default=10)
    parser.add_argument("--top-n-matches", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    module_rows = load_jsonl(args.module)
    job_rows = load_jsonl(args.job)
    profiles = build_job_profiles(job_rows, ssoc_level=args.ssoc_level)

    results = []
    top1_overlap = 0
    empty_modules = 0

    for row in module_rows:
        skills = row.get("canonical_skills") or []
        if not skills:
            empty_modules += 1
        matches = score_module_against_profiles(
            module_skills=skills,
            profiles=profiles,
            top_k_job_skills=args.top_k_job_skills,
            top_n_matches=args.top_n_matches,
        )
        if matches and matches[0]["strict_overlap_count"] > 0:
            top1_overlap += 1

        results.append(
            {
                "module_id": row.get("id"),
                "source": row.get("source"),
                "title": row.get("title"),
                "canonical_skills": skills,
                "top_matches": matches,
            }
        )

    module_count = len(module_rows)
    effective_module_count = module_count - empty_modules
    non_empty_results = [r for r in results if r["canonical_skills"]]

    summary = {
        "module_count": module_count,
        "effective_module_count": effective_module_count,
        "top1_overlap_rate": round(top1_overlap / effective_module_count, 4)
        if effective_module_count
        else 0.0,
        "average_top1_score": round(
            sum(r["top_matches"][0]["alignment_score"] for r in non_empty_results if r["top_matches"])
            / effective_module_count,
            4,
        )
        if effective_module_count
        else 0.0,
    }

    payload = {
        "ssoc_level": args.ssoc_level,
        "summary": summary,
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved canonical alignment: {args.output}")


if __name__ == "__main__":
    main()
