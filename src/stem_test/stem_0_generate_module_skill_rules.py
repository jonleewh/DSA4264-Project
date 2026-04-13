import argparse
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "cleaned_module_rows_STEM.jsonl"
DEFAULT_RULES_FILE = PROJECT_ROOT / "src" / "stem_test" / "module_skill_rules.py"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reference" / "module_skill_rules_generated_report.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FRAMEWORK = PROJECT_ROOT / "data" / "reference" / "canonical_skill_framework_v4.json"


GENERIC_SUBSTRINGS = {
    "module",
    "course",
    "student",
    "assessment",
    "assignment",
    "semester",
    "lecture",
    "tutorial",
    "prerequisite",
    "credit",
}

GENERIC_ACTION_WORDS = {
    "analysis",
    "design",
    "engineering",
    "problem",
    "problems",
    "research",
    "ability",
    "aim",
    "aims",
    "apply",
    "applied",
    "approach",
    "approaches",
    "cover",
    "covered",
    "covers",
    "develop",
    "developed",
    "developing",
    "include",
    "includes",
    "including",
    "introduce",
    "introduces",
    "provide",
    "provides",
    "skill",
    "skills",
    "understand",
    "understanding",
    "use",
    "used",
    "using",
    "work",
    "working",
}

SHORT_TOKEN_ALLOWLIST = {"sql", "cad", "r", "c++", "c#"}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_phrase(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"^[^a-z0-9+#/]+|[^a-z0-9+#/]+$", "", text)
    return text


def titleize_skill(phrase: str) -> str:
    small = {"and", "or", "of", "for", "to", "in", "on", "with", "the", "a", "an"}
    parts = phrase.split()
    titled = []
    for i, p in enumerate(parts):
        if i > 0 and p in small:
            titled.append(p)
        else:
            titled.append(p.capitalize())
    return " ".join(titled)


def load_rows(path: Path, max_rows: Optional[int]) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def is_candidate_phrase(phrase: str, min_tokens: int, max_tokens: int) -> bool:
    if not phrase:
        return False
    tokens = phrase.split()
    if len(tokens) < min_tokens or len(tokens) > max_tokens:
        return False
    if len(phrase) < 3 or len(phrase) > 80:
        return False
    if all(t in ENGLISH_STOP_WORDS for t in tokens):
        return False
    if any(ch.isdigit() for ch in phrase):
        return False
    if any(sub in phrase for sub in GENERIC_SUBSTRINGS):
        return False
    if all(t in GENERIC_ACTION_WORDS for t in tokens):
        return False
    if tokens[0] in GENERIC_ACTION_WORDS:
        return False
    if len(tokens) == 1 and len(tokens[0]) <= 3 and tokens[0] not in SHORT_TOKEN_ALLOWLIST:
        return False
    return True


def extract_phrases(
    rows: list[dict],
    model_name: str,
    top_n: int,
    min_tokens: int,
    max_tokens: int,
):
    descriptions = [normalize_text(row.get("description") or "") for row in rows]
    descriptions = [d for d in descriptions if d]

    try:
        st_model = SentenceTransformer(model_name, local_files_only=True)
        print(f"Loaded model from cache: {model_name}")
    except Exception:
        st_model = SentenceTransformer(model_name)
        print(f"Downloaded model: {model_name}")

    kw_model = KeyBERT(model=st_model)

    phrase_freq = Counter()
    phrase_docfreq = Counter()
    for idx, description in enumerate(descriptions, start=1):
        ranked = kw_model.extract_keywords(
            description[:2500],
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,
            diversity=0.8,
        )
        doc_phrases = set()
        for phrase, _score in ranked:
            p = normalize_phrase(phrase)
            if not is_candidate_phrase(
                p,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            ):
                continue
            phrase_freq[p] += 1
            doc_phrases.add(p)
        for p in doc_phrases:
            phrase_docfreq[p] += 1

        if idx % 200 == 0 or idx == len(descriptions):
            print(f"KeyBERT processed {idx}/{len(descriptions)} descriptions...", flush=True)

    return phrase_freq, phrase_docfreq, st_model, len(descriptions)


def choose_seed_phrases(
    phrase_freq: Counter,
    phrase_docfreq: Counter,
    total_docs: int,
    min_support: int,
    max_doc_ratio: float,
) -> list[str]:
    phrases = []
    for p, freq in phrase_freq.items():
        if freq < min_support:
            continue
        if phrase_docfreq[p] / max(total_docs, 1) > max_doc_ratio:
            continue
        phrases.append(p)

    # Score: reward support + mildly reward specificity and multi-word phrases.
    phrases.sort(
        key=lambda p: (
            phrase_freq[p] * (1.0 + math.log1p(len(p.split()))),
            phrase_docfreq[p],
            p,
        ),
        reverse=True,
    )
    return phrases


def cluster_phrases(phrases: list[str], model: SentenceTransformer, distance_threshold: float):
    if not phrases:
        return {}

    emb = model.encode(phrases, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb)

    if len(phrases) == 1:
        return {0: [phrases[0]]}

    clustering = AgglomerativeClustering(
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = clustering.fit_predict(emb)

    grouped = defaultdict(list)
    for phrase, label in zip(phrases, labels.tolist()):
        grouped[int(label)].append(phrase)
    return dict(grouped)


def build_rules(
    grouped: dict[int, list[str]],
    phrase_freq: Counter,
    phrase_docfreq: Counter,
    min_patterns: int,
    max_patterns: int,
):
    rules = []
    details = []

    for _, members in grouped.items():
        members = sorted(
            members,
            key=lambda p: (phrase_freq[p], phrase_docfreq[p], len(p.split()), p),
            reverse=True,
        )
        if len(members) < min_patterns:
            continue

        canonical_source = members[0]
        canonical_skill = titleize_skill(canonical_source)
        patterns = members[:max_patterns]

        rules.append({"patterns": patterns, "skills": [canonical_skill]})
        details.append(
            {
                "canonical_skill": canonical_skill,
                "cluster_size": len(members),
                "patterns": patterns,
                "all_members": members,
                "support": sum(phrase_freq[p] for p in members),
                "top_pattern_support": phrase_freq[patterns[0]],
            }
        )

    rules.sort(
        key=lambda r: (len(r["patterns"]), phrase_freq[r["patterns"][0]], r["skills"][0]),
        reverse=True,
    )
    details.sort(key=lambda d: (d["support"], d["cluster_size"], d["canonical_skill"]), reverse=True)

    return rules, details


def group_phrases_by_framework(
    phrases: list[str],
    phrase_freq: Counter,
    phrase_docfreq: Counter,
    model: SentenceTransformer,
    framework_path: Path,
    map_threshold: float,
):
    framework = json.loads(framework_path.read_text(encoding="utf-8"))
    canonical_skills = [row["canonical_skill"] for row in framework.get("skills", []) if row.get("canonical_skill")]
    canonical_skills = list(dict.fromkeys(canonical_skills))
    if not canonical_skills:
        raise ValueError(f"No canonical skills found in framework: {framework_path}")

    phrase_emb = model.encode(phrases, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    skill_emb = model.encode(canonical_skills, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    sims = np.matmul(np.asarray(phrase_emb), np.asarray(skill_emb).T)

    grouped = defaultdict(list)
    for idx, phrase in enumerate(phrases):
        row = sims[idx]
        best_i = int(np.argmax(row))
        best_score = float(row[best_i])
        if best_score < map_threshold:
            continue
        grouped[canonical_skills[best_i]].append((phrase, best_score))

    rules = []
    details = []
    for skill, pairs in grouped.items():
        # Deduplicate phrase variants while preserving highest score/frequency first.
        seen = set()
        ordered = sorted(
            pairs,
            key=lambda x: (phrase_freq[x[0]], phrase_docfreq[x[0]], x[1], x[0]),
            reverse=True,
        )
        patterns = []
        for phrase, _score in ordered:
            if phrase in seen:
                continue
            seen.add(phrase)
            patterns.append(phrase)
        if not patterns:
            continue

        rules.append({"patterns": patterns, "skills": [skill]})
        details.append(
            {
                "canonical_skill": skill,
                "cluster_size": len(patterns),
                "patterns": patterns[:12],
                "all_members": patterns,
                "support": sum(phrase_freq[p] for p in patterns),
                "top_pattern_support": phrase_freq[patterns[0]],
            }
        )

    rules.sort(
        key=lambda r: (len(r["patterns"]), phrase_freq[r["patterns"][0]], r["skills"][0]),
        reverse=True,
    )
    details.sort(key=lambda d: (d["support"], d["cluster_size"], d["canonical_skill"]), reverse=True)
    return rules, details


def format_rules_literal(rules: list[dict]) -> str:
    lines = ["MODULE_SKILL_RULES = ["]
    for rule in rules:
        patterns = ", ".join(json.dumps(p, ensure_ascii=False) for p in rule["patterns"])
        skill = json.dumps(rule["skills"][0], ensure_ascii=False)
        lines.append(f"    {{\"patterns\": [{patterns}], \"skills\": [{skill}]}},")
    lines.append("]")
    return "\n".join(lines)


def format_canonical_literal(rules: list[dict]) -> str:
    skills = sorted({rule["skills"][0] for rule in rules})
    lines = ["CANONICAL_MODULE_SKILLS = ["]
    for s in skills:
        lines.append(f"    {json.dumps(s, ensure_ascii=False)},")
    lines.append("]")
    return "\n".join(lines)


def replace_block(text: str, start_marker: str, end_marker: str, replacement_block: str) -> str:
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"Could not find block from {start_marker} to {end_marker}")
    replacement = replacement_block + "\n\n" + end_marker
    return pattern.sub(replacement, text)


def write_rules_file(path: Path, rules: list[dict]):
    original = path.read_text(encoding="utf-8")
    rules_literal = format_rules_literal(rules)
    canonical_literal = format_canonical_literal(rules)

    updated = replace_block(
        original,
        "MODULE_SKILL_RULES = [",
        "PRACTICAL_ANCHORS = [",
        rules_literal,
    )
    updated = replace_block(
        updated,
        "CANONICAL_MODULE_SKILLS = [",
        "STRICT_CANONICAL_EVIDENCE = {",
        canonical_literal,
    )

    path.write_text(updated, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MODULE_SKILL_RULES from module descriptions using KeyBERT + semantic clustering."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--rules-file", type=Path, default=DEFAULT_RULES_FILE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--framework", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--min-phrase-tokens", type=int, default=1)
    parser.add_argument("--max-phrase-tokens", type=int, default=3)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--max-doc-ratio", type=float, default=0.12)
    parser.add_argument("--distance-threshold", type=float, default=0.32)
    parser.add_argument("--semantic-map-threshold", type=float, default=0.44)
    parser.add_argument("--min-patterns-per-rule", type=int, default=2)
    parser.add_argument("--max-patterns-per-rule", type=int, default=6)
    parser.add_argument("--max-rules", type=int, default=120)
    parser.add_argument(
        "--grouping-mode",
        choices=["cluster", "framework"],
        default="framework",
        help="cluster: unsupervised phrase clustering; framework: map phrases to canonical framework skills.",
    )
    parser.add_argument("--write-rules", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    rows = load_rows(args.input, args.max_rows)
    print(f"Loaded {len(rows)} module rows from {args.input}")

    phrase_freq, phrase_docfreq, st_model, total_docs = extract_phrases(
        rows=rows,
        model_name=args.model,
        top_n=args.top_n,
        min_tokens=args.min_phrase_tokens,
        max_tokens=args.max_phrase_tokens,
    )

    seeds = choose_seed_phrases(
        phrase_freq=phrase_freq,
        phrase_docfreq=phrase_docfreq,
        total_docs=total_docs,
        min_support=args.min_support,
        max_doc_ratio=args.max_doc_ratio,
    )

    grouped = {}
    if args.grouping_mode == "cluster":
        grouped = cluster_phrases(
            phrases=seeds,
            model=st_model,
            distance_threshold=args.distance_threshold,
        )
        rules, details = build_rules(
            grouped=grouped,
            phrase_freq=phrase_freq,
            phrase_docfreq=phrase_docfreq,
            min_patterns=args.min_patterns_per_rule,
            max_patterns=args.max_patterns_per_rule,
        )
    else:
        if not args.framework.exists():
            raise FileNotFoundError(f"Framework not found: {args.framework}")
        rules, details = group_phrases_by_framework(
            phrases=seeds,
            phrase_freq=phrase_freq,
            phrase_docfreq=phrase_docfreq,
            model=st_model,
            framework_path=args.framework,
            map_threshold=args.semantic_map_threshold,
        )
        # Keep only rules with enough patterns, then cap top patterns.
        filtered_rules = []
        for rule in rules:
            if len(rule["patterns"]) < args.min_patterns_per_rule:
                continue
            filtered_rules.append(
                {
                    "patterns": rule["patterns"][: args.max_patterns_per_rule],
                    "skills": rule["skills"],
                }
            )
        rules = filtered_rules
        allowed = {r["skills"][0] for r in rules}
        details = [d for d in details if d["canonical_skill"] in allowed]

    if args.max_rules > 0:
        rules = rules[: args.max_rules]
        keep = {r["skills"][0] for r in rules}
        details = [d for d in details if d["canonical_skill"] in keep]

    report_payload = {
        "input": str(args.input),
        "total_rows": len(rows),
        "unique_candidate_phrases": len(phrase_freq),
        "seed_phrase_count": len(seeds),
        "cluster_count": len(grouped),
        "grouping_mode": args.grouping_mode,
        "generated_rule_count": len(rules),
        "generated_canonical_skill_count": len({r["skills"][0] for r in rules}),
        "params": {
            "top_n": args.top_n,
            "min_phrase_tokens": args.min_phrase_tokens,
            "max_phrase_tokens": args.max_phrase_tokens,
            "min_support": args.min_support,
            "max_doc_ratio": args.max_doc_ratio,
            "distance_threshold": args.distance_threshold,
            "semantic_map_threshold": args.semantic_map_threshold,
            "min_patterns_per_rule": args.min_patterns_per_rule,
            "max_patterns_per_rule": args.max_patterns_per_rule,
            "max_rules": args.max_rules,
        },
        "rules": details,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote generation report: {args.report}")

    if args.write_rules:
        write_rules_file(args.rules_file, rules)
        print(f"Updated rules file: {args.rules_file}")
    else:
        print("Dry run complete (no rules file updated). Use --write-rules to apply.")

    print(f"Generated {len(rules)} rules from {len(rows)} module rows.")


if __name__ == "__main__":
    main()
    "design",
    "engineering",
    "problem",
    "problems",
    "research",
