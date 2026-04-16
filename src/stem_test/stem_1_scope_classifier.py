import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DEFAULT_COURSES_INPUT = PROJECT_ROOT / "data" / "cleaned_data" / "combined_courses_cleaned.pkl"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_STEM_OUTPUT = PROJECT_ROOT / "data" / "cleaned_data" / "cleaned_module_rows_STEM.jsonl"
DEFAULT_NON_STEM_OUTPUT = PROJECT_ROOT / "data" / "cleaned_data" / "cleaned_module_rows_non_STEM.jsonl"

NUS_FILE = REFERENCE_DIR / "nus_stem_classification_v1.json"
NTU_FILE = REFERENCE_DIR / "ntu_stem_classification_v1.json"
SUTD_FILE = REFERENCE_DIR / "sutd_stem_classification_v1.json"


def _norm(value: str | None) -> str:
    return (value or "").strip()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Classification file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _to_set(values: list[str] | None) -> set[str]:
    return {_norm(v) for v in (values or []) if _norm(v)}


STEM_STRONG_PATTERNS = [
    r"\bmathematics?\b",
    r"\bmath\b",
    r"\bscience\b",
    r"\bscientific\b",
    r"\bengineering\b",
    r"\btechnology\b",
    r"\bcoding\b",
    r"\bcode\b",
    r"\bprogramming\b",
    r"\bcomputer\b",
    r"\bcomputation(al)?\b",
    r"\balgorithm(s|ic)?\b",
    r"\bdata structures?\b",
    r"\bsoftware\b",
    r"\bpython\b",
    r"\br programming\b|\bprogramming in r\b|\blanguage r\b",
    r"\bsql\b",
    r"\bstatistics?\b",
    r"\bstatistical\b",
    r"\beconometric(s)?\b",
    r"\bregression\b",
    r"\bprobability\b",
    r"\bstochastic\b",
    r"\btime series\b",
    r"\boptimization\b|\boptimisation\b",
    r"\boperations research\b",
    r"\blinear algebra\b",
    r"\bcalculus\b",
    r"\bnumerical (method|methods)\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bneural network(s)?\b",
    r"\bdata mining\b",
    r"\bforecast(ing)?\b",
    r"\bsimulation\b",
    r"\bphysics?\b",
    r"\bchemistry\b",
    r"\bmolecular biology\b",
    r"\bcell biology\b",
    r"\bgenetics?\b",
    r"\bbiotechnology\b",
    r"\bbiomedical\b",
    r"\bbioinformatics\b",
    r"\banatomy\b",
    r"\bphysiology\b",
    r"\bbiochemistry\b",
    r"\bmicrobiology\b",
    r"\bimmunology\b",
]

STEM_WEAK_PATTERNS = [
    r"\bquantitative\b",
    r"\banalytics?\b",
    r"\bmathematical model(l)?ing\b",
    r"\bempirical\b",
    r"\bdata analysis\b",
]

QUANT_BLOCKLIST_PATTERNS = [
    r"\bliterary analysis\b",
    r"\bfilm analysis\b",
    r"\bcritical analysis\b",
    r"\btextual analysis\b",
    r"\bhistorical analysis\b",
    r"\bphilosophical analysis\b",
]

NON_STEM_CONTEXT_PATTERNS = [
    r"\bsociolog(y|ical)\b",
    r"\banthropolog(y|ical)\b",
    r"\bcultural\b",
    r"\bethnograph(y|ic)\b",
    r"\bhistorical\b",
    r"\bhistory\b",
    r"\bphilosoph(y|ical)\b",
    r"\bliterature\b",
    r"\bart(s|istic)?\b",
    r"\bmortuary\b",
    r"\bfunerar(y|ies)\b",
    r"\britual(s)?\b",
]

STEM_PROTOTYPE_SENTENCES = [
    "This module teaches programming, algorithms, and computational problem solving.",
    "Students build predictive models using statistics, machine learning, and data analysis.",
    "The course covers experimental design, scientific methods, and quantitative reasoning.",
    "Learners apply linear algebra, calculus, and optimization in engineering systems.",
    "This class studies molecular biology, genetics, and biochemical laboratory techniques.",
    "The module includes software engineering, databases, and computer systems.",
    "Students write code to simulate physical systems and evaluate model accuracy.",
    "The curriculum focuses on probability, inference, and regression for real datasets.",
    "Laboratory sessions train students to collect, measure, and analyze scientific observations.",
    "The course develops numerical methods for solving differential equations in engineering.",
    # Ambiguous STEM-leaning examples
    "Students discuss ethics in AI while implementing and testing machine learning pipelines.",
    "The module examines technology policy through quantitative analysis of digital platform data.",
]

NON_STEM_PROTOTYPE_SENTENCES = [
    "This module examines culture, history, and social theory through critical interpretation.",
    "Students discuss philosophy, ethics, and political thought using textual analysis.",
    "The course focuses on literature, narrative methods, and close reading.",
    "Learners study anthropological perspectives, ethnography, and social practices.",
    "This class explores art history, media criticism, and cultural representation.",
    "Students analyze historical documents to compare arguments across time periods.",
    "The seminar evaluates political ideas through conceptual reasoning and debate.",
    "The course interprets novels and films through critical and theoretical frameworks.",
    "Learners conduct qualitative inquiry on identity, society, and cultural meaning.",
    "The module studies religion and philosophy using interpretive reading and discussion.",
    # Ambiguous non-STEM-leaning examples
    "The class discusses the social impact of technology through critical essays and theory.",
    "Students examine scientific modernity in literature and philosophy rather than technical methods.",
]


def _split_sentences(text: str | None) -> list[str]:
    raw = _norm(text)
    if not raw:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", raw)
    return [p.strip() for p in parts if p and p.strip()]


def _safe_l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


class SentenceSemanticStemScorer:
    """Sentence-level semantic scorer that contrasts STEM vs non-STEM meaning."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        margin: float = 0.04,
    ):
        self.model_name = model_name
        self.margin = margin
        self._model = None
        self._stem_centroid: np.ndarray | None = None
        self._non_stem_centroid: np.ndarray | None = None
        self.available = False
        self._sentence_cache: dict[tuple[str, str], dict[str, float | int]] = {}
        self._document_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._initialize()

    def _initialize(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            self.available = False
            return

        try:
            print(
                f"[STEM scope] Loading semantic model from local cache: {self.model_name}",
                flush=True,
            )
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            try:
                print(
                    f"[STEM scope] Local cache unavailable, loading semantic model: {self.model_name}",
                    flush=True,
                )
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self.available = False
                return

        stem_vectors = self._model.encode(
            STEM_PROTOTYPE_SENTENCES,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        non_stem_vectors = self._model.encode(
            NON_STEM_PROTOTYPE_SENTENCES,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._stem_centroid = _safe_l2_normalize(np.mean(stem_vectors, axis=0, keepdims=True))[0]
        self._non_stem_centroid = _safe_l2_normalize(np.mean(non_stem_vectors, axis=0, keepdims=True))[0]
        self.available = True
        print("[STEM scope] Semantic model ready.", flush=True)

    def score_sentences(self, title: str | None, description: str | None) -> dict[str, float | int]:
        cache_key = (_norm(title), _norm(description))
        cached = self._sentence_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        if not self.available or self._model is None or self._stem_centroid is None or self._non_stem_centroid is None:
            result = {
                "support_count": 0,
                "oppose_count": 0,
                "total_sentences": 0,
                "avg_margin": 0.0,
                "max_stem_similarity": 0.0,
                "confidence": 0.0,
            }
            self._sentence_cache[cache_key] = dict(result)
            return result

        sentences = _split_sentences(title) + _split_sentences(description)
        if not sentences:
            result = {
                "support_count": 0,
                "oppose_count": 0,
                "total_sentences": 0,
                "avg_margin": 0.0,
                "max_stem_similarity": 0.0,
                "confidence": 0.0,
            }
            self._sentence_cache[cache_key] = dict(result)
            return result

        vectors = self._model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        stem_scores = np.dot(vectors, self._stem_centroid)
        non_stem_scores = np.dot(vectors, self._non_stem_centroid)
        margins = stem_scores - non_stem_scores

        support_mask = margins >= self.margin
        oppose_mask = margins <= -self.margin
        support_count = int(np.sum(support_mask))
        oppose_count = int(np.sum(oppose_mask))
        confidence = support_count / (support_count + oppose_count + 1e-9)

        result = {
            "support_count": support_count,
            "oppose_count": oppose_count,
            "total_sentences": int(len(sentences)),
            "avg_margin": float(np.mean(margins)) if len(margins) else 0.0,
            "max_stem_similarity": float(np.max(stem_scores)) if len(stem_scores) else 0.0,
            "confidence": float(confidence),
        }
        self._sentence_cache[cache_key] = dict(result)
        return result

    def score_document(self, title: str | None, description: str | None) -> dict[str, float]:
        cache_key = (_norm(title), _norm(description))
        cached = self._document_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        if not self.available or self._model is None or self._stem_centroid is None or self._non_stem_centroid is None:
            result = {
                "stem_similarity": 0.0,
                "non_stem_similarity": 0.0,
                "margin": 0.0,
            }
            self._document_cache[cache_key] = dict(result)
            return result

        text = " ".join([_norm(title), _norm(description)]).strip()
        if not text:
            result = {
                "stem_similarity": 0.0,
                "non_stem_similarity": 0.0,
                "margin": 0.0,
            }
            self._document_cache[cache_key] = dict(result)
            return result

        vector = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        stem_similarity = float(np.dot(vector, self._stem_centroid))
        non_stem_similarity = float(np.dot(vector, self._non_stem_centroid))
        result = {
            "stem_similarity": stem_similarity,
            "non_stem_similarity": non_stem_similarity,
            "margin": stem_similarity - non_stem_similarity,
        }
        self._document_cache[cache_key] = dict(result)
        return result


def _stem_signal_score(title: str | None, description: str | None) -> int:
    text = f"{_norm(title).lower()} {_norm(description).lower()}"
    if not text.strip():
        return 0
    if any(re.search(pat, text) for pat in QUANT_BLOCKLIST_PATTERNS):
        return 0

    score = 0
    for pat in STEM_STRONG_PATTERNS:
        if re.search(pat, text):
            score += 2
    for pat in STEM_WEAK_PATTERNS:
        if re.search(pat, text):
            score += 1
    return score


def _strong_pattern_hits(title: str | None, description: str | None) -> int:
    text = f"{_norm(title).lower()} {_norm(description).lower()}"
    if not text.strip():
        return 0
    return sum(1 for pat in STEM_STRONG_PATTERNS if re.search(pat, text))


def _has_non_stem_context(title: str | None, description: str | None) -> bool:
    text = f"{_norm(title).lower()} {_norm(description).lower()}"
    if not text.strip():
        return False
    return any(re.search(pat, text) for pat in NON_STEM_CONTEXT_PATTERNS)


def is_stem_semantic_module(title: str | None, description: str | None, min_score: int = 2) -> bool:
    """Keyword fallback only (semantic sentence encoder handled separately)."""
    score = _stem_signal_score(title, description)
    strong_hits = _strong_pattern_hits(title, description)
    if score < min_score:
        return False
    if strong_hits <= 1 and _has_non_stem_context(title, description):
        return False
    return True


class StemScopeClassifier:
    """Shared classifier for NUS/NTU/SUTD scope bucketing.

    Buckets:
    - clear_stem
    - unclear_or_mixed
    """

    def __init__(
        self,
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sentence_vote_diff_threshold: int = 1,
        semantic_margin: float = 0.04,
        paragraph_stem_margin: float = 0.06,
        paragraph_non_stem_margin: float = 0.02,
        enable_semantic_encoder: bool = True,
    ):
        self.nus = _load_json(NUS_FILE)
        self.ntu = _load_json(NTU_FILE)
        self.sutd = _load_json(SUTD_FILE)

        self.sentence_vote_diff_threshold = sentence_vote_diff_threshold
        self.paragraph_stem_margin = paragraph_stem_margin
        self.paragraph_non_stem_margin = paragraph_non_stem_margin
        self.semantic_encoder = (
            SentenceSemanticStemScorer(
                model_name=semantic_model,
                margin=semantic_margin,
            )
            if enable_semantic_encoder
            else None
        )

        self.nus_fac = {
            "clear_stem": _to_set(self.nus.get("faculties", {}).get("clear_stem")),
            "clear_non_stem": _to_set(self.nus.get("faculties", {}).get("clear_non_stem")),
            "unclear_or_mixed": _to_set(self.nus.get("faculties", {}).get("unclear_or_mixed")),
        }
        self.nus_dept = {
            "clear_stem": _to_set(self.nus.get("departments", {}).get("clear_stem")),
            "clear_non_stem": _to_set(self.nus.get("departments", {}).get("clear_non_stem")),
            "unclear_or_mixed": _to_set(self.nus.get("departments", {}).get("unclear_or_mixed")),
        }

        self.ntu_dept = {
            "clear_stem": _to_set(self.ntu.get("departments", {}).get("clear_stem")),
            "clear_non_stem": _to_set(self.ntu.get("departments", {}).get("clear_non_stem")),
            "unclear_or_mixed": _to_set(self.ntu.get("departments", {}).get("unclear_or_mixed")),
        }

        self.sutd_dept = {
            "clear_stem": _to_set(self.sutd.get("departments", {}).get("clear_stem")),
            "clear_non_stem": _to_set(self.sutd.get("departments", {}).get("clear_non_stem")),
            "unclear_or_mixed": _to_set(self.sutd.get("departments", {}).get("unclear_or_mixed")),
        }

    def classify_module_scope(
        self,
        source: str | None,
        department: str | None = None,
        faculty: str | None = None,
        title: str | None = None,
        description: str | None = None,
        quant_min_score: int = 2,
    ) -> dict[str, str]:
        source_n = _norm(source).upper()
        dept_n = _norm(department)
        fac_n = _norm(faculty)

        base_bucket = "unclear_or_mixed"
        base_reason = "unknown_source"

        if source_n == "NUS":
            # Priority: explicit STEM match > explicit non-STEM match > unclear.
            if dept_n in self.nus_dept["clear_stem"] or fac_n in self.nus_fac["clear_stem"]:
                base_bucket, base_reason = "clear_stem", "nus_department_or_faculty_clear_stem"
            elif dept_n in self.nus_dept["clear_non_stem"] or fac_n in self.nus_fac["clear_non_stem"]:
                base_bucket, base_reason = "clear_non_stem", "nus_department_or_faculty_clear_non_stem"
            else:
                base_bucket, base_reason = "unclear_or_mixed", "nus_unlisted_or_mixed"

        elif source_n == "NTU":
            if dept_n in self.ntu_dept["clear_stem"]:
                base_bucket, base_reason = "clear_stem", "ntu_department_clear_stem"
            elif dept_n in self.ntu_dept["clear_non_stem"]:
                base_bucket, base_reason = "clear_non_stem", "ntu_department_clear_non_stem"
            else:
                base_bucket, base_reason = "unclear_or_mixed", "ntu_unlisted_or_mixed"

        elif source_n == "SUTD":
            if not dept_n:
                base_bucket, base_reason = "unclear_or_mixed", "sutd_department_missing"
            elif dept_n in self.sutd_dept["clear_stem"]:
                base_bucket, base_reason = "clear_stem", "sutd_department_clear_stem"
            elif dept_n in self.sutd_dept["clear_non_stem"]:
                base_bucket, base_reason = "clear_non_stem", "sutd_department_clear_non_stem"
            else:
                base_bucket, base_reason = "unclear_or_mixed", "sutd_unlisted_or_mixed"

        if base_bucket != "clear_stem":
            if self.semantic_encoder is not None and self.semantic_encoder.available:
                document_semantic = self.semantic_encoder.score_document(title, description)
                if document_semantic["margin"] <= -self.paragraph_non_stem_margin:
                    return {
                        "scope_bucket": base_bucket,
                        "scope_reason": "paragraph_semantic_non_stem_guard",
                    }
                if document_semantic["margin"] >= self.paragraph_stem_margin:
                    return {
                        "scope_bucket": "clear_stem",
                        "scope_reason": "paragraph_semantic_stem_override",
                    }

                semantic_details = self.semantic_encoder.score_sentences(title, description)
                if (
                    document_semantic["margin"] >= 0.0
                    and (semantic_details["support_count"] - semantic_details["oppose_count"])
                    >= self.sentence_vote_diff_threshold
                ):
                    return {
                        "scope_bucket": "clear_stem",
                        "scope_reason": "stem_semantic_sentence_override",
                    }

            if is_stem_semantic_module(title, description, min_score=quant_min_score):
                return {"scope_bucket": "clear_stem", "scope_reason": "stem_keyword_override"}

        return {"scope_bucket": base_bucket, "scope_reason": base_reason}


def load_stem_scope_classifier(
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sentence_vote_diff_threshold: int = 1,
    semantic_margin: float = 0.04,
    paragraph_stem_margin: float = 0.06,
    paragraph_non_stem_margin: float = 0.02,
    enable_semantic_encoder: bool = True,
) -> StemScopeClassifier:
    return StemScopeClassifier(
        semantic_model=semantic_model,
        sentence_vote_diff_threshold=sentence_vote_diff_threshold,
        semantic_margin=semantic_margin,
        paragraph_stem_margin=paragraph_stem_margin,
        paragraph_non_stem_margin=paragraph_non_stem_margin,
        enable_semantic_encoder=enable_semantic_encoder,
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported JSON shape in {path}. Expected a top-level list or JSONL rows.")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_courses_from_pkl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned courses input not found: {path}")

    df = pd.read_pickle(path)
    required_cols = {"code", "title", "description", "department", "university"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required course columns in {path}: {missing}"
        )

    rows: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        code = _norm(record.get("code"))
        university = _norm(record.get("university"))
        description = _norm(record.get("description"))
        if not code or not university or not description:
            continue

        rows.append(
            {
                "id": f"{university}::{code}",
                "source": university,
                "code": code,
                "title": _norm(record.get("title")),
                "department": _norm(record.get("department")),
                "university": university,
                "description": description,
                "skills": record.get("skills_embedding") if isinstance(record.get("skills_embedding"), list) else [],
                "hard_skills": record.get("hard_skills") if isinstance(record.get("hard_skills"), list) else [],
                "soft_skills": record.get("soft_skills") if isinstance(record.get("soft_skills"), list) else [],
                "num_skills": record.get("num_skills"),
            }
        )

    return rows


def build_module_meta_lookup(processed_dir: Path):
    lookup = {}

    nus_path = processed_dir / "nus_cleaned.json"
    if nus_path.exists():
        for record in _load_rows(nus_path):
            code = _norm(record.get("moduleCode"))
            if not code:
                continue
            lookup[f"NUS::{code}"] = {
                "department": record.get("department"),
                "faculty": record.get("faculty"),
            }

    ntu_path = processed_dir / "ntu_cleaned.json"
    if ntu_path.exists():
        for record in _load_rows(ntu_path):
            code = _norm(record.get("code"))
            if not code:
                continue
            lookup[f"NTU::{code}"] = {
                "department": record.get("department"),
                "faculty": record.get("faculty"),
            }

    sutd_path = processed_dir / "sutd_cleaned.json"
    if sutd_path.exists():
        for record in _load_rows(sutd_path):
            code = _norm(record.get("code"))
            if not code:
                continue
            lookup[f"SUTD::{code}"] = {
                "department": record.get("department"),
                "faculty": record.get("faculty"),
            }

    return lookup


def build_stem_rows(
    rows: list[dict[str, Any]],
    processed_dir: Path,
    quant_min_score: int = 2,
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sentence_vote_diff_threshold: int = 1,
    semantic_margin: float = 0.04,
    paragraph_stem_margin: float = 0.06,
    paragraph_non_stem_margin: float = 0.02,
    disable_semantic_encoder: bool = False,
):
    clf = load_stem_scope_classifier(
        semantic_model=semantic_model,
        sentence_vote_diff_threshold=sentence_vote_diff_threshold,
        semantic_margin=semantic_margin,
        paragraph_stem_margin=paragraph_stem_margin,
        paragraph_non_stem_margin=paragraph_non_stem_margin,
        enable_semantic_encoder=not disable_semantic_encoder,
    )
    module_meta = build_module_meta_lookup(processed_dir)

    stem_rows: list[dict[str, Any]] = []
    for row in rows:
        meta = module_meta.get(_norm(row.get("id")), {})
        title = row.get("title")
        description = row.get("description")
        out = clf.classify_module_scope(
            source=row.get("source"),
            department=meta.get("department") or row.get("department"),
            faculty=meta.get("faculty") or row.get("faculty"),
            title=title,
            description=description,
            quant_min_score=quant_min_score,
        )
        if out["scope_bucket"] == "clear_stem":
            semantic_available = bool(clf.semantic_encoder is not None and clf.semantic_encoder.available)
            if semantic_available:
                document_semantic = clf.semantic_encoder.score_document(title, description)
                sentence_semantic = clf.semantic_encoder.score_sentences(title, description)
            else:
                document_semantic = {
                    "stem_similarity": 0.0,
                    "non_stem_similarity": 0.0,
                    "margin": 0.0,
                }
                sentence_semantic = {
                    "support_count": 0,
                    "oppose_count": 0,
                    "total_sentences": 0,
                    "avg_margin": 0.0,
                    "max_stem_similarity": 0.0,
                    "confidence": 0.0,
                }

            enriched = {
                **row,
                "scope_bucket": out["scope_bucket"],
                "scope_reason": out["scope_reason"],
                "stem_semantic_metrics": {
                    "semantic_model_available": semantic_available,
                    "document": document_semantic,
                    "sentences": sentence_semantic,
                    "thresholds": {
                        "sentence_vote_diff_threshold": sentence_vote_diff_threshold,
                        "sentence_margin_threshold": semantic_margin,
                        "paragraph_stem_margin": paragraph_stem_margin,
                        "paragraph_non_stem_margin": paragraph_non_stem_margin,
                    },
                },
            }
            stem_rows.append(enriched)
    return stem_rows


def build_non_stem_rows(
    rows: list[dict[str, Any]],
    processed_dir: Path,
    quant_min_score: int = 2,
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sentence_vote_diff_threshold: int = 1,
    semantic_margin: float = 0.04,
    paragraph_stem_margin: float = 0.06,
    paragraph_non_stem_margin: float = 0.02,
    disable_semantic_encoder: bool = False,
):
    clf = load_stem_scope_classifier(
        semantic_model=semantic_model,
        sentence_vote_diff_threshold=sentence_vote_diff_threshold,
        semantic_margin=semantic_margin,
        paragraph_stem_margin=paragraph_stem_margin,
        paragraph_non_stem_margin=paragraph_non_stem_margin,
        enable_semantic_encoder=not disable_semantic_encoder,
    )
    module_meta = build_module_meta_lookup(processed_dir)

    non_stem_rows: list[dict[str, Any]] = []
    for row in rows:
        meta = module_meta.get(_norm(row.get("id")), {})
        title = row.get("title")
        description = row.get("description")
        out = clf.classify_module_scope(
            source=row.get("source"),
            department=meta.get("department") or row.get("department"),
            faculty=meta.get("faculty") or row.get("faculty"),
            title=title,
            description=description,
            quant_min_score=quant_min_score,
        )
        if out["scope_bucket"] != "clear_stem":
            semantic_available = bool(clf.semantic_encoder is not None and clf.semantic_encoder.available)
            if semantic_available:
                document_semantic = clf.semantic_encoder.score_document(title, description)
                sentence_semantic = clf.semantic_encoder.score_sentences(title, description)
            else:
                document_semantic = {
                    "stem_similarity": 0.0,
                    "non_stem_similarity": 0.0,
                    "margin": 0.0,
                }
                sentence_semantic = {
                    "support_count": 0,
                    "oppose_count": 0,
                    "total_sentences": 0,
                    "avg_margin": 0.0,
                    "max_stem_similarity": 0.0,
                    "confidence": 0.0,
                }

            enriched = {
                **row,
                "scope_bucket": out["scope_bucket"],
                "scope_reason": out["scope_reason"],
                "stem_semantic_metrics": {
                    "semantic_model_available": semantic_available,
                    "document": document_semantic,
                    "sentences": sentence_semantic,
                    "thresholds": {
                        "sentence_vote_diff_threshold": sentence_vote_diff_threshold,
                        "sentence_margin_threshold": semantic_margin,
                        "paragraph_stem_margin": paragraph_stem_margin,
                        "paragraph_non_stem_margin": paragraph_non_stem_margin,
                    },
                },
            }
            non_stem_rows.append(enriched)
    return non_stem_rows


def build_scoped_rows_with_metrics(
    rows: list[dict[str, Any]],
    processed_dir: Path,
    quant_min_score: int = 2,
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sentence_vote_diff_threshold: int = 1,
    semantic_margin: float = 0.04,
    paragraph_stem_margin: float = 0.06,
    paragraph_non_stem_margin: float = 0.02,
    disable_semantic_encoder: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("[STEM scope] Initializing scope classifier...", flush=True)
    clf = load_stem_scope_classifier(
        semantic_model=semantic_model,
        sentence_vote_diff_threshold=sentence_vote_diff_threshold,
        semantic_margin=semantic_margin,
        paragraph_stem_margin=paragraph_stem_margin,
        paragraph_non_stem_margin=paragraph_non_stem_margin,
        enable_semantic_encoder=not disable_semantic_encoder,
    )
    module_meta = build_module_meta_lookup(processed_dir)

    stem_rows: list[dict[str, Any]] = []
    non_stem_rows: list[dict[str, Any]] = []
    total = len(rows)
    print(f"[STEM scope] Classifying {total} module rows...", flush=True)
    for index, row in enumerate(rows, start=1):
        meta = module_meta.get(_norm(row.get("id")), {})
        title = row.get("title")
        description = row.get("description")
        out = clf.classify_module_scope(
            source=row.get("source"),
            department=meta.get("department") or row.get("department"),
            faculty=meta.get("faculty") or row.get("faculty"),
            title=title,
            description=description,
            quant_min_score=quant_min_score,
        )

        semantic_available = bool(clf.semantic_encoder is not None and clf.semantic_encoder.available)
        if semantic_available:
            document_semantic = clf.semantic_encoder.score_document(title, description)
            sentence_semantic = clf.semantic_encoder.score_sentences(title, description)
        else:
            document_semantic = {
                "stem_similarity": 0.0,
                "non_stem_similarity": 0.0,
                "margin": 0.0,
            }
            sentence_semantic = {
                "support_count": 0,
                "oppose_count": 0,
                "total_sentences": 0,
                "avg_margin": 0.0,
                "max_stem_similarity": 0.0,
                "confidence": 0.0,
            }

        enriched = {
            **row,
            "scope_bucket": out["scope_bucket"],
            "scope_reason": out["scope_reason"],
            "stem_semantic_metrics": {
                "semantic_model_available": semantic_available,
                "document": document_semantic,
                "sentences": sentence_semantic,
                "thresholds": {
                    "sentence_vote_diff_threshold": sentence_vote_diff_threshold,
                    "sentence_margin_threshold": semantic_margin,
                    "paragraph_stem_margin": paragraph_stem_margin,
                    "paragraph_non_stem_margin": paragraph_non_stem_margin,
                },
            },
        }

        if out["scope_bucket"] == "clear_stem":
            stem_rows.append(enriched)
        else:
            non_stem_rows.append(enriched)

        if index % 250 == 0 or index == total:
            print(
                f"[STEM scope] Processed {index}/{total} rows "
                f"(STEM={len(stem_rows)}, non-STEM={len(non_stem_rows)})",
                flush=True,
            )

    return stem_rows, non_stem_rows


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Classify module rows into STEM scope buckets using shared classification JSON files.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input module file (.json list or .jsonl rows).")
    parser.add_argument("--source", type=str, default=None, choices=["NUS", "NTU", "SUTD"])
    parser.add_argument("--department-key", type=str, default="department")
    parser.add_argument("--faculty-key", type=str, default="faculty")
    parser.add_argument("--summary", action="store_true", help="Print bucket/reason counts.")
    parser.add_argument("--show-samples", type=int, default=0, help="Show N sample classified rows.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for classified rows (.jsonl).")
    parser.add_argument(
        "--quant-min-score",
        type=int,
        default=2,
        help="Minimum keyword STEM score for fallback override.",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model for sentence-level STEM semantics.",
    )
    parser.add_argument(
        "--sentence-vote-diff-threshold",
        type=int,
        default=1,
        help="Minimum (support_count - oppose_count) for sentence-level STEM override in the gray zone.",
    )
    parser.add_argument(
        "--semantic-margin",
        type=float,
        default=0.04,
        help="Minimum (stem_sim - non_stem_sim) margin for a sentence to count as STEM support.",
    )
    parser.add_argument(
        "--paragraph-stem-margin",
        type=float,
        default=0.06,
        help="Minimum paragraph-level (stem_sim - non_stem_sim) for immediate STEM override.",
    )
    parser.add_argument(
        "--paragraph-non-stem-margin",
        type=float,
        default=0.02,
        help="Minimum paragraph-level (non_stem_sim - stem_sim) to veto STEM overrides.",
    )
    parser.add_argument(
        "--disable-semantic-encoder",
        action="store_true",
        help="Disable sentence-level semantic encoder and use keyword fallback only.",
    )
    parser.add_argument(
        "--build-stem-rows",
        action="store_true",
        help="Build STEM-only rows from combined_courses_cleaned.pkl and save data/cleaned_data/cleaned_module_rows_STEM.jsonl.",
    )
    parser.add_argument("--courses-input", type=Path, default=DEFAULT_COURSES_INPUT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--stem-output", type=Path, default=DEFAULT_STEM_OUTPUT)
    parser.add_argument("--non-stem-output", type=Path, default=DEFAULT_NON_STEM_OUTPUT)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_build_mode = args.build_stem_rows or (args.input is None and args.source is None)
    if run_build_mode:
        rows = load_courses_from_pkl(args.courses_input)
        stem_rows, non_stem_rows = build_scoped_rows_with_metrics(
            rows,
            processed_dir=args.processed_dir,
            quant_min_score=args.quant_min_score,
            semantic_model=args.semantic_model,
            sentence_vote_diff_threshold=args.sentence_vote_diff_threshold,
            semantic_margin=args.semantic_margin,
            paragraph_stem_margin=args.paragraph_stem_margin,
            paragraph_non_stem_margin=args.paragraph_non_stem_margin,
            disable_semantic_encoder=args.disable_semantic_encoder,
        )
        _write_jsonl(args.stem_output, stem_rows)
        _write_jsonl(args.non_stem_output, non_stem_rows)
        print(f"Input rows: {len(rows)}")
        print(f"Saved STEM cleaned module rows: {len(stem_rows)} -> {args.stem_output}")
        print(f"Saved non-STEM cleaned module rows: {len(non_stem_rows)} -> {args.non_stem_output}")
        return

    if args.input is None or args.source is None:
        raise SystemExit("Provide both --input and --source, or use --build-stem-rows.")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = _load_rows(args.input)
    clf = load_stem_scope_classifier(
        semantic_model=args.semantic_model,
        sentence_vote_diff_threshold=args.sentence_vote_diff_threshold,
        semantic_margin=args.semantic_margin,
        paragraph_stem_margin=args.paragraph_stem_margin,
        paragraph_non_stem_margin=args.paragraph_non_stem_margin,
        enable_semantic_encoder=not args.disable_semantic_encoder,
    )

    classified = []
    bucket_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in rows:
        out = clf.classify_module_scope(
            source=args.source,
            department=row.get(args.department_key),
            faculty=row.get(args.faculty_key),
            title=row.get("title"),
            description=row.get("description"),
            quant_min_score=args.quant_min_score,
        )
        merged = {
            **row,
            "scope_bucket": out["scope_bucket"],
            "scope_reason": out["scope_reason"],
        }
        classified.append(merged)
        bucket_counts[out["scope_bucket"]] = bucket_counts.get(out["scope_bucket"], 0) + 1
        reason_counts[out["scope_reason"]] = reason_counts.get(out["scope_reason"], 0) + 1

    if args.summary or not args.output:
        print(f"Input rows: {len(rows)}")
        print("Bucket counts:")
        for k in sorted(bucket_counts):
            print(f"  {k}: {bucket_counts[k]}")
        print("Reason counts:")
        for k in sorted(reason_counts):
            print(f"  {k}: {reason_counts[k]}")

    if args.show_samples > 0:
        print(f"Samples (first {args.show_samples}):")
        for row in classified[: args.show_samples]:
            row_id = row.get("id") or row.get("code") or row.get("moduleCode") or ""
            title = row.get("title") or ""
            print(f"  {row_id} | {title} | {row['scope_bucket']} | {row['scope_reason']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for row in classified:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved classified rows: {len(classified)} -> {args.output}")


if __name__ == "__main__":
    main()
