import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = PROJECT_ROOT / "data" / "test"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

DEFAULT_MODULE_INPUT = TEST_DIR / "module_descriptions_test.jsonl"
DEFAULT_INDEPENDENT_OUTPUT = TEST_DIR / "module_descriptions_test_with_skills_independent.jsonl"
DEFAULT_FRAMEWORK = REFERENCE_DIR / "canonical_skill_framework_v4.json"
DEFAULT_JOB_CANONICAL = TEST_DIR / "job_skills_canonical.jsonl"
DEFAULT_MODULE_CANONICAL = TEST_DIR / "module_skills_canonical_independent.jsonl"
DEFAULT_ALIGNMENT_OUTPUT = TEST_DIR / "module_job_alignment_independent.json"


def run_step(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            "Run the baseline pipeline first so the shared test datasets, framework, "
            "and job-side canonical outputs are available."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the experimental independent module-skill extraction path and align it "
            "against the same job-side canonical baseline outputs for direct comparison."
        )
    )
    parser.add_argument("--module-input", type=Path, default=DEFAULT_MODULE_INPUT)
    parser.add_argument("--independent-output", type=Path, default=DEFAULT_INDEPENDENT_OUTPUT)
    parser.add_argument("--framework", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--job-canonical", type=Path, default=DEFAULT_JOB_CANONICAL)
    parser.add_argument("--module-canonical-output", type=Path, default=DEFAULT_MODULE_CANONICAL)
    parser.add_argument("--alignment-output", type=Path, default=DEFAULT_ALIGNMENT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--semantic-min-score", type=float, default=0.62)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_exists(args.module_input, "baseline module test dataset")
    ensure_exists(args.framework, "canonical skill framework")
    ensure_exists(args.job_canonical, "baseline job canonical dataset")

    extract_cmd = [
        sys.executable,
        "src/create_test/experimental/extract_module_skills_independent.py",
        "--module-input",
        str(args.module_input),
        "--output",
        str(args.independent_output),
        "--top-k",
        str(args.top_k),
        "--semantic-min-score",
        str(args.semantic_min_score),
    ]
    if args.max_rows is not None:
        extract_cmd.extend(["--max-rows", str(args.max_rows)])
    run_step(extract_cmd)

    canonicalize_cmd = [
        sys.executable,
        "src/create_test/baseline/canonical_skill_mapper.py",
        "--target",
        "custom",
        "--framework",
        str(args.framework),
        "--input-jsonl",
        str(args.independent_output),
        "--output-jsonl",
        str(args.module_canonical_output),
    ]
    run_step(canonicalize_cmd)

    align_cmd = [
        sys.executable,
        "src/create_test/baseline/align_module_job_canonical.py",
        "--module",
        str(args.module_canonical_output),
        "--job",
        str(args.job_canonical),
        "--output",
        str(args.alignment_output),
    ]
    run_step(align_cmd)

    print("Independent comparison outputs ready:", flush=True)
    print(f"- Extracted module skills: {args.independent_output}")
    print(f"- Canonical module skills: {args.module_canonical_output}")
    print(f"- Alignment results: {args.alignment_output}")


if __name__ == "__main__":
    main()
