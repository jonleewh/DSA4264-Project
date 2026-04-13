#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-test}"
if [[ "$MODE" != "test" && "$MODE" != "full" ]]; then
  echo "Usage: bash src/create_test/run_experimental_pipeline.sh [test|full]" >&2
  exit 2
fi

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
elif [[ -x "$ROOT_DIR/penv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/penv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No Python interpreter found. Expected .venv/bin/python, venv/bin/python, penv/bin/python, python3, or python." >&2
  exit 127
fi

MODULE_DATASET="$ROOT_DIR/data/test/module_descriptions_test.jsonl"
JOB_DATASET="$ROOT_DIR/data/test/job_descriptions_test.jsonl"
FRAMEWORK="$ROOT_DIR/data/reference/canonical_skill_framework_v4.json"
JOB_SSOC="$ROOT_DIR/data/test/job_ssoc345_with_skills_from_original.jsonl"
JOB_CANONICAL="$ROOT_DIR/data/test/job_skills_canonical.jsonl"
BASELINE_ALIGNMENT="$ROOT_DIR/data/test/module_job_alignment_canonical.json"

line_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  wc -l < "$path" | tr -d ' '
}

needs_baseline_run() {
  if [[ ! -f "$MODULE_DATASET" || ! -f "$JOB_DATASET" || ! -f "$FRAMEWORK" || ! -f "$JOB_SSOC" || ! -f "$JOB_CANONICAL" || ! -f "$BASELINE_ALIGNMENT" ]]; then
    return 0
  fi

  if [[ "$MODE" == "full" ]]; then
    local module_lines
    local job_lines
    module_lines="$(line_count "$MODULE_DATASET")"
    job_lines="$(line_count "$JOB_DATASET")"
    if (( module_lines <= 500 || job_lines <= 1000 )); then
      return 0
    fi
  fi

  return 1
}

if needs_baseline_run; then
  echo "Running baseline pipeline first to prepare required comparison artifacts (${MODE} mode)..."
  bash src/create_test/run_baseline_pipeline.sh "$MODE"
else
  echo "Reusing existing baseline artifacts for experimental comparison (${MODE} mode)."
fi

"$PYTHON_BIN" src/create_test/experimental/run_independent_comparison.py

if [[ "$MODE" == "full" ]]; then
  echo "Experimental comparison pipeline completed for full dataset."
else
  echo "Experimental comparison pipeline completed for test dataset."
fi
