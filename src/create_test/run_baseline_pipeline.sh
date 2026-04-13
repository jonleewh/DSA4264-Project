#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-test}"
if [[ "$MODE" != "test" && "$MODE" != "full" ]]; then
  echo "Usage: bash src/create_test/run_baseline_pipeline.sh [test|full]" >&2
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

if [[ "$MODE" == "full" ]]; then
  "$PYTHON_BIN" src/create_test/baseline/create_test_datasets.py --full-dataset
else
  "$PYTHON_BIN" src/create_test/baseline/create_test_datasets.py
fi

"$PYTHON_BIN" src/create_test/baseline/build_canonical_skill_framework.py
"$PYTHON_BIN" src/create_test/baseline/extract_job_ssoc3_from_original.py
"$PYTHON_BIN" src/create_test/baseline/canonical_skill_mapper.py
"$PYTHON_BIN" src/create_test/baseline/align_module_job_canonical.py

if [[ "$MODE" == "full" ]]; then
  echo "Full baseline pipeline completed."
else
  echo "Test baseline pipeline completed."
fi
