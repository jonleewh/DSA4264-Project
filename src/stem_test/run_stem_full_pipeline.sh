#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-test}"
if [[ "$MODE" != "test" && "$MODE" != "full" ]]; then
  echo "Usage: bash src/stem_test/run_stem_full_pipeline.sh [test|full]" >&2
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

"$PYTHON_BIN" src/stem_test/stem_1_generate_scope_classifications.py
"$PYTHON_BIN" src/stem_test/stem_1_scope_classifier.py
"$PYTHON_BIN" src/stem_test/stem_2_create_test_datasets.py
"$PYTHON_BIN" src/stem_test/stem_5_build_canonical_skill_framework.py
if [[ "${RUN_RULE_REGEN:-0}" == "1" ]]; then
  "$PYTHON_BIN" src/stem_test/stem_0_generate_module_skill_rules.py --write-rules
else
  echo "Skipping stem_0_generate_module_skill_rules.py (set RUN_RULE_REGEN=1 to enable rule regeneration)."
fi

if [[ "$MODE" == "full" ]]; then
  "$PYTHON_BIN" src/stem_test/stem_3_extract_module_skills_independent.py --full-dataset
  "$PYTHON_BIN" src/stem_test/stem_4_extract_job_ssoc3_from_original.py --full-dataset
  "$PYTHON_BIN" src/stem_test/stem_6_canonical_skill_mapper.py --full-dataset --map-stem-pipeline
  "$PYTHON_BIN" src/stem_test/stem_8_align_module_job_canonical.py --full-dataset
  echo "Full STEM pipeline completed. Outputs are in data/stem_full/"
else
  "$PYTHON_BIN" src/stem_test/stem_3_extract_module_skills_independent.py
  "$PYTHON_BIN" src/stem_test/stem_4_extract_job_ssoc3_from_original.py
  "$PYTHON_BIN" src/stem_test/stem_6_canonical_skill_mapper.py --map-stem-pipeline
  "$PYTHON_BIN" src/stem_test/stem_8_align_module_job_canonical.py
  echo "Test STEM pipeline completed. Outputs are in data/test/"
fi
