#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

COURSES_INPUT="$ROOT_DIR/data/cleaned_data/combined_courses_cleaned.pkl"
STEM_ROWS_OUTPUT="$ROOT_DIR/data/cleaned_data/cleaned_module_rows_STEM.jsonl"
NON_STEM_ROWS_OUTPUT="$ROOT_DIR/data/cleaned_data/cleaned_module_rows_non_STEM.jsonl"
SCOPE_SAKEY_OUTPUT="$ROOT_DIR/src/stem_test/stem_1_sankey_diagram.png"

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

needs_scope_rebuild() {
  [[ ! -f "$STEM_ROWS_OUTPUT" ]] && return 0
  [[ ! -f "$NON_STEM_ROWS_OUTPUT" ]] && return 0
  [[ "$COURSES_INPUT" -nt "$STEM_ROWS_OUTPUT" ]] && return 0
  [[ "$COURSES_INPUT" -nt "$NON_STEM_ROWS_OUTPUT" ]] && return 0
  return 1
}

needs_sankey_rebuild() {
  [[ ! -f "$SCOPE_SAKEY_OUTPUT" ]] && return 0
  [[ "$STEM_ROWS_OUTPUT" -nt "$SCOPE_SAKEY_OUTPUT" ]] && return 0
  [[ "$NON_STEM_ROWS_OUTPUT" -nt "$SCOPE_SAKEY_OUTPUT" ]] && return 0
  return 1
}

echo "[STEM] Running step 1a: scope classifications"
"$PYTHON_BIN" src/stem_test/stem_1_generate_scope_classifications.py

if [[ "${FORCE_STEM_SCOPE_REBUILD:-0}" == "1" ]] || needs_scope_rebuild; then
  echo "[STEM] Running step 1b: scope classifier"
  "$PYTHON_BIN" src/stem_test/stem_1_scope_classifier.py
else
  echo "[STEM] Reusing cached scope outputs:"
  echo "       $STEM_ROWS_OUTPUT"
  echo "       $NON_STEM_ROWS_OUTPUT"
fi

if [[ "${FORCE_STEM_SCOPE_REBUILD:-0}" == "1" ]] || needs_sankey_rebuild; then
  echo "[STEM] Running step 1c: sankey diagram"
  "$PYTHON_BIN" src/stem_test/stem_1_generate_sankey.py
else
  echo "[STEM] Reusing cached sankey diagram: $SCOPE_SAKEY_OUTPUT"
fi

echo "[STEM] Running step 2: test dataset creation"
"$PYTHON_BIN" src/stem_test/stem_2_create_test_datasets.py
echo "[STEM] Running step 5: canonical framework"
"$PYTHON_BIN" src/stem_test/stem_5_build_canonical_skill_framework.py
if [[ "${RUN_RULE_REGEN:-0}" == "1" ]]; then
  "$PYTHON_BIN" src/stem_test/stem_0_generate_module_skill_rules.py --write-rules
else
  echo "Skipping stem_0_generate_module_skill_rules.py (set RUN_RULE_REGEN=1 to enable rule regeneration)."
fi

if [[ "$MODE" == "full" ]]; then
  echo "[STEM] Running step 3: module skill extraction (full)"
  "$PYTHON_BIN" src/stem_test/stem_3_extract_module_skills_independent.py --full-dataset
  echo "[STEM] Running step 4: job SSOC enrichment (full)"
  "$PYTHON_BIN" src/stem_test/stem_4_extract_job_ssoc3_from_original.py --full-dataset
  echo "[STEM] Running step 6: canonical mapping (full)"
  "$PYTHON_BIN" src/stem_test/stem_6_canonical_skill_mapper.py --full-dataset --map-stem-pipeline
  echo "[STEM] Running step 8: module-job alignment (full)"
  "$PYTHON_BIN" src/stem_test/stem_8_align_module_job_canonical.py --full-dataset
  echo "Full STEM pipeline completed. Outputs are in data/stem_full/"
else
  echo "[STEM] Running step 3: module skill extraction (test)"
  "$PYTHON_BIN" src/stem_test/stem_3_extract_module_skills_independent.py
  echo "[STEM] Running step 4: job SSOC enrichment (test)"
  "$PYTHON_BIN" src/stem_test/stem_4_extract_job_ssoc3_from_original.py
  echo "[STEM] Running step 6: canonical mapping (test)"
  "$PYTHON_BIN" src/stem_test/stem_6_canonical_skill_mapper.py --map-stem-pipeline
  echo "[STEM] Running step 8: module-job alignment (test)"
  "$PYTHON_BIN" src/stem_test/stem_8_align_module_job_canonical.py
  echo "Test STEM pipeline completed. Outputs are in data/test/"
fi
