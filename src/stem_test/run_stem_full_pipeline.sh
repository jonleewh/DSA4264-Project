#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

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

"$PYTHON_BIN" src/stem_test/stem_prepare_portable_inputs.py
"$PYTHON_BIN" src/stem_test/stem_1a_generate_scope_classifications.py
"$PYTHON_BIN" src/stem_test/stem_1b_scope_classifier.py
"$PYTHON_BIN" src/stem_test/stem_2_create_test_datasets.py
"$PYTHON_BIN" src/stem_test/stem_5_build_canonical_skill_framework.py
if [[ "${RUN_RULE_REGEN:-0}" == "1" ]]; then
  "$PYTHON_BIN" src/stem_test/stem_0_generate_module_skill_rules.py --write-rules
else
  echo "Skipping stem_0_generate_module_skill_rules.py (set RUN_RULE_REGEN=1 to enable rule regeneration)."
fi
"$PYTHON_BIN" src/stem_test/stem_3f_extract_module_skills_independent.py
"$PYTHON_BIN" src/stem_test/stem_4f_extract_job_ssoc3_from_original.py
"$PYTHON_BIN" src/stem_test/stem_5_build_canonical_skill_framework.py
"$PYTHON_BIN" src/stem_test/stem_6f_canonical_skill_mapper.py --map-stem-pipeline
"$PYTHON_BIN" src/stem_test/stem_8f_align_module_job_canonical.py

echo "Full STEM pipeline completed. Outputs are in data/stem_full/"
