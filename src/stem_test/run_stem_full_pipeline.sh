#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python src/stem_test/stem_1_scope_classifier.py
python src/stem_test/stem_2_create_test_datasets.py
python src/stem_test/stem_0_generate_module_skill_rules.py --write-rules
python src/stem_test/stem_3f_extract_module_skills_independent.py
python src/stem_test/stem_4f_extract_job_ssoc3_from_original.py
python src/stem_test/stem_5_build_canonical_skill_framework.py
python src/stem_test/stem_6f_canonical_skill_mapper.py --map-stem-pipeline
python src/stem_test/stem_8f_align_module_job_canonical.py

echo "Full STEM pipeline completed. Outputs are in data/stem_full/"
