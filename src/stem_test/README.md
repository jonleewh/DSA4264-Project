# `src/stem_test`

This folder contains the STEM-focused pipeline.

It is the STEM counterpart to `src/create_test`, with an explicit module-scope stage before downstream skill extraction and alignment.

## Recommended Usage

Run from repository root.

Test-sized run:

```bash
bash src/stem_test/run_stem_full_pipeline.sh
```

Full-dataset run:

```bash
bash src/stem_test/run_stem_full_pipeline.sh full
```

## What The Runner Actually Executes

`run_stem_full_pipeline.sh` runs these steps in order:

1. `stem_1_generate_scope_classifications.py`
   - builds university STEM scope reference files from `data/cleaned_data/combined_courses_cleaned.pkl`
2. `stem_1_scope_classifier.py`
   - classifies modules into STEM/non-STEM buckets and writes:
        - `data/cleaned_data/cleaned_module_rows_STEM.jsonl`
        - `data/cleaned_data/cleaned_module_rows_non_STEM.jsonl`
3. `stem_1_generate_sankey.py`
   - generates scope-flow summary:
        - `src/stem_test/stem_1_sankey_diagram.png`
4. `stem_2_create_test_datasets.py`
   - creates STEM test datasets used by downstream extraction/mapping
5. `stem_5_build_canonical_skill_framework.py`
   - builds/refreshes the shared canonical skill framework
6. optional `stem_0_generate_module_skill_rules.py --write-rules` (only when `RUN_RULE_REGEN=1`)
   - regenerates module-side rule candidates in `module_skill_rules.py`
7. `stem_3_extract_module_skills_independent.py` (`--full-dataset` in full mode)
   - extracts module-side STEM skills
8. `stem_4_extract_job_ssoc3_from_original.py` (`--full-dataset` in full mode)
   - enriches job-side rows with SSOC hierarchy
9. `stem_6_canonical_skill_mapper.py` (`--full-dataset --map-stem-pipeline` in full mode)
   - maps module and job skills into canonical skill space
10. `stem_8_align_module_job_canonical.py` (`--full-dataset` in full mode)
   - aligns STEM modules to SSOC job groups in canonical space

## Core Outputs

From `stem_1_scope_classifier.py`:
- `data/cleaned_data/cleaned_module_rows_STEM.jsonl`
- `data/cleaned_data/cleaned_module_rows_non_STEM.jsonl`

From `stem_1_generate_sankey.py`:
- `src/stem_test/stem_1_sankey_diagram.png`

From downstream alignment:
- test mode outputs in `data/test/`
- full mode outputs in `data/stem_full/`

## Manual Run Order

Use this only when you need step-by-step control.

Test flow:

```bash
python src/stem_test/stem_1_generate_scope_classifications.py
python src/stem_test/stem_1_scope_classifier.py
python src/stem_test/stem_1_generate_sankey.py
python src/stem_test/stem_2_create_test_datasets.py
python src/stem_test/stem_5_build_canonical_skill_framework.py
python src/stem_test/stem_3_extract_module_skills_independent.py
python src/stem_test/stem_4_extract_job_ssoc3_from_original.py
python src/stem_test/stem_6_canonical_skill_mapper.py --map-stem-pipeline
python src/stem_test/stem_8_align_module_job_canonical.py
```

Full flow:

```bash
python src/stem_test/stem_1_generate_scope_classifications.py
python src/stem_test/stem_1_scope_classifier.py
python src/stem_test/stem_1_generate_sankey.py
python src/stem_test/stem_2_create_test_datasets.py
python src/stem_test/stem_5_build_canonical_skill_framework.py
python src/stem_test/stem_3_extract_module_skills_independent.py --full-dataset
python src/stem_test/stem_4_extract_job_ssoc3_from_original.py --full-dataset
python src/stem_test/stem_6_canonical_skill_mapper.py --full-dataset --map-stem-pipeline
python src/stem_test/stem_8_align_module_job_canonical.py --full-dataset
```

## Notes

- `stem_5_build_canonical_skill_framework.py` reuses the same framework source as `src/create_test/baseline`.
- `legacy/` contains superseded scripts and is not part of the supported run path.
