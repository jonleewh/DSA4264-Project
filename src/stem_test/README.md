# `src/stem_test`

This folder contains the STEM-focused pipeline.

It is the STEM counterpart to `src/create_test`, but adds a STEM scoping step before building the downstream datasets and alignment outputs.

## Recommended Usage

Test-dataset run:

```bash
bash src/stem_test/run_stem_full_pipeline.sh
```

- This runs the full STEM pipeline in order on the default test-sized dataset.

Full-dataset run:

```bash
bash src/stem_test/run_stem_full_pipeline.sh full
```

- This runs the same STEM pipeline in full-dataset mode.

## Active Pipeline

1. `stem_1_generate_scope_classifications.py`
   - builds university STEM classification reference files from `data/cleaned_data/combined_courses_cleaned.pkl`
2. `stem_1_scope_classifier.py`
   - filters the cleaned course PKL down to STEM-only module rows
   - writes `data/cleaned_data/cleaned_module_rows_STEM.jsonl`
3. `stem_2_create_test_datasets.py`
   - creates the STEM test datasets from:
   - `data/cleaned_data/cleaned_module_rows_STEM.jsonl`
   - `data/cleaned_data/jobs_cleaned.pkl`
4. `stem_3_extract_module_skills_independent.py`
   - extracts module-side STEM skills
5. `stem_4_extract_job_ssoc3_from_original.py`
   - enriches job-side rows with SSOC hierarchy
6. `stem_5_build_canonical_skill_framework.py`
   - builds the shared canonical skill framework
7. `stem_6_canonical_skill_mapper.py`
   - maps module and job skills into the canonical framework
8. `stem_8_align_module_job_canonical.py`
   - aligns STEM modules to job groups in canonical skill space

## Manual Run Order

Run these from the repository root.

Test-dataset flow:

```bash
python src/stem_test/stem_1_generate_scope_classifications.py
python src/stem_test/stem_1_scope_classifier.py
python src/stem_test/stem_2_create_test_datasets.py
python src/stem_test/stem_5_build_canonical_skill_framework.py
python src/stem_test/stem_3_extract_module_skills_independent.py
python src/stem_test/stem_4_extract_job_ssoc3_from_original.py
python src/stem_test/stem_6_canonical_skill_mapper.py --map-stem-pipeline
python src/stem_test/stem_8_align_module_job_canonical.py
```

Full-dataset mode:

```bash
python src/stem_test/stem_1_generate_scope_classifications.py
python src/stem_test/stem_1_scope_classifier.py
python src/stem_test/stem_2_create_test_datasets.py
python src/stem_test/stem_5_build_canonical_skill_framework.py
python src/stem_test/stem_3_extract_module_skills_independent.py --full-dataset
python src/stem_test/stem_4_extract_job_ssoc3_from_original.py --full-dataset
python src/stem_test/stem_6_canonical_skill_mapper.py --full-dataset --map-stem-pipeline
python src/stem_test/stem_8_align_module_job_canonical.py --full-dataset
```

## Notes

- `stem_5_build_canonical_skill_framework.py` reuses the same canonical framework as `src/create_test/baseline`
- `legacy/` contains superseded STEM scripts that are not part of the supported pipeline
