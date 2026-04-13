# `src/create_test`

This folder is organized into three tracks:

- `baseline/`
  - the official general pipeline that starts from notebook-cleaned PKL files
  - this is the path users should run first
- `experimental/`
  - the supported side-by-side comparison path against the baseline
  - currently focused on the independent module skill extractor comparison
- `legacy/`
  - older preprocessing, benchmarking, and alternate extraction scripts retained temporarily for reference

## Baseline run order

1. `baseline/create_test_datasets.py`
2. `baseline/build_canonical_skill_framework.py`
3. `baseline/extract_job_ssoc3_from_original.py`
4. `baseline/canonical_skill_mapper.py`
5. `baseline/align_module_job_canonical.py`

## Shell shortcuts

Baseline pipeline:

```bash
bash src/create_test/run_baseline_pipeline.sh
```

- This runs the baseline pipeline in order using the default test-sized export.
- To run the same baseline pipeline on the full dataset:

```bash
bash src/create_test/run_baseline_pipeline.sh full
```

Experimental comparison pipeline:

```bash
bash src/create_test/run_experimental_pipeline.sh
```

- This runs the experimental comparison pipeline on the default test-sized dataset.
- It reuses existing baseline artifacts when they are already present, and only runs the baseline pipeline first if those required inputs are missing.

```bash
bash src/create_test/run_experimental_pipeline.sh full
```

- This runs the same experimental comparison flow in full-dataset mode.
- In `full` mode, the shortcut checks whether the current baseline artifacts already look like full-dataset outputs; if not, it runs the baseline full pipeline first.

## Notes

- The baseline pipeline assumes the cleaning notebooks have already produced:
  - `data/cleaned_data/combined_courses_cleaned.pkl`
  - `data/cleaned_data/jobs_cleaned.pkl`
- `baseline/create_test_datasets.py` uses development samples by default:
  - 500 module rows
  - 1000 job rows
- `baseline/build_canonical_skill_framework.py` writes:
  - `data/reference/canonical_skill_framework_v4.json`
- The STEM pipeline now reuses that same framework file as well, so both flows stay aligned on a single canonical vocabulary.
- `baseline/canonical_skill_mapper.py` now canonicalizes both module and job skill files by default.
- Experimental scripts are not required for the main end-to-end run.
- For a direct baseline-vs-independent comparison, first run the full baseline path once, then run:
  - `experimental/run_independent_comparison.py`
- `run_experimental_pipeline.sh` is the simpler shortcut for the supported comparison flow on either the test dataset or the full dataset.
- That experimental comparison path reuses:
  - the same baseline module test rows
  - the same canonical skill framework
  - the same job-side canonical output
- This keeps the comparison focused on one change only:
  - notebook module skills versus independently extracted module skills
- Additional exploratory extractors and older benchmark helpers now live under `legacy/` so the supported experimental path stays easy to follow.
