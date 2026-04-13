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

## Notes

- The baseline pipeline assumes the cleaning notebooks have already produced:
  - `data/cleaned_data/combined_courses_cleaned.pkl`
  - `data/cleaned_data/jobs_cleaned.pkl`
- `baseline/create_test_datasets.py` uses development samples by default:
  - 500 module rows
  - 1000 job rows
- Use `baseline/create_test_datasets.py --full-dataset` for the full baseline export.
- `baseline/build_canonical_skill_framework.py` writes:
  - `data/reference/canonical_skill_framework_v4.json`
- `baseline/canonical_skill_mapper.py` now canonicalizes both module and job skill files by default.
- Use `baseline/canonical_skill_mapper.py --target module`, `--target job`, or `--target custom` for narrower runs.
- Experimental scripts are not required for the main end-to-end run.
- For a direct baseline-vs-independent comparison, first run the full baseline path once, then run:
  - `experimental/run_independent_comparison.py`
- That experimental comparison path reuses:
  - the same baseline module test rows
  - the same canonical skill framework
  - the same job-side canonical output
- This keeps the comparison focused on one change only:
  - notebook module skills versus independently extracted module skills
- Additional exploratory extractors and older benchmark helpers now live under `legacy/` so the supported experimental path stays easy to follow.
- Legacy scripts can be removed later once the team confirms they are no longer needed.
