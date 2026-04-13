# DSA4264-Project

## Overview
Our project analyses how well Singapore university courses prepare students today for real-world jobs. The following setup instructions will help you get started with our project. Ensure that the dependencies are properly installed, and the data pipeline is configured correctly.

## Repository Strucutre

Include the repository tree here

## How to run the project

1. **Clone our repository**

    ```{}
    git clone https://github.com/jonleewh/DSA4264-Project.git
    ```

2. **Install data files**

    To get started, please download the data files:

    - 

3. **Create a virtual environment** (use Python 3.11 or higher)

    Run these commands from the **repository root**. We recommend creating the environment as `.venv/`.

    Windows (PowerShell):
    ```powershell
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    deactivate
    ```

    Windows (Command Prompt):
    ```bat
    py -3.11 -m venv .venv
    .\.venv\Scripts\activate
    deactivate
    ```

    macOS / Linux:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    deactivate
    ```

    If `py` is unavailable on Windows, use `python -m venv .venv` instead. If `python3` points to Python older than 3.11, use the exact interpreter name installed on your machine, such as `python3.11`.

4. **Install dependencies**

    To install the required dependencies, run:
    ```bash
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

    This also installs the Jupyter dependencies used by the notebooks.

5. **Start Jupyter for the notebooks**

    Run Jupyter from the **repository root** so the notebooks can find the project files:
    ```bash
    python -m jupyter lab
    ```

    Then open the notebooks in `src/notebooks/`.

6. **Run the scraper scripts**

    Run these from the repository root in this order:
    ```bash
    python src/scrappers/NTUMods.py
    python src/scrappers/NTUModsDesc.py
    python src/scrappers/NUSModsAPI.py
    python src/scrappers/SUTDCourses.py
    python src/scrappers/SUTDDesc.py
    ```

7. **Run the cleaning notebooks**

    The baseline pipeline assumes the notebooks have already produced:
    - `data/cleaned_data/combined_courses_cleaned.pkl`
    - `data/cleaned_data/jobs_cleaned.pkl`

8. **Run the general model pipeline**

    The official general pipeline now lives in `src/create_test/baseline/`.

    Development / inspection mode:
    ```bash
    python src/create_test/baseline/create_test_datasets.py
    ```

    This creates sampled downstream datasets by default:
    - 500 module rows
    - 1000 job rows

    Full dataset mode:
    ```bash
    python src/create_test/baseline/create_test_datasets.py --full-dataset
    ```

    Build the canonical skill framework used by the mapper:
    ```bash
    python src/create_test/baseline/build_canonical_skill_framework.py
    ```

    Then continue with:
    ```bash
    python src/create_test/baseline/extract_job_ssoc3_from_original.py
    ```

    Canonical mapping for both module-side and job-side skills:
    ```bash
    python src/create_test/baseline/canonical_skill_mapper.py
    ```

    Optional narrower runs for debugging:
    ```bash
    python src/create_test/baseline/canonical_skill_mapper.py --target module
    python src/create_test/baseline/canonical_skill_mapper.py --target job
    ```

    Alignment:
    ```bash
    python src/create_test/baseline/align_module_job_canonical.py
    ```

    Optional direct comparison against the independent module skill extractor:
    ```bash
    python src/create_test/experimental/run_independent_comparison.py
    ```

    This reuses the same baseline test rows, canonical framework, and job-side canonical outputs, so the comparison isolates only the module skill extraction method.

9. **Set up pre-commit hooks**



10. **Run the data pipelines**



## Remarks
- **Do not remove any lines from the `.gitignore` file** provided in the repository to prevent committing unnecessary or sensitive files.
- **Do not commit any data or credentials** to the repository. Keep all credentials and configurations in your local directory.
- `src/create_test/baseline/` is the official general pipeline.
- `src/create_test/experimental/` contains the supported side-by-side comparison path for the independent module skill extractor.
- `src/create_test/legacy/` contains older alternate extractors, benchmarking helpers, and reference scripts retained temporarily.
