# DSA4264-Project

## Overview

This project studies how well Singapore university courses prepare students for real-world jobs.

The repository contains two supported downstream pipelines:

- `src/create_test/`
  - the general baseline pipeline
  - also includes an experimental comparison path
- `src/stem_test/`
  - the STEM-focused pipeline

Both pipelines now start from notebook-cleaned data outputs and include shell-script shortcuts for the recommended runs.

## Project Setup

1. **Clone the repository**

```bash
git clone https://github.com/jonleewh/DSA4264-Project.git
cd DSA4264-Project
```

2. **Install the required data files**

Place these files in the project before running the pipelines:

- `problem2.zip`
  - extract its contents so the raw job data ends up under `data/data/`
- `ssoc2020.xlsx`
  - place in `data/`
- `ntu_dept_mapping.xlsx`
  - place in `data/`

3. **Create `.env`**

Create a `.env` file in the repository root with:

```bash
nus_api="https://api.nusmods.com/v2/2025-2026/moduleInfo.json"
```

4. **Create a virtual environment**  
Use Python 3.11 or higher.

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

If `python3` points to an older version, use the exact interpreter name installed on your machine, such as `python3.11` or `python3.12`.

5. **Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This also installs the Jupyter dependencies used by the notebooks.

## Data Preparation

1. **Run the scrapers**

Run these from the repository root in this order:

```bash
python src/scrappers/NTUMods.py
python src/scrappers/NTUModsDesc.py
python src/scrappers/NUSModsAPI.py
python src/scrappers/SUTDCourses.py
python src/scrappers/SUTDDesc.py
```

You can also use the scraper shortcut:

```bash
bash src/scrappers/run_scrappers_pipeline.sh
```

2. **Run the cleaning notebooks**

Start Jupyter from the repository root:

```bash
python -m jupyter lab
```

Then run the notebooks in `src/notebooks/`.

The downstream pipelines assume the notebooks have already produced:

- `data/cleaned_data/combined_courses_cleaned.pkl`
- `data/cleaned_data/jobs_cleaned.pkl`

## Recommended Runs

### General Baseline Pipeline

Test-sized run:

```bash
bash src/create_test/run_baseline_pipeline.sh
```

Full-dataset run:

```bash
bash src/create_test/run_baseline_pipeline.sh full
```

### General Experimental Comparison Pipeline

Test-sized run:

```bash
bash src/create_test/run_experimental_pipeline.sh
```

Full-dataset run:

```bash
bash src/create_test/run_experimental_pipeline.sh full
```

This experimental path compares the notebook-based module-skill route against the independent module-skill extractor while reusing the same baseline framework and job-side outputs.

### STEM Pipeline

Test-sized run:

```bash
bash src/stem_test/run_stem_full_pipeline.sh
```

Full-dataset run:

```bash
bash src/stem_test/run_stem_full_pipeline.sh full
```

## Folder Guides

- `src/create_test/README.md`
  - detailed notes for the general baseline and experimental pipelines
- `src/stem_test/README.md`
  - detailed notes for the STEM-focused pipeline

## Remarks

- Do not commit credentials or local data files.
- Keep secrets such as `.env` values only in your local environment.
- `src/create_test/` is the supported general pipeline area.
- `src/stem_test/` is the supported STEM pipeline area.
- `legacy/` folders contain older scripts kept for reference and are not part of the recommended run flow.
