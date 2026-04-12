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

6. **Set up pre-commit hooks**



7. **Run the data pipelines**



## Remarks
- **Do not remove any lines from the `.gitignore` file** provided in the repository to prevent committing unnecessary or sensitive files.
- **Do not commit any data or credentials** to the repository. Keep all credentials and configurations in your local directory.
