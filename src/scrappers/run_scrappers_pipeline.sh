#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No Python interpreter found. Expected .venv/bin/python, venv/bin/python, python3, or python." >&2
  exit 127
fi

RUN_NUS=1
RUN_NTU=1
RUN_SUTD=1
DRY_RUN=0
NTU_DESC_LIMIT="${NTU_DESC_LIMIT:-}"

usage() {
  cat <<'EOF'
Usage: src/scrappers/run_scrappers_pipeline.sh [options]

Runs scraper pipeline for NUS, NTU, and SUTD in dependency order.

Options:
  --only-nus           Run only NUS scraper
  --only-ntu           Run only NTU scrapers
  --only-sutd          Run only SUTD scrapers
  --skip-nus           Skip NUS scraper
  --skip-ntu           Skip NTU scrapers
  --skip-sutd          Skip SUTD scrapers
  --ntu-limit N        Pass --limit N to NTUModsDesc.py
  --dry-run            Print commands without executing
  -h, --help           Show this help

Environment:
  NTU_DESC_LIMIT       Equivalent to --ntu-limit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-nus)
      RUN_NUS=1
      RUN_NTU=0
      RUN_SUTD=0
      shift
      ;;
    --only-ntu)
      RUN_NUS=0
      RUN_NTU=1
      RUN_SUTD=0
      shift
      ;;
    --only-sutd)
      RUN_NUS=0
      RUN_NTU=0
      RUN_SUTD=1
      shift
      ;;
    --skip-nus)
      RUN_NUS=0
      shift
      ;;
    --skip-ntu)
      RUN_NTU=0
      shift
      ;;
    --skip-sutd)
      RUN_SUTD=0
      shift
      ;;
    --ntu-limit)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --ntu-limit" >&2
        exit 2
      fi
      NTU_DESC_LIMIT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

run_step() {
  local label="$1"
  shift
  echo ""
  echo "==> ${label}"
  echo "Command: $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

if [[ "$RUN_NUS" == "1" ]]; then
  run_step "NUS: Fetch module info from NUSMods API" \
    "$PYTHON_BIN" src/scrappers/NUSModsAPI.py
else
  echo "Skipping NUS scraper."
fi

if [[ "$RUN_NTU" == "1" ]]; then
  run_step "NTU: Fetch course list" \
    "$PYTHON_BIN" src/scrappers/NTUMods.py

  if [[ -n "${NTU_DESC_LIMIT}" ]]; then
    run_step "NTU: Fetch course details (limited)" \
      "$PYTHON_BIN" src/scrappers/NTUModsDesc.py --limit "$NTU_DESC_LIMIT"
  else
    run_step "NTU: Fetch course details" \
      "$PYTHON_BIN" src/scrappers/NTUModsDesc.py
  fi
else
  echo "Skipping NTU scrapers."
fi

if [[ "$RUN_SUTD" == "1" ]]; then
  run_step "SUTD: Fetch course list" \
    "$PYTHON_BIN" src/scrappers/SUTDCourses.py
  run_step "SUTD: Fetch course descriptions" \
    "$PYTHON_BIN" src/scrappers/SUTDDesc.py
else
  echo "Skipping SUTD scrapers."
fi

echo ""
echo "Scraper pipeline completed."
echo "Output files are written under: $ROOT_DIR/data"
