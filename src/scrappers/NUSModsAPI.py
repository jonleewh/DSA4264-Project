import requests
import json
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "NUSModsInfo.json"

def fetch_data_from_api(url, params=None, output_path=DEFAULT_OUTPUT_PATH):
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        with out.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved JSON to: {out.resolve()}")
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    api_url = "https://api.nusmods.com/v2/2025-2026/moduleInfo.json"
    fetch_data_from_api(api_url)
