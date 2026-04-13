import argparse
import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_BUILDER = PROJECT_ROOT / "src" / "create_test" / "baseline" / "build_canonical_skill_framework.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "reference" / "canonical_skill_framework_v4.json"


def load_shared_builder():
    spec = importlib.util.spec_from_file_location("shared_canonical_builder", SHARED_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the shared canonical skill framework used by both "
            "the baseline create_test and STEM pipelines."
        )
    )
    parser.add_argument("--version", default="v4", help="Framework version label.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the shared framework JSON.",
    )
    args = parser.parse_args()

    shared_builder = load_shared_builder()
    framework = shared_builder.build_framework(args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(framework, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote shared canonical framework to {args.output}")


if __name__ == "__main__":
    main()
