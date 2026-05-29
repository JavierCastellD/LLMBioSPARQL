"""
Run the QueryExtractor over every question in a JSON file and save results.

Usage:
    python tests/run_extraction.py                                  # defaults to queries/test_prompts.json
    python tests/run_extraction.py queries/examples_extended.json
    python tests/run_extraction.py queries/examples_simple.json -o results/simple_out.json

Each entry in the output file contains:
    question   – original question
    expected   – ground-truth 'extracted' dict from the input file (if present)
    predicted  – dict returned by QueryExtractor
    error      – error message if extraction failed (field absent on success)
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extractor import QueryExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QueryExtractor over a JSON question file.")
    parser.add_argument(
        "input",
        nargs="?",
        default="queries/test_prompts.json",
        help="Path to input JSON file (default: queries/test_prompts.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output JSON file (default: results/<input_stem>_<timestamp>.json)",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{input_path.stem}_{timestamp}.json"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)

    extractor = QueryExtractor()
    results = []
    n = len(items)

    print(f"Input : {input_path}  ({n} questions)")
    print(f"Output: {output_path}")
    print(f"Model : {extractor.model_name}")
    print()

    for i, item in enumerate(items, start=1):
        question = item["question"]
        print(f"[{i}/{n}] {question}")

        entry = {
            "question": question,
            "expected": item.get("extracted"),
        }

        try:
            entry["predicted"] = extractor.extract(question)
            print(f"       → OK")
        except Exception as exc:
            entry["predicted"] = None
            entry["error"] = str(exc)
            print(f"       → ERROR: {exc}")

        results.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    failed = sum(1 for r in results if "error" in r)
    print(f"\nDone: {n - failed}/{n} succeeded. Results saved to {output_path}")


if __name__ == "__main__":
    main()
