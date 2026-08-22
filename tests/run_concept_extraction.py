"""Run concept extraction over a batch of questions and save the results.

Usage:
    python tests/run_concept_extraction.py <input.json> <output.json>

<input.json> is a list of objects, each with at least a "question" key (any
other keys are ignored). <output.json> is written as a list of
{"question": ..., "extractor": ...} objects, one per input question, where
"extractor" is whatever ConceptExtractor.extract() returned (either the
extracted {"outputs", "nodes", "triples"} or a structured error).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.concept_extraction.extractor import ConceptExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help='JSON file with a list of {"question": ...} objects')
    parser.add_argument("output", type=Path, help="Where to write the JSON results")
    args = parser.parse_args()

    questions = json.loads(args.input.read_text(encoding="utf-8"))

    extractor = ConceptExtractor()
    results = []
    for i, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"[{i}/{len(questions)}] {question}")
        results.append({"question": question, "extractor": extractor.extract(question)})

    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
