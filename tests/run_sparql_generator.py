"""Run the SPARQL generator over a batch of examples and save the results.

Usage:
    python tests/run_sparql_generator.py <input.json> <output.json>

<input.json> is a list of objects with "question", "extractor", "linker", and
"output_designation" keys (the format used in queries/train_examples.json and
queries/test_examples.json). The path planner is run internally from each
example's "extractor" nodes/triples. <output.json> is written as a list of
{"question": ..., "query": ..., "warnings": ..., ["error": ...]} objects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pathmap.planner import PathPlanner
from src.sparql_generator.generator import SPARQLGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file with a list of enriched examples")
    parser.add_argument("output", type=Path, help="Where to write the JSON results")
    args = parser.parse_args()

    examples = json.loads(args.input.read_text(encoding="utf-8"))

    planner = PathPlanner()
    generator = SPARQLGenerator()
    results = []
    for i, item in enumerate(examples, start=1):
        question = item["question"]
        extraction = item["extractor"]
        print(f"[{i}/{len(examples)}] {question}")

        plan = planner.resolve_relations(extraction["nodes"], extraction["triples"])
        result = generator.generate(
            question,
            extraction,
            plan,
            item["linker"]["input"],
            item["linker"]["output"],
            item["output_designation"]["output"],
        )
        results.append({"question": question, **result})
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
