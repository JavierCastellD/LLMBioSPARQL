"""Run the ablation baseline (direct question -> SPARQL, no intermediate
pipeline stages) over a batch of questions and save the results.

Usage:
    python tests/run_direct_generator.py <input.json> <output.json>

<input.json> is a list of objects with at least a "question" key (any other
keys, e.g. gold "extractor"/"linker"/"sparql", are ignored -- this baseline
only ever sees the question). <output.json> is written as a list of
{"question": ..., "query": ..., ["prompt": ...], ["error": ...]} objects, the
same shape run_sparql_generator.py writes, so evaluation.ipynb can score this
baseline with the same query-execution logic used for the structured
pipeline's SPARQL-generation sections.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.direct_generator.generator import DirectSPARQLGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help='JSON file with a list of {"question": ...} objects')
    parser.add_argument("output", type=Path, help="Where to write the JSON results")
    args = parser.parse_args()

    examples = json.loads(args.input.read_text(encoding="utf-8"))

    generator = DirectSPARQLGenerator()
    results = []
    for i, item in enumerate(examples, start=1):
        question = item["question"]
        print(f"[{i}/{len(examples)}] {question}")

        result = generator.generate(question)
        results.append({"question": question, **result})
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
