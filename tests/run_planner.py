"""Run the path planner over a batch of concept-extraction results and save
the results.

Usage:
    python tests/run_planner.py <input.json> <output.json>

<input.json> is a list of objects, each with a "question" key and an
"extractor" key holding a concept-extraction result: {"nodes": ..., "triples":
...} (e.g. the "result" field written by run_concept_extraction.py).
<output.json> is written as a list of {"question": ..., "result": ...}
objects, one per input item, where "result" is whatever
PathPlanner.resolve_relations() returned.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pathmap.planner import PathPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help='JSON file with a list of {"question": ..., "extractor": ...} objects')
    parser.add_argument("output", type=Path, help="Where to write the JSON results")
    args = parser.parse_args()

    items = json.loads(args.input.read_text(encoding="utf-8"))

    planner = PathPlanner()
    results = []
    for i, item in enumerate(items, start=1):
        question = item["question"]
        extracted = item["extractor"]
        print(f"[{i}/{len(items)}] {question}")
        result = planner.resolve_relations(extracted["nodes"], extracted["triples"])
        results.append({"question": question, "result": result})
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
