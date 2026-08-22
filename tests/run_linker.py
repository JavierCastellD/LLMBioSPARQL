"""Run entity linking over a batch of examples and save the results.

Usage:
    python tests/run_linker.py <input.json> <output.json>

<input.json> is a list of objects, each with at least a "question" key and a
"linker" key whose "input" field contains the span nodes to link (the format
used in queries/train_examples.json). <output.json> is written as a list of
{"question": ..., "linker": {"input": ..., "output": ...}} objects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.entity_linker.linker import EntityLinker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file with a list of examples containing linker.input")
    parser.add_argument("output", type=Path, help="Where to write the JSON results")
    args = parser.parse_args()

    examples = json.loads(args.input.read_text(encoding="utf-8"))

    linker = EntityLinker()
    results = []
    for i, item in enumerate(examples, start=1):
        question = item["question"]
        span_nodes = item["linker"]["input"]
        print(f"[{i}/{len(examples)}] {question}")
        output = linker.link(question, span_nodes)
        results.append({"question": question, "linker": {"input": span_nodes, "output": output}})
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
