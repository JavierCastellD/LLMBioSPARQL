"""Run output designation over a batch of examples and save the results.

Usage:
    python tests/run_output_designation.py <input.json> <output.json>

<input.json> is a list of objects, each with a "question" key and an "extractor"
key whose value is the concept-extraction result (the format written by
run_concept_extraction.py). <output.json> is written as a list of
{"question": ..., "output_designation": ...} objects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.output_designation.designator import OutputDesignator


def _build_output_nodes(extraction: dict) -> list[dict]:
    """Build the output_nodes list from a concept-extraction result.

    Parameters:
        extraction (dict):
            Concept-extraction result with keys "outputs" and "nodes".

    Returns:
        List of dicts with node_id, type, and optional span — one per output node.
    """
    outputs = extraction.get("outputs", [])
    nodes = extraction.get("nodes", {})
    result = []
    for nid in outputs:
        node = nodes.get(nid, {})
        entry = {"node_id": nid, "type": node.get("type", "unknown")}
        if "span" in node:
            entry["span"] = node["span"]
        result.append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file with a list of {question, extractor} objects")
    parser.add_argument("output", type=Path, help="Where to write the JSON results")
    args = parser.parse_args()

    examples = json.loads(args.input.read_text(encoding="utf-8"))

    designator = OutputDesignator()
    results = []
    for i, item in enumerate(examples, start=1):
        question = item["question"]
        output_nodes = _build_output_nodes(item["extractor"])
        print(f"[{i}/{len(examples)}] {question}")
        result = designator.designate(question, output_nodes)
        results.append({"question": question, "output_designation": result})
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
