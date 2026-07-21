"""Run a single question through the full NL->SPARQL pipeline and display each
stage's result: concept extraction, output designation, entity linking, path
planning, and SPARQL generation.

Usage:
    python tests/run_pipeline.py "Which genes are involved in DNA repair in humans?"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pathmap.planner import print_resolved
from src.pipeline import Text2SPARQLPipeline


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} \"<question>\"")
        sys.exit(1)
    question = sys.argv[1]

    result = Text2SPARQLPipeline().run(question)

    print("--- Concept extraction ---")
    print(json.dumps(result["extraction"], indent=2, ensure_ascii=False))
    if "skipped" in result:
        print(f"\nSKIPPED: {result['skipped']}")
        return

    print("\n--- Output designation ---")
    print(json.dumps(result["output_designation"], indent=2, ensure_ascii=False))

    print("\n--- Entity linker ---")
    print(json.dumps(result["linker"], indent=2, ensure_ascii=False))

    print("\n--- Path plan ---")
    print_resolved(result["plan"])

    print("\n--- SPARQL generation ---")
    generation = result["generation"]
    print(generation["query"])
    if generation.get("warnings"):
        print("\nBrief-assembly warnings:")
        for w in generation["warnings"]:
            print(f"  - {w}")
    if "error" in generation:
        print(f"\nLive validation never succeeded: {generation['error']}")


if __name__ == "__main__":
    main()
