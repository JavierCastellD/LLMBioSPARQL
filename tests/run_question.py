"""Run a single question through concept extraction and the path planner,
and display the resulting plan.

Usage:
    python tests/run_question.py "Which genes are involved in DNA repair in humans?"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.concept_extraction.extractor import ConceptExtractor
from src.pathmap.planner import PathPlanner, print_resolved


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} \"<question>\"")
        sys.exit(1)
    question = sys.argv[1]

    extracted = ConceptExtractor().extract(question)
    print("--- Concept extraction ---")
    print(json.dumps(extracted, indent=2, ensure_ascii=False))

    if "error" in extracted:
        return

    print("\n--- Plan ---")
    result = PathPlanner().resolve_relations(extracted["nodes"], extracted["triples"])
    print_resolved(result)


if __name__ == "__main__":
    main()
