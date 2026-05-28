"""
Smoke tests for QueryExtractor.  Run with:
  python -m pytest tests/test_extractor.py -v
or just:
  python tests/test_extractor.py
Requires Ollama to be running locally with the configured model pulled.
"""

import pprint
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor import QueryExtractor

SIMPLE_QUESTIONS = [
    "Which proteins are encoded by the TOX3 gene?",
    "What genes are associated with breast cancer according to OMIM?",
    "Which transcription factors regulate the BRCA1 gene?",
    "What biological processes is the protein TP53_HUMAN involved in?",
    "Are there enhancers that regulate the TOX3 gene in breast tissue?",
]


def run_tests():
    extractor = QueryExtractor()
    passed = 0
    failed = 0

    for question in SIMPLE_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        try:
            result = extractor.extract(question)
            _validate_structure(result)
            pprint.pprint(result)
            print("PASS")
            passed += 1
        except Exception as exc:
            print(f"FAIL — {exc}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(SIMPLE_QUESTIONS)}")
    return failed == 0


def _validate_structure(result: dict) -> None:
    assert isinstance(result, dict), "result must be a dict"
    assert "entities" in result, "missing 'entities'"
    assert "output" in result, "missing 'output'"
    assert "relationships" in result, "missing 'relationships'"
    assert isinstance(result["entities"], list), "'entities' must be a list"
    assert isinstance(result["output"], list), "'output' must be a list"
    assert isinstance(result["relationships"], list), "'relationships' must be a list"
    for rel in result["relationships"]:
        assert isinstance(rel, dict), "each relationship must be a dict"
        assert "subject" in rel and "predicate" in rel and "object" in rel, \
            f"relationship missing keys: {rel}"


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
