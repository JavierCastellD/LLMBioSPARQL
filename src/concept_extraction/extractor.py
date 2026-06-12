"""Concept extraction: turn a natural-language question into typed nodes and
relation triples (see docs/pipeline_concept_extraction_planner_spec.md, Task 2).
"""
from __future__ import annotations

import json
import re
from collections import deque
from typing import Any

import ollama

from ..config import EXTRACTOR_MAX_RETRIES, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_THINK
from ..pathmap.builder import VALID_OBJECT_TYPES
from ..pathmap.planner import RELATION_NAMES
from .prompts_loader import render_prompt

# Allowed entity types for a node's "type". Mirrors the builder's
# VALID_OBJECT_TYPES, plus "disease" (the planner aliases it to "phenotype")
# in case the model emits it despite the prompt's "use phenotype" guidance.
ENTITY_TYPES = VALID_OBJECT_TYPES | {"disease"}


def _extract_json(text: str) -> Any | None:
    """Parse "text" as JSON, tolerating code fences and surrounding prose.

    Parameters:
        text (str): The raw LLM completion.

    Returns:
        Any | None: The parsed JSON value, or None if no JSON object could be
        recovered from "text".
    """
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _connected_to_output(nodes: dict, triples: list[dict]) -> set[str]:
    """Return the set of node ids reachable from "OUTPUT" via "triples", treated
    as an undirected graph.
    """
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    for triple in triples:
        subj, obj = triple.get("subject"), triple.get("object")
        if subj in adjacency and obj in adjacency:
            adjacency[subj].add(obj)
            adjacency[obj].add(subj)

    visited = {"OUTPUT"}
    queue = deque(["OUTPUT"])
    while queue:
        node_id = queue.popleft()
        for neighbour in adjacency.get(node_id, ()):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def _validate(parsed: Any) -> list[str]:
    """Validate a parsed concept-extraction response.

    Parameters:
        parsed (Any): The JSON value returned by _extract_json (possibly None).

    Returns:
        list[str]: Human-readable problems found, or [] if "parsed" is valid.
    """
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]

    errors: list[str] = []

    nodes = parsed.get("nodes")
    if not isinstance(nodes, dict):
        return ['"nodes" must be an object mapping node ids to {"type": ...}']

    triples = parsed.get("triples")
    if not isinstance(triples, list):
        return ['"triples" must be a list']

    if "OUTPUT" not in nodes:
        errors.append('"nodes" must contain a node with id "OUTPUT"')

    for node_id, node in nodes.items():
        if not isinstance(node, dict) or "type" not in node:
            errors.append(f'node {node_id!r} is missing "type"')
            continue
        if node["type"] not in ENTITY_TYPES:
            errors.append(
                f'node {node_id!r} has type {node["type"]!r}, '
                f'which is not one of the allowed entity types'
            )

    for i, triple in enumerate(triples):
        if not isinstance(triple, dict) or not {"subject", "relation", "object"} <= triple.keys():
            errors.append(f'triples[{i}] must have "subject", "relation", and "object"')
            continue
        if triple["relation"] not in RELATION_NAMES:
            errors.append(
                f'triples[{i}] has relation {triple["relation"]!r}, '
                f'which is not one of the allowed relation names'
            )
        for key in ("subject", "object"):
            if triple[key] not in nodes:
                errors.append(f'triples[{i}]["{key}"] references unknown node id {triple[key]!r}')

    if errors:
        return errors

    if not triples:
        if "span" not in nodes["OUTPUT"]:
            errors.append(
                '"triples" is empty, so OUTPUT must have a "span" '
                '(an attribute-only question must name the entity)'
            )
    else:
        connected = _connected_to_output(nodes, triples)
        disconnected = set(nodes) - connected
        if disconnected:
            errors.append(
                f'nodes not connected to "OUTPUT" via "triples": {sorted(disconnected)}'
            )

    return errors


def _repair_prompt(original_prompt: str, previous_response: str, errors: list[str]) -> str:
    """Build a follow-up prompt asking the model to fix a previous invalid response."""
    issues = "\n".join(f"- {e}" for e in errors)
    return (
        f"{original_prompt}\n\n"
        f"Your previous answer was:\n{previous_response}\n\n"
        f"That answer is INVALID for these reasons:\n{issues}\n\n"
        f"Output ONLY a corrected JSON object that fixes these issues."
    )


def build_prompt(question: str) -> str:
    """Render the concept-extraction prompt for "question"."""
    return render_prompt(
        "concept_extraction",
        ENTITY_TYPES="\n".join(f"- {t}" for t in sorted(VALID_OBJECT_TYPES)),
        RELATION_NAMES="\n".join(f"- {r}" for r in RELATION_NAMES),
        QUESTION=question,
    )


class ConceptExtractor:
    """Extracts typed nodes and relation triples from a natural-language question
    via an LLM, backed by a local Ollama server.
    """

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL, temperature: float = 0.0) -> None:
        self._client = ollama.Client(host=base_url)
        self._model = model
        self._temperature = temperature

    def extract(self, question: str) -> dict[str, Any]:
        """Extract typed nodes and relation triples from a natural-language question.

        Parameters:
            question (str): The natural-language question.

        Returns:
            dict[str, Any]: On success, {"nodes": ..., "triples": ...} as described in
            docs/pipeline_concept_extraction_planner_spec.md (Task 2). On failure
            (after EXTRACTOR_MAX_RETRIES attempts), a structured error:
            {"error": "extraction_failed", "details": [...], "raw": <last response>}.
        """
        prompt = build_prompt(question)

        raw = self._complete(prompt)
        for attempt in range(EXTRACTOR_MAX_RETRIES):
            parsed = _extract_json(raw)
            errors = _validate(parsed)
            if not errors:
                return parsed
            if attempt == EXTRACTOR_MAX_RETRIES - 1:
                return {"error": "extraction_failed", "details": errors, "raw": raw}
            raw = self._complete(_repair_prompt(prompt, raw, errors))

        raise AssertionError("unreachable")  # EXTRACTOR_MAX_RETRIES >= 1

    def _complete(self, prompt: str) -> str:
        response = self._client.generate(
            model=self._model,
            prompt=prompt,
            think=OLLAMA_THINK,
            options={"temperature": self._temperature},
        )
        return response["response"]

