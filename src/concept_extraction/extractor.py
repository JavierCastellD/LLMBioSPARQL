"""Concept extraction: turn a natural-language question into typed nodes and
relation triples (see docs/pipeline_concept_extraction_planner_spec.md, Task 2).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from ..config import CONCEPT_EXTRACTION_PROMPT, EXTRACTOR_MAX_RETRIES
from ..pathmap.builder import VALID_OBJECT_TYPES
from ..pathmap.planner import RELATION_NAMES
from ..utils.llm import LLMCallable, OllamaLLM, llm_call_with_retry
from ..utils.prompts_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Allowed entity types for a node's "type". Mirrors the builder's
# VALID_OBJECT_TYPES, plus "disease" (the planner aliases it to "phenotype")
# in case the model emits it despite the prompt's "use phenotype" guidance.
ENTITY_TYPES = VALID_OBJECT_TYPES | {"disease"}


def _connected_to_output(nodes: dict, triples: list[dict], outputs: list[str]) -> set[str]:
    """Return the set of node ids reachable from any output id via triples, treated as an undirected graph.

    Parameters:
        nodes (dict):
            Mapping of node id to node descriptor dict.
        triples (list[dict]):
            List of triple dicts, each with "subject", "relation", and "object" keys.
        outputs (list[str]):
            Node ids the question asks to return (BFS roots).

    Returns:
        Set of node ids connected to any output id, including the output ids themselves.
    """
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    for triple in triples:
        subj, obj = triple.get("subject"), triple.get("object")
        if subj in adjacency and obj in adjacency:
            adjacency[subj].add(obj)
            adjacency[obj].add(subj)

    visited = {o for o in outputs if o in adjacency}
    queue = deque(visited)
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
        parsed (Any):
            The JSON value returned by extract_json (possibly None).

    Returns:
        A list of human-readable problems found, or an empty list if the response is valid.
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

    outputs = parsed.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return ['"outputs" must be a non-empty list of node ids']

    for node_id in outputs:
        if node_id not in nodes:
            errors.append(f'"outputs" references unknown node id {node_id!r}')

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
        missing_span = [o for o in outputs if "span" not in nodes[o]]
        if missing_span:
            errors.append(
                f'"triples" is empty, so every output node must have a "span" '
                f'(an attribute-only question must name the entity): {missing_span}'
            )
    else:
        connected = _connected_to_output(nodes, triples, outputs)
        disconnected = set(nodes) - connected
        if disconnected:
            errors.append(
                f'nodes not connected to any output via "triples": {sorted(disconnected)}'
            )

    return errors


def build_extraction_prompt(question: str) -> str:
    """Render the concept-extraction prompt template for a given question.

    Parameters:
        question (str):
            The natural-language question to embed in the prompt.

    Returns:
        The fully rendered prompt string ready to send to the LLM.
    """
    return render_prompt(
        CONCEPT_EXTRACTION_PROMPT,
        prompts_dir=_PROMPTS_DIR,
        ENTITY_TYPES="\n".join(f"- {t}" for t in sorted(VALID_OBJECT_TYPES)),
        RELATION_NAMES="\n".join(f"- {r}" for r in RELATION_NAMES),
        QUESTION=question,
    )


class ConceptExtractor:
    """Extracts typed nodes and relation triples from a natural-language question via an LLM.

    Attributes:
        llm_client (LLMCallable):
            Callable that takes a prompt string and returns the model's response.
            Defaults to an OllamaLLM instance.
    """

    def __init__(self, llm_client: LLMCallable | None = None) -> None:
        """Initializes the attributes of the class.

        Parameters:
            llm_client (LLMCallable):
                Callable (prompt: str) -> str. Defaults to OllamaLLM.
        """
        self._llm = llm_client or OllamaLLM()

    def extract(self, question: str) -> dict[str, Any]:
        """Extract typed nodes and relation triples from a natural-language question.

        Parameters:
            question (str):
                The natural-language question to extract concepts from.

        Returns:
            On success, a dict with "nodes" and "triples" keys as described in
            docs/pipeline_concept_extraction_planner_spec.md (Task 2).
            On failure after EXTRACTOR_MAX_RETRIES attempts, {"error": "extraction_failed"}.
        """
        prompt = build_extraction_prompt(question)
        result = llm_call_with_retry(self._llm, prompt, _validate, EXTRACTOR_MAX_RETRIES)
        if result is None:
            return {"error": "extraction_failed"}
        return result
