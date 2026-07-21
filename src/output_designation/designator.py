"""Output designation: identify which literal attributes a question asks to return.

Runs after concept extraction. Given the output nodes, asks the LLM which named
attributes (e.g. synonym, definition, start position) the question explicitly
requests for each output entity.

The LLM client is an injectable callable (str -> str) so any backend works.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import OUTPUT_DESIGNATION_MAX_RETRIES, OUTPUT_DESIGNATION_PROMPT
from ..utils.llm import LLMCallable, OllamaLLM, llm_call_with_retry
from ..utils.prompts_loader import render_prompt

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Preferred attribute names. Extend this list as new attribute requests are observed.
SUGGESTED_ATTRIBUTES = [
    "synonym",
    "definition",
    "start position",
    "end position",
    "close match",
    "strand",
    "description",
    "comment",
]


def _render_extraction(output_nodes: list[dict]) -> str:
    """Render a compact representation of the output nodes for the LLM prompt.

    Parameters:
        output_nodes (list[dict]):
            List of dicts, each with keys node_id (str), type (str), and
            optionally span (str).

    Returns:
        A single line like "Outputs: n0 (gene, \"GLA\"), n1 (protein)".
    """
    parts = []
    for node in output_nodes:
        nid = node["node_id"]
        node_type = node["type"]
        span = node.get("span")
        if span:
            parts.append(f'{nid} ({node_type}, "{span}")')
        else:
            parts.append(f"{nid} ({node_type})")
    return "Outputs: " + ", ".join(parts)


def build_designation_prompt(question: str, output_nodes: list[dict]) -> str:
    """Render the output-designation prompt for a given question and output nodes.

    Parameters:
        question (str):
            The natural-language question to analyse.
        output_nodes (list[dict]):
            List of output node dicts, each with node_id, type, and optional span.

    Returns:
        The fully rendered prompt string ready to send to the LLM.
    """
    return render_prompt(
        OUTPUT_DESIGNATION_PROMPT,
        prompts_dir=_PROMPTS_DIR,
        QUESTION=question,
        EXTRACTION=_render_extraction(output_nodes),
    )


def _validate(parsed: Any, output_ids: set[str]) -> list[str]:
    """Validate an output-designation LLM response.

    Parameters:
        parsed (Any):
            The JSON value returned by extract_json (possibly None).
        output_ids (set[str]):
            The valid output node ids.

    Returns:
        A list of human-readable problems found, or an empty list if valid.
    """
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]
    attributes = parsed.get("attributes")
    if not isinstance(attributes, list):
        return ['"attributes" must be a list']
    errors: list[str] = []
    for i, entry in enumerate(attributes):
        if not isinstance(entry, dict):
            errors.append(f"attributes[{i}] is not an object")
            continue
        on = entry.get("on")
        if on not in output_ids:
            errors.append(
                f'attributes[{i}]: "on" value {on!r} is not a valid output node id'
                f" (valid: {sorted(output_ids)})"
            )
        if not isinstance(entry.get("attribute"), str) or not entry["attribute"].strip():
            errors.append(f'attributes[{i}]: "attribute" must be a non-empty string')
    return errors


class OutputDesignator:
    """Identifies which literal attributes a question asks to return for each output entity.

    Attributes:
        llm_client (LLMCallable):
            Callable (prompt: str) -> str. Defaults to OllamaLLM.
    """

    def __init__(self, llm_client: LLMCallable | None = None) -> None:
        """Initializes the attributes of the class.

        Parameters:
            llm_client (LLMCallable):
                Callable (prompt: str) -> str. Defaults to OllamaLLM.
        """
        self._llm = llm_client or OllamaLLM()

    def designate(self, question: str, output_nodes: list[dict]) -> dict:
        """Identify the literal attributes the question requests for each output node.

        Parameters:
            question (str):
                The natural-language question.
            output_nodes (list[dict]):
                List of output node dicts, each with keys:
                  - node_id (str): unique identifier for this node.
                  - type    (str): entity type (e.g. "gene", "organism").
                  - span    (str): optional natural-language surface form.

        Returns:
            On success, a dict {"attributes": [{"on": <node_id>, "attribute": <name>}, ...]}.
            On failure after all retries, {"error": "designation_failed"}.
        """
        output_ids = {n["node_id"] for n in output_nodes}
        prompt = build_designation_prompt(question, output_nodes)
        result = llm_call_with_retry(
            self._llm,
            prompt,
            lambda p: _validate(p, output_ids),
            OUTPUT_DESIGNATION_MAX_RETRIES,
        )
        if result is None:
            return {"error": "designation_failed"}
        return result
