"""SPARQL generator: the LLM authors a query from a code-prepared assembly brief.

Division of labour (mirrors the spec):
  Code — resolves every fact (predicates, graphs, IRIs) into the brief, and
         validates the authored query by executing it (Step "validation loop").
  LLM  — owns structure: assembles the brief's edges into GRAPH blocks, adds
         missing output/attribute/anchor triples, materializes groundings, and
         drops anything the brief doesn't actually need.

The LLM client is an injectable callable (str -> str) so any backend works.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests
import urllib3

from ..config import (
    BIOGATEWAY_SPARQL_ENDPOINT,
    SPARQL_GENERATOR_MAX_REPAIR,
    SPARQL_GENERATOR_OFFLINE_MODE,
    SPARQL_GENERATOR_PROMPT,
    SPARQL_TIMEOUT,
    SPARQL_VERIFY_SSL,
)
from ..entity_linker.label_config import LabelConfig, build_label_config
from ..utils.llm import LLMCallable, OllamaLLM
from ..utils.prompts_loader import render_prompt
from .attribute_resolver import AttributeConfig, build_attribute_config
from .brief_builder import build_brief

if not SPARQL_VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_DEFAULT_EXCEL = Path(__file__).resolve().parents[2] / "data" / "BGW_graphs.xlsx"


def extract_sparql(text: str) -> str:
    """Extract the SPARQL query text from raw LLM output, tolerating code fences.

    Parameters:
        text (str):
            Raw LLM completion, expected to be plain SPARQL, possibly wrapped
            in a ```sparql ... ``` or ``` ... ``` fence.

    Returns:
        The SPARQL query text with any surrounding fence removed.
    """
    text = text.strip()
    fence = re.search(r"```(?:sparql)?\s*(.*?)\s*```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text


def build_generation_prompt(brief: dict) -> str:
    """Render the SPARQL-generation prompt for a given assembly brief.

    Parameters:
        brief (dict):
            The assembly brief built by brief_builder.build_brief().

    Returns:
        The fully rendered prompt string ready to send to the LLM.
    """
    return render_prompt(
        SPARQL_GENERATOR_PROMPT,
        prompts_dir=_PROMPTS_DIR,
        BRIEF=json.dumps(brief, indent=2),
    )


def _build_repair_prompt(original_prompt: str, query: str, error: str) -> str:
    """Append the failed query and its error to the original prompt for a repair attempt.

    Parameters:
        original_prompt (str):
            The prompt that produced the invalid query.
        query (str):
            The SPARQL query that failed to parse or execute.
        error (str):
            The error message returned by the endpoint.

    Returns:
        A new prompt asking the LLM to fix the query's structure/syntax.
    """
    return (
        f"{original_prompt}\n\n"
        f"Your previous answer was:\n{query}\n\n"
        f"Running that query against the endpoint failed with this error:\n{error}\n\n"
        f"Fix the query's structure/syntax so it executes. Do not change the facts "
        f"(predicates/graphs/IRIs) — only fix how they're assembled. Output ONLY the "
        f"corrected SPARQL query."
    )


def _validate_live(
    query: str,
    endpoint: str = BIOGATEWAY_SPARQL_ENDPOINT,
    timeout: int = SPARQL_TIMEOUT,
    verify_ssl: bool = SPARQL_VERIFY_SSL,
) -> str | None:
    """Execute a SPARQL query against the endpoint to check it parses and runs.

    Parameters:
        query (str):
            The SPARQL query to validate.
        endpoint (str):
            SPARQL endpoint URL. Defaults to BIOGATEWAY_SPARQL_ENDPOINT.
        timeout (int):
            Request timeout in seconds. Defaults to SPARQL_TIMEOUT.
        verify_ssl (bool):
            Whether to verify SSL certificates. Defaults to SPARQL_VERIFY_SSL.

    Returns:
        None if the query executed successfully, otherwise an error message.
    """
    try:
        resp = requests.post(
            endpoint,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
            verify=verify_ssl,
        )
        resp.raise_for_status()
        resp.json()
    except Exception as exc:  # noqa: BLE001 — any failure is a validation error to report back
        return str(exc)
    return None


class SPARQLGenerator:
    """Authors a SPARQL query from an assembly brief, with a live-validation repair loop.

    Attributes:
        llm_client (LLMCallable):
            Callable (prompt: str) -> str. Defaults to OllamaLLM.
        excel_path (Path):
            Path to BGW_graphs.xlsx for label/attribute config derivation.
            If None, uses data/BGW_graphs.xlsx relative to the project root.
        max_repair (int):
            Maximum number of repair attempts after a failed live validation.
        offline (bool):
            When True, skip live validation entirely (no network calls).
    """

    def __init__(
        self,
        llm_client: LLMCallable | None = None,
        excel_path: Path | None = None,
        max_repair: int = SPARQL_GENERATOR_MAX_REPAIR,
        offline: bool = SPARQL_GENERATOR_OFFLINE_MODE,
    ) -> None:
        """Initializes the attributes of the class.

        Parameters:
            llm_client (LLMCallable):
                Callable (prompt: str) -> str. Defaults to OllamaLLM.
            excel_path (Path):
                Path to BGW_graphs.xlsx. If None, the default project path is used.
            max_repair (int):
                Maximum number of repair attempts after a failed live validation.
            offline (bool):
                When True, skip live validation entirely (no network calls).
        """
        self._llm = llm_client or OllamaLLM()
        self._max_repair = max_repair
        self._offline = offline

        xlsx = excel_path or _DEFAULT_EXCEL
        self._label_cfg: LabelConfig = build_label_config(xlsx) if xlsx.exists() else {}
        self._attr_cfg: AttributeConfig = build_attribute_config(xlsx) if xlsx.exists() else {}

    def generate(
        self,
        question: str,
        extraction: dict,
        planner_result: dict,
        linker_input: list[dict],
        linker_output: list[dict],
        designation: dict,
    ) -> dict:
        """Build the assembly brief and author a SPARQL query for it.

        Parameters:
            question (str):
                The natural-language question.
            extraction (dict):
                Concept-extraction result: {"outputs", "nodes", "triples"}.
            planner_result (dict):
                PathPlanner.resolve_relations() result.
            linker_input (list[dict]):
                The span nodes passed to EntityLinker.link().
            linker_output (list[dict]):
                The corresponding EntityLinker.link() results.
            designation (dict):
                Output-designation result: {"attributes": [...]}.

        Returns:
            A dict with keys:
              "query"    — the authored SPARQL query (str).
              "brief"    — the assembly brief that was sent to the LLM.
              "warnings" — brief-assembly warnings (fail-soft drops), list[str].
              "error"    — present only if live validation never succeeded.
        """
        brief, warnings = build_brief(
            question,
            extraction,
            planner_result,
            linker_input,
            linker_output,
            designation,
            label_cfg=self._label_cfg,
            attr_cfg=self._attr_cfg,
        )

        prompt = build_generation_prompt(brief)
        raw = self._llm(prompt)
        query = extract_sparql(raw)

        if self._offline:
            return {"query": query, "brief": brief, "warnings": warnings}

        error = _validate_live(query)
        attempt = 0
        while error is not None and attempt < self._max_repair:
            raw = self._llm(_build_repair_prompt(prompt, query, error))
            query = extract_sparql(raw)
            error = _validate_live(query)
            attempt += 1

        result = {"query": query, "brief": brief, "warnings": warnings}
        if error is not None:
            result["error"] = error
        return result
