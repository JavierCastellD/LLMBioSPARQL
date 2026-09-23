"""Ablation baseline: one LLM call authors SPARQL directly from the question.

No concept extraction, output designation, entity linking, or path planning
-- the LLM sees the raw question and a flat schema dump, and must resolve
concepts, relations, and entity grounding entirely on its own. Everything
else (the model, the live-validation/repair loop, the endpoint) is shared
with the structured pipeline's SPARQLGenerator so the comparison isolates
the effect of the intermediate stages, not incidental differences.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests
import urllib3

from ..config import (
    BIOGATEWAY_SPARQL_ENDPOINT,
    DIRECT_GENERATOR_FEWSHOT_FILE,
    DIRECT_GENERATOR_MAX_REPAIR,
    DIRECT_GENERATOR_PROMPT,
    SPARQL_TIMEOUT,
    SPARQL_VERIFY_SSL,
)
from ..utils.llm import LLMCallable, OllamaLLM
from ..utils.prompts_loader import render_prompt
from .schema_context import build_prefix_block, build_schema_text

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


def _build_fewshot_block(fewshot_path: Path) -> str:
    """Render the question/SPARQL few-shot examples used to prompt the LLM.

    Parameters:
        fewshot_path (Path):
            JSON file with a list of {"question", "sparql", ...} objects
            (only these two keys are used).

    Returns:
        One "Question: ...\\nSPARQL:\\n..." block per example, blank-line
        separated. Empty string if fewshot_path does not exist.
    """
    if not fewshot_path.exists():
        return ""
    examples = json.loads(fewshot_path.read_text(encoding="utf-8"))
    blocks = [f"Question: {ex['question']}\nSPARQL:\n{ex['sparql']}" for ex in examples]
    return "\n\n".join(blocks)


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
        A new prompt asking the LLM to fix the query.
    """
    return (
        f"{original_prompt}\n\n"
        f"Your previous answer was:\n{query}\n\n"
        f"Running that query against the endpoint failed with this error:\n{error}\n\n"
        f"Fix the query so it executes. Output ONLY the corrected SPARQL query."
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


class DirectSPARQLGenerator:
    """Authors a SPARQL query straight from a question, with a live-validation repair loop.

    Attributes:
        llm_client (LLMCallable):
            Callable (prompt: str) -> str. Defaults to OllamaLLM.
        excel_path (Path):
            Path to BGW_graphs.xlsx, used only to render the schema text.
        fewshot_path (Path):
            JSON file of {"question", "sparql"} few-shot examples.
        max_repair (int):
            Maximum number of repair attempts after a failed live validation.
    """

    def __init__(
        self,
        llm_client: LLMCallable | None = None,
        excel_path: Path | None = None,
        fewshot_path: Path | None = None,
        max_repair: int = DIRECT_GENERATOR_MAX_REPAIR,
    ) -> None:
        """Initializes the attributes of the class and pre-renders the static preamble.

        Parameters:
            llm_client (LLMCallable):
                Callable (prompt: str) -> str. Defaults to OllamaLLM.
            excel_path (Path):
                Path to BGW_graphs.xlsx. If None, the default project path is used.
            fewshot_path (Path):
                Path to the few-shot JSON file. If None, DIRECT_GENERATOR_FEWSHOT_FILE
                (project-relative) is used.
            max_repair (int):
                Maximum number of repair attempts after a failed live validation.
        """
        self._llm = llm_client or OllamaLLM()
        self._max_repair = max_repair

        xlsx = excel_path or _DEFAULT_EXCEL
        fewshot = fewshot_path or (Path(__file__).resolve().parents[2] / DIRECT_GENERATOR_FEWSHOT_FILE)

        self._schema_text = build_schema_text(xlsx) if xlsx.exists() else ""
        self._prefix_block = build_prefix_block()
        self._fewshot_block = _build_fewshot_block(fewshot)

    def generate(self, question: str) -> dict:
        """Author a SPARQL query directly from the question, no intermediate stages.

        Parameters:
            question (str):
                The natural-language question to convert to SPARQL.

        Returns:
            A dict with keys:
              "query"  — the authored SPARQL query (str).
              "prompt" — the initial prompt sent to the LLM, for inspection.
              "error"  — present only if live validation never succeeded.
        """
        prompt = render_prompt(
            DIRECT_GENERATOR_PROMPT,
            prompts_dir=_PROMPTS_DIR,
            PREFIXES=self._prefix_block,
            SCHEMA=self._schema_text,
            EXAMPLES=self._fewshot_block,
            QUESTION=question,
        )
        raw = self._llm(prompt)
        query = extract_sparql(raw)

        error = _validate_live(query)
        attempt = 0
        while error is not None and attempt < self._max_repair:
            raw = self._llm(_build_repair_prompt(prompt, query, error))
            query = extract_sparql(raw)
            error = _validate_live(query)
            attempt += 1

        result = {"query": query, "prompt": prompt}
        if error is not None:
            result["error"] = error
        return result
