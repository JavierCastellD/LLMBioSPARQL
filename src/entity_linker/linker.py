"""Entity linker: ground concept-extraction spans to BioGateway URIs/filters.

Division of labour (mirrors the spec):
  LLM  — proposes search terms and predicates (Steps 1 & 3).
  Code — runs SPARQL probes and branches deterministically (Step 2).

The LLM client is an injectable callable (str -> str) so any backend works.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import (
    LINKER_MAX_ITERATIONS,
    LINKER_PROPOSAL_PROMPT,
    LINKER_REFORMULATION_PROMPT,
    LLM_CONFIRM_SINGLE_MATCH,
)
from ..utils.llm import LLMCallable, OllamaLLM, extract_json, llm_call_with_retry
from ..utils.prompts_loader import render_prompt
from .label_config import LabelConfig, build_label_config, predicate_label_menu, get_predicates
from .sparql_probe import probe as sparql_probe

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_DEFAULT_EXCEL = Path(__file__).resolve().parents[2] / "data" / "BGW_graphs.xlsx"

TYPE_TO_GRAPH: dict[str, str] = {
    "organism":           "taxon",
    "biological process": "go",
    "molecular function": "go",
    "cellular component": "go",
    "phenotype":          "omim",
    "gene":               "gene",
    "protein":            "prot",
    "CRM":                "crm",
    "tad":                "tad",
    "chromosome":         "gene",
    "genome assembly":    "gene",
}


def build_proposal_prompt(spans_payload: list[dict]) -> str:
    """Render the batch proposal prompt for a list of span descriptors.

    Parameters:
        spans_payload (list[dict]):
            List of dicts, each with node_id, span, type, graph, and available_predicates.

    Returns:
        The fully rendered prompt string ready to send to the LLM.
    """
    return render_prompt(
        LINKER_PROPOSAL_PROMPT,
        prompts_dir=_PROMPTS_DIR,
        SPANS_JSON=json.dumps(spans_payload, indent=2),
    )


def build_reformulation_prompt(node: dict, prev_proposal: dict, menu: list[str]) -> str:
    """Render the zero-result reformulation prompt for a single span.

    Parameters:
        node (dict):
            Span node dict with keys span, type, and graph.
        prev_proposal (dict):
            The previous proposal that yielded zero SPARQL results,
            with keys search_term, mode, and predicates.
        menu (list[str]):
            Friendly predicate names available for the node's graph.

    Returns:
        The fully rendered prompt string ready to send to the LLM.
    """
    return render_prompt(
        LINKER_REFORMULATION_PROMPT,
        prompts_dir=_PROMPTS_DIR,
        SPAN=node["span"],
        TYPE=node["type"],
        GRAPH=node["graph"],
        PREV_TERM=prev_proposal.get("search_term", ""),
        PREV_MODE=prev_proposal.get("mode", "exact"),
        PREV_PREDS=json.dumps(prev_proposal.get("predicates", [])),
        AVAILABLE_PREDS=json.dumps(menu),
    )


# ---------------------------------------------------------------------------
# Proposal validation — checked after every LLM call
# ---------------------------------------------------------------------------

_VALID_MODES = {"exact", "regex"}


def _validate_proposals(parsed, expected_ids: set[str]) -> list[str]:
    """Validate a batch-proposal LLM response.

    Parameters:
        parsed (Any):
            The JSON value returned by extract_json (possibly None).
        expected_ids (set[str]):
            The set of node_ids that must appear in the response.

    Returns:
        A list of human-readable problems found, or an empty list if the response is valid.
    """
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]
    proposals = parsed.get("proposals")
    if not isinstance(proposals, list):
        return ['"proposals" must be a list']
    errors: list[str] = []
    seen_ids: set[str] = set()
    for i, p in enumerate(proposals):
        if not isinstance(p, dict):
            errors.append(f"proposals[{i}] is not an object")
            continue
        nid = p.get("node_id")
        if nid not in expected_ids:
            errors.append(f"proposals[{i}]: node_id {nid!r} not in expected ids")
        seen_ids.add(nid)
        if p.get("mode") not in _VALID_MODES:
            errors.append(f"proposals[{i}]: mode must be 'exact' or 'regex'")
        if not isinstance(p.get("predicates"), list) or not p["predicates"]:
            errors.append(f"proposals[{i}]: predicates must be a non-empty list")
        if not isinstance(p.get("search_term"), str) or not p["search_term"].strip():
            errors.append(f"proposals[{i}]: search_term must be a non-empty string")
    missing = expected_ids - seen_ids
    if missing:
        errors.append(f"proposals missing for node_ids: {sorted(missing)}")
    return errors


def _validate_single_proposal(parsed) -> list[str]:
    """Validate a single reformulation proposal from the LLM.

    Parameters:
        parsed (Any):
            The JSON value returned by extract_json (possibly None).

    Returns:
        A list of human-readable problems found, or an empty list if the response is valid.
    """
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]
    errors: list[str] = []
    if parsed.get("mode") not in _VALID_MODES:
        errors.append("mode must be 'exact' or 'regex'")
    if not isinstance(parsed.get("predicates"), list) or not parsed["predicates"]:
        errors.append("predicates must be a non-empty list")
    if not isinstance(parsed.get("search_term"), str) or not parsed["search_term"].strip():
        errors.append("search_term must be a non-empty string")
    return errors


# ---------------------------------------------------------------------------
# EntityLinker
# ---------------------------------------------------------------------------

class EntityLinker:
    """Ground concept-extraction spans to BioGateway URIs or regex filters.

    Attributes:
        llm_client (LLMCallable):
            Callable (prompt: str) -> str. Defaults to OllamaLLM.
        excel_path (Path):
            Path to BGW_graphs.xlsx for label-predicate config derivation.
            If None, uses data/BGW_graphs.xlsx relative to the project root.
        max_iterations (int):
            Maximum number of reformulation attempts per zero-result span.
        llm_confirm_single (bool):
            When True, a single SPARQL match triggers an extra LLM call to confirm
            it is the right entity before binding. False (default) binds automatically.
    """

    def __init__(
        self,
        llm_client: LLMCallable | None = None,
        excel_path: Path | None = None,
        max_iterations: int = LINKER_MAX_ITERATIONS,
        llm_confirm_single: bool = LLM_CONFIRM_SINGLE_MATCH,
    ) -> None:
        """Initializes the attributes of the class.

        Parameters:
            llm_client (LLMCallable):
                Callable (prompt: str) -> str. Defaults to OllamaLLM.
            excel_path (Path):
                Path to BGW_graphs.xlsx. If None, the default project path is used.
            max_iterations (int):
                Maximum number of reformulation attempts per zero-result span.
            llm_confirm_single (bool):
                When True, a single SPARQL match is confirmed by the LLM before binding.
        """
        self._llm = llm_client or OllamaLLM()
        self._max_iter = max_iterations
        self._llm_confirm_single = llm_confirm_single

        xlsx = excel_path or _DEFAULT_EXCEL
        self._label_cfg: LabelConfig = build_label_config(xlsx) if xlsx.exists() else {}

        # Probe results are cached per (graph, term, mode, predicates) within the session.
        self._cache: dict[tuple, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def link(
        self,
        span_nodes: list[dict],
        proposals_override: dict[str, dict] | None = None,
    ) -> list[dict]:
        """Link a list of spans to BioGateway entities.

        Parameters:
            span_nodes (list[dict]):
                List of dicts, each with keys:
                  - node_id (str): unique identifier for this span.
                  - span    (str): the natural-language surface form.
                  - type    (str): entity type (e.g. "organism", "gene").
            proposals_override (dict[str, dict]):
                Optional mapping of node_id to proposal dict, used to skip
                the LLM proposal step in tests and validation scripts.

        Returns:
            A list of result dicts, one per input span_node, in the same order.
        """
        if not span_nodes:
            return []

        # Derive the target graph from the entity type.
        nodes = [
            {**n, "graph": TYPE_TO_GRAPH[n["type"]]}
            for n in span_nodes
        ]

        # Step 1 — get LLM proposals for all spans in a single call (or use the override).
        if proposals_override is not None:
            proposals = proposals_override
        else:
            proposals = self._batch_propose(nodes)

        # Step 2 — run a SPARQL probe for each span and branch on the result count.
        results: list[dict] = []
        for node in nodes:
            nid = node["node_id"]
            proposal = proposals.get(nid) or _fallback_proposal(node["span"])
            result = self._probe_and_branch(node, proposal)
            results.append(result)

        # Step 3 — reformulate only the spans that returned zero results.
        for i, result in enumerate(results):
            if result["mode"] != "_zero":
                continue
            node = nodes[i]
            proposal = proposals.get(node["node_id"]) or _fallback_proposal(node["span"])
            results[i] = self._reformulate_loop(node, proposal)

        # Convert the internal "_zero" sentinel (exhausted reformulation) to "unresolved".
        for r in results:
            if r["mode"] == "_zero":
                r["mode"] = "unresolved"

        return results

    # ------------------------------------------------------------------
    # Step 1: batch LLM proposal
    # ------------------------------------------------------------------

    def _batch_propose(self, nodes: list[dict]) -> dict[str, dict]:
        """Call the LLM once to get search-term proposals for all spans.

        Parameters:
            nodes (list[dict]):
                Span node dicts with graph names already normalised to short form.

        Returns:
            A mapping of node_id to proposal dict. Falls back to _fallback_proposal
            for all spans if all LLM retries fail.
        """
        # Build the payload shown to the LLM: span info + available predicates per graph.
        spans_payload = [
            {
                "node_id": n["node_id"],
                "span": n["span"],
                "type": n["type"],
                "graph": n["graph"],
                "available_predicates": predicate_label_menu(self._label_cfg, n["graph"]),
            }
            for n in nodes
        ]
        prompt = build_proposal_prompt(spans_payload)

        expected_ids = {n["node_id"] for n in nodes}
        parsed = llm_call_with_retry(self._llm, prompt, lambda p: _validate_proposals(p, expected_ids))
        if parsed is not None:
            return {p["node_id"]: p for p in parsed["proposals"]}

        # All retries failed; fall back to searching each span as-is.
        return {n["node_id"]: _fallback_proposal(n["span"]) for n in nodes}

    # ------------------------------------------------------------------
    # Step 2: probe + branch
    # ------------------------------------------------------------------

    def _probe_and_branch(self, node: dict, proposal: dict) -> dict:
        """Run a SPARQL probe and branch on the distinct-URI count.

        Parameters:
            node (dict):
                Span node dict with keys span, type, and graph.
            proposal (dict):
                LLM proposal with keys search_term, mode, and predicates.

        Returns:
            A result dict with one of four modes:
              "bind"   — single URI; keys: uri, term, label_predicate.
              "many"   — multiple URIs, exact search; keys: term, label_predicate.
              "filter" — multiple URIs, regex search; keys: term, predicates.
              "_zero"  — internal sentinel that triggers reformulation.
        """
        span = node["span"]
        graph = node["graph"]
        term = proposal["search_term"]
        mode = proposal.get("mode", "exact")
        predicate_labels = proposal.get("predicates", ["preferred label"])

        # Resolve predicate label names (e.g. "synonyms") to full IRIs.
        pred_iris = get_predicates(self._label_cfg, graph, predicate_labels)
        if not pred_iris:
            pred_iris = ["http://www.w3.org/2004/02/skos/core#prefLabel"]

        # Cache probes within the session to avoid redundant HTTP requests.
        cache_key = (graph, term, mode, frozenset(pred_iris))
        if cache_key not in self._cache:
            try:
                self._cache[cache_key] = sparql_probe(graph, pred_iris, term, mode)
            except Exception as exc:
                return {
                    "span": span,
                    "mode": "unresolved",
                    "candidates": 0,
                    "error": str(exc),
                }

        result = self._cache[cache_key]
        count = result["count"]
        matches = result["matches"]

        if count == 0:
            # Return a sentinel so the caller knows to attempt reformulation.
            return {
                "span": span,
                "mode": "_zero",
                "candidates": 0,
                "_last_term": term,
                "_last_mode": mode,
                "_last_preds": predicate_labels,
                "_last_iris": pred_iris,
            }

        if count == 1:
            match = matches[0]
            if self._llm_confirm_single:
                # Optional LLM confirmation before committing the binding.
                if not self._sanity_check(span, match):
                    return {"span": span, "mode": "unresolved", "candidates": 1}
            return {
                "span": span,
                "mode": "bind",
                "uri": match["uri"],
                "term": match["label"],
                "label_predicate": match["predicate"],
                "candidates": 1,
            }

        if mode != "regex":
            # Many exact matches: the search was over a single label predicate.
            return {
                "span": span,
                "mode": "many",
                "term": term,
                "label_predicate": pred_iris[0],
                "candidates": count,
            }

        # Regex search with many matches: pass all searched predicates to the SPARQL generator.
        return {
            "span": span,
            "mode": "filter",
            "term": term,
            "predicates": pred_iris,
            "candidates": count,
        }

    # ------------------------------------------------------------------
    # Step 3: reformulation loop
    # ------------------------------------------------------------------

    def _reformulate_loop(self, node: dict, initial_proposal: dict) -> dict:
        """Ask the LLM for a new search term up to max_iterations times for a zero-result span.

        Parameters:
            node (dict):
                Span node dict with keys span, type, and graph.
            initial_proposal (dict):
                The proposal that yielded zero results and triggered the loop.

        Returns:
            The first non-zero result dict from _probe_and_branch, or an
            "unresolved" result if all iterations are exhausted.
        """
        span = node["span"]
        proposal = initial_proposal

        for _iter in range(self._max_iter):
            new_proposal = self._llm_reformulate(node, proposal)
            result = self._probe_and_branch(node, new_proposal)
            if result["mode"] != "_zero":
                return result
            # Carry the new proposal forward so the LLM sees what was already tried.
            proposal = new_proposal

        return {"span": span, "mode": "unresolved", "candidates": 0}

    def _llm_reformulate(self, node: dict, prev_proposal: dict) -> dict:
        """Ask the LLM to propose a new search term after a zero-result probe.

        Parameters:
            node (dict):
                Span node dict with keys span, type, and graph.
            prev_proposal (dict):
                The previous proposal that yielded zero results.

        Returns:
            A new proposal dict with keys search_term, mode, and predicates.
            Falls back to _fallback_proposal if all LLM retries fail.
        """
        menu = predicate_label_menu(self._label_cfg, node["graph"])
        prompt = build_reformulation_prompt(node, prev_proposal, menu)
        parsed = llm_call_with_retry(self._llm, prompt, _validate_single_proposal)
        return parsed if parsed is not None else _fallback_proposal(node["span"])

    # ------------------------------------------------------------------
    # Optional LLM confirmation (only active when llm_confirm_single=True)
    # ------------------------------------------------------------------

    def _sanity_check(self, span: str, match: dict) -> bool:
        """Ask the LLM whether the single SPARQL match is the intended entity.

        Parameters:
            span (str):
                The natural-language surface form of the entity.
            match (dict):
                The single SPARQL match with keys uri, label, and predicate.

        Returns:
            True to approve the binding, False to reject it. Defaults to True
            if the LLM response cannot be parsed.
        """
        prompt = (
            f'Is the following database entity the correct match for the span "{span}"?\n\n'
            f'  URI:   {match["uri"]}\n'
            f'  Label: {match["label"]}\n\n'
            f'Answer with a JSON object: {{"accept": true}} or {{"accept": false}}.'
        )
        raw = self._llm(prompt)
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            return bool(parsed.get("accept", True))
        return True  # default to accepting if the response can't be parsed


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _fallback_proposal(span: str) -> dict:
    """Build a last-resort proposal that searches for the raw span text.

    Parameters:
        span (str):
            The natural-language surface form to search for.

    Returns:
        A proposal dict with search_term set to the span and predicates covering
        the three most common label categories.
    """
    return {
        "search_term": span,
        "mode": "exact",
        "predicates": ["preferred label", "label", "synonyms"],
    }
