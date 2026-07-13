"""SPARQL probe: case-insensitive entity lookup against a BioGateway named graph.

Supports two match modes:
  "exact"  — anchored regex (^TERM$, case-insensitive)
  "regex"  — pattern passed as-is, case-insensitive flag added

Returns a plain dict so it is easy to cache and test without live network.
"""
from __future__ import annotations

import re
import warnings

import requests
import urllib3

from ..config import (
    BIOGATEWAY_GRAPH_BASE,
    BIOGATEWAY_SPARQL_ENDPOINT,
    SPARQL_TIMEOUT,
    SPARQL_VERIFY_SSL,
)

if not SPARQL_VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 20 rows is enough to distinguish 0 / 1 / many distinct URIs.
_LIMIT = 20


def _graph_iri(graph_name: str) -> str:
    """Return the full IRI for a short graph name, or pass a full IRI through unchanged.

    Parameters:
        graph_name (str):
            Short graph name (e.g. "taxon") or a full IRI.

    Returns:
        The full graph IRI.
    """
    if graph_name.startswith("http"):
        return graph_name
    return BIOGATEWAY_GRAPH_BASE + graph_name


def _sparql_string_escape(s: str) -> str:
    """Escape backslashes and quotes so the string is safe inside a SPARQL double-quoted literal.

    Parameters:
        s (str):
            The raw string to escape.

    Returns:
        The escaped string safe for embedding in a SPARQL double-quoted literal.
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_query(graph_iri: str, predicate_iris: list[str], term: str, mode: str) -> str:
    """Build the SPARQL SELECT that retrieves up to _LIMIT matching (uri, pred, label) rows.

    Parameters:
        graph_iri (str):
            Full IRI of the named graph to query.
        predicate_iris (list[str]):
            List of predicate IRIs to filter on.
        term (str):
            Search term or regex pattern depending on mode.
        mode (str):
            Either "exact" (term is escaped and anchored) or "regex" (term is used as-is).

    Returns:
        A SPARQL SELECT query string.
    """
    preds = ", ".join(f"<{p}>" for p in predicate_iris)

    if mode == "exact":
        # Escape regex metacharacters in the literal term, then anchor it.
        pattern = "^" + re.escape(term) + "$"
        filter_str = f'FILTER(REGEX(STR(?label), "{_sparql_string_escape(pattern)}", "i"))'
    else:
        # The LLM already produced a regex pattern; just pass it through.
        filter_str = f'FILTER(REGEX(STR(?label), "{_sparql_string_escape(term)}", "i"))'

    return (
        f"SELECT DISTINCT ?uri ?pred ?label WHERE {{\n"
        f"  GRAPH <{graph_iri}> {{\n"
        f"    ?uri ?pred ?label .\n"
        f"    FILTER(?pred IN ({preds}))\n"
        f"    {filter_str}\n"
        f"  }}\n"
        f"}} LIMIT {_LIMIT}"
    )


def probe(
    graph: str,
    predicate_iris: list[str],
    term: str,
    mode: str = "exact",
    endpoint: str = BIOGATEWAY_SPARQL_ENDPOINT,
    timeout: int = SPARQL_TIMEOUT,
    verify_ssl: bool = SPARQL_VERIFY_SSL,
) -> dict:
    """Run a SPARQL probe against a BioGateway named graph and return result metadata.

    Parameters:
        graph (str):
            Short graph name (e.g. "taxon") or full graph IRI.
        predicate_iris (list[str]):
            Predicate IRIs to search across.
        term (str):
            Search term or regex pattern depending on mode.
        mode (str):
            Either "exact" (anchored, escaped regex) or "regex" (LLM-provided pattern).
        endpoint (str):
            SPARQL endpoint URL. Defaults to BIOGATEWAY_SPARQL_ENDPOINT.
        timeout (int):
            Request timeout in seconds. Defaults to SPARQL_TIMEOUT.
        verify_ssl (bool):
            Whether to verify SSL certificates. Defaults to SPARQL_VERIFY_SSL.

    Returns:
        A dict with keys "count" (number of distinct URIs found), "matches" (list of
        dicts with uri, label, and predicate), and "query" (the SPARQL query string).
        Raises requests.HTTPError on HTTP errors.
    """
    full_iri = _graph_iri(graph)
    query = _build_query(full_iri, predicate_iris, term, mode)

    resp = requests.post(
        endpoint,
        data={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=timeout,
        verify=verify_ssl,
    )
    resp.raise_for_status()

    bindings = resp.json()["results"]["bindings"]

    # Deduplicate by URI so that one entity matching via several predicates still counts as 1.
    seen_uris: set[str] = set()
    matches: list[dict] = []
    for b in bindings:
        uri = b["uri"]["value"] if "uri" in b else ""
        label = b["label"]["value"] if "label" in b else ""
        pred = b["pred"]["value"] if "pred" in b else ""
        if uri not in seen_uris:
            seen_uris.add(uri)
            matches.append({"uri": uri, "label": label, "predicate": pred})

    return {"count": len(seen_uris), "matches": matches, "query": query}
