"""Derive per-graph label-predicate config from BGW_graphs.xlsx.

For each named graph, scans its spreadsheet sheet for literal-typed predicates
whose property label contains "name", "label", "synonym", or "title", and maps
them to the friendly names exposed to the LLM:
  "preferred label", "label", "synonym", "exact synonym", "broad synonym",
  "narrow synonym", "related synonym", "title".
"""
from __future__ import annotations

import re
from pathlib import Path

import openpyxl

# Fallback predicate set for graphs not found in the spreadsheet.
FALLBACK = {
    "preferred label": ["http://www.w3.org/2004/02/skos/core#prefLabel"],
    "label":           ["http://www.w3.org/2000/01/rdf-schema#label"],
    "synonym":         ["http://www.w3.org/2004/02/skos/core#altLabel"],
    "exact synonym":   ["http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"],
    "broad synonym":   ["http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym"],
    "narrow synonym":  ["http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym"],
    "related synonym": ["http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"],
}

# Type alias: graph_name -> {friendly_name -> [predicate_iri, ...]}
LabelConfig = dict[str, dict[str, list[str]]]


def _friendly_name(prop_label: str) -> str | None:
    """Map a property label string to a predicate label category.

    Detection order avoids overlap: "synonym" before "name"/"label".
    Uses whole-word matching for "name" so that "namespace" is not confused
    with a label predicate.

    Parameters:
        prop_label (str):
            The property label string from the spreadsheet.

    Returns:
        One of "synonyms", "preferred label", "label", "title", or None if
        the label does not match any known category.
    """
    lbl = prop_label.lower()
    if "synonym" in lbl:
        if "exact" in lbl:
            return "exact synonym"
        if "broad" in lbl:
            return "broad synonym"
        if "narrow" in lbl:
            return "narrow synonym"
        if "related" in lbl:
            return "related synonym"
        return "synonym"
    # \bname\b matches "has name" but NOT "namespace" or "obo_namespace"
    if re.search(r"\bname\b", lbl):
        return "preferred label"
    if "label" in lbl:
        return "label"
    if "title" in lbl:
        return "title"
    return None


def graph_names_from_classification(wb) -> list[str]:
    """Collect short graph names from the workbook's 'graphs' classification sheet.

    Parameters:
        wb: An openpyxl Workbook for BGW_graphs.xlsx.

    Returns:
        Short graph names (e.g. "gene", "taxon") for every Entity/Relation graph.
    """
    graph_names: list[str] = []
    for row in wb["graphs"].iter_rows(min_row=2, values_only=True):
        iri, gtype = row[0], row[1]
        if iri and gtype and gtype in ("Entity graph", "Relation graph"):
            name = str(iri).rstrip("/").split("/")[-1]
            graph_names.append(name)
    return graph_names


def find_graph_sheet(wb, graph_name: str):
    """Locate a graph's worksheet, trying both spreadsheet naming conventions.

    Parameters:
        wb: An openpyxl Workbook for BGW_graphs.xlsx.
        graph_name (str):
            Short graph name (e.g. "gene", "taxon").

    Returns:
        The matching Worksheet, or None if neither "<name> graph" nor "<name>"
        exists as a sheet.
    """
    for candidate in (f"{graph_name} graph", graph_name):
        if candidate in wb.sheetnames:
            return wb[candidate]
    return None


def build_label_config(excel_path: Path) -> LabelConfig:
    """Parse BGW_graphs.xlsx and return the per-graph label-predicate mapping.

    Only considers rows where the object node type is "literal". Unknown graphs
    fall back to FALLBACK at query time (see get_predicates()).

    Parameters:
        excel_path (Path):
            Path to the BGW_graphs.xlsx spreadsheet.

    Returns:
        A mapping of graph name to predicate label name to list of predicate IRIs.
    """
    wb = openpyxl.load_workbook(excel_path)

    config: LabelConfig = {}
    for graph_name in graph_names_from_classification(wb):
        ws = find_graph_sheet(wb, graph_name)
        if ws is None:
            continue

        predicates: dict[str, list[str]] = {}
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue  # skip header row
            pred_iri, prop_label, obj_node_type = row[0], row[1], row[2]
            if pred_iri is None or prop_label is None:
                continue
            # Only literal-valued predicates can carry entity labels.
            if str(obj_node_type).lower() != "literal":
                continue

            fname = _friendly_name(str(prop_label))
            if fname is None:
                continue

            pred_str = str(pred_iri)
            bucket = predicates.setdefault(fname, [])
            if pred_str not in bucket:
                bucket.append(pred_str)

        if predicates:
            config[graph_name] = predicates

    return config


def get_predicates(
    config: LabelConfig,
    graph_name: str,
    predicate_labels: list[str],
) -> list[str]:
    """Resolve predicate label names to IRIs for a given graph.

    Falls back to FALLBACK only when the graph itself is entirely unknown (not
    present in config). For a known graph, a friendly name it doesn't have
    resolves to no IRIs — it does NOT borrow an unrelated graph's default,
    since callers (e.g. brief_builder's name-predicate probing) rely on an
    empty result to mean "this graph doesn't have this" and try the next
    option themselves. Never raises.

    Parameters:
        config (LabelConfig):
            The per-graph label-predicate mapping built by build_label_config().
        graph_name (str):
            Short name of the target graph (e.g. "taxon", "gene").
        predicate_labels (list[str]):
            Friendly predicate category names to resolve (e.g. ["preferred label", "synonyms"]).

    Returns:
        A deduplicated list of predicate IRIs in the order of predicate_labels.
    """
    graph_cfg = config.get(graph_name, FALLBACK)
    iris: list[str] = []
    seen: set[str] = set()
    for fname in predicate_labels:
        for iri in graph_cfg.get(fname, []):
            if iri not in seen:
                seen.add(iri)
                iris.append(iri)
    return iris


def predicate_label_menu(config: LabelConfig, graph_name: str) -> list[str]:
    """Return the sorted list of predicate label names available for a given graph.

    Parameters:
        config (LabelConfig):
            The per-graph label-predicate mapping built by build_label_config().
        graph_name (str):
            Short name of the target graph (e.g. "taxon", "gene").

    Returns:
        Sorted list of predicate label names available for the graph,
        falling back to FALLBACK keys if the graph is not in the config.
    """
    return sorted(config.get(graph_name, FALLBACK).keys())
