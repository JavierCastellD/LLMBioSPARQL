"""Derive per-graph label-predicate config from BGW_graphs.xlsx.

For each named graph, scans its spreadsheet sheet for literal-typed predicates
whose property label contains "name", "label", "synonym", or "title", and maps
them to the friendly names exposed to the LLM:
  "preferred label", "label", "synonyms", "title".
"""
from __future__ import annotations

import re
from pathlib import Path

import openpyxl

# Fallback predicate set for graphs not found in the spreadsheet.
FALLBACK = {
    "preferred label": ["http://www.w3.org/2004/02/skos/core#prefLabel"],
    "label":           ["http://www.w3.org/2000/01/rdf-schema#label"],
    "synonyms": [
        "http://www.w3.org/2004/02/skos/core#altLabel",
        "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    ],
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
        return "synonyms"
    # \bname\b matches "has name" but NOT "namespace" or "obo_namespace"
    if re.search(r"\bname\b", lbl):
        return "preferred label"
    if "label" in lbl:
        return "label"
    if "title" in lbl:
        return "title"
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

    # Collect short graph names from the 'graphs' classification sheet.
    graph_names: list[str] = []
    for row in wb["graphs"].iter_rows(min_row=2, values_only=True):
        iri, gtype = row[0], row[1]
        if iri and gtype and gtype in ("Entity graph", "Relation graph"):
            name = str(iri).rstrip("/").split("/")[-1]
            graph_names.append(name)

    config: LabelConfig = {}
    for graph_name in graph_names:
        # Sheet names follow two conventions: "gene graph" or plain "taxon".
        ws = None
        for candidate in (f"{graph_name} graph", graph_name):
            if candidate in wb.sheetnames:
                ws = wb[candidate]
                break
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

    Falls back to FALLBACK when the graph is not in the config or a friendly
    name is not mapped for that graph. Never raises.

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
        # If a friendly name is missing from this graph's config, use the FALLBACK entry.
        for iri in graph_cfg.get(fname, FALLBACK.get(fname, [])):
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
