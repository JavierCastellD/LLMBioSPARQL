"""Derive per-graph attribute-predicate config from BGW_graphs.xlsx.

Complements entity_linker/label_config.py: that module resolves the NAME/LABEL/
SYNONYM family of predicates used for entity linking; this one resolves the
other attribute names an output-designation step may ask for (definition,
start/end position, strand, close match, description, comment). Unlike
label_config, this includes resource-typed predicates too (e.g. "close match"
points to another entity, not a literal).
"""
from __future__ import annotations

from pathlib import Path

import openpyxl

from ..entity_linker.label_config import find_graph_sheet, graph_names_from_classification

# Type alias: graph_name -> {attribute_name -> [predicate_iri, ...]}
AttributeConfig = dict[str, dict[str, list[str]]]


def _friendly_attribute_name(prop_label: str) -> str | None:
    """Map a spreadsheet property label to an attribute category.

    Parameters:
        prop_label (str):
            The property label string from the spreadsheet.

    Returns:
        One of "close match", "start position", "end position", "strand",
        "definition", "comment", "description", or None if the label does not
        match any known attribute category.
    """
    lbl = prop_label.lower()
    if "close" in lbl and "match" in lbl:
        return "close match"
    if "start" in lbl and "position" in lbl:
        return "start position"
    if "end" in lbl and "position" in lbl:
        return "end position"
    if "strand" in lbl:
        return "strand"
    if "definition" in lbl:
        return "definition"
    if lbl == "comment":
        return "comment"
    if lbl == "description":
        return "description"
    return None


def normalize_attribute_name(name: str) -> str:
    """Normalize a loosely-phrased attribute name (from output designation) to a
    canonical category, using the same keyword rules as the spreadsheet scan.

    Parameters:
        name (str):
            The attribute name as emitted by output designation (e.g. "synonym",
            "start position", possibly with different phrasing/casing).

    Returns:
        A canonical category string. Falls back to the lowercased, stripped
        input if it matches no known keyword pattern.
    """
    lbl = name.strip().lower()
    if "close" in lbl and "match" in lbl:
        return "close match"
    if "start" in lbl and "position" in lbl:
        return "start position"
    if "end" in lbl and "position" in lbl:
        return "end position"
    if "strand" in lbl:
        return "strand"
    if "definition" in lbl:
        return "definition"
    if "synonym" in lbl:
        return "synonym"
    if "comment" in lbl:
        return "comment"
    if "description" in lbl:
        return "description"
    return lbl


def build_attribute_config(excel_path: Path) -> AttributeConfig:
    """Parse BGW_graphs.xlsx and return the per-graph attribute-predicate mapping.

    Considers both literal- and resource-typed rows, unlike label_config's scan,
    since attributes such as "close match" point to another entity.

    Parameters:
        excel_path (Path):
            Path to the BGW_graphs.xlsx spreadsheet.

    Returns:
        A mapping of graph name to attribute name to list of predicate IRIs.
    """
    wb = openpyxl.load_workbook(excel_path)

    config: AttributeConfig = {}
    for graph_name in graph_names_from_classification(wb):
        ws = find_graph_sheet(wb, graph_name)
        if ws is None:
            continue

        attributes: dict[str, list[str]] = {}
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue  # skip header row
            pred_iri, prop_label = row[0], row[1]
            if pred_iri is None or prop_label is None:
                continue

            aname = _friendly_attribute_name(str(prop_label))
            if aname is None:
                continue

            pred_str = str(pred_iri)
            bucket = attributes.setdefault(aname, [])
            if pred_str not in bucket:
                bucket.append(pred_str)

        if attributes:
            config[graph_name] = attributes

    return config


def resolve_attribute(config: AttributeConfig, graph_name: str, attribute_name: str) -> str | None:
    """Resolve an attribute name to a single predicate IRI for a given graph.

    Parameters:
        config (AttributeConfig):
            The per-graph attribute-predicate mapping built by build_attribute_config().
        graph_name (str):
            Short name of the target graph (e.g. "gene", "taxon").
        attribute_name (str):
            Canonical attribute category (see normalize_attribute_name()).

    Returns:
        The first matching predicate IRI, or None if this graph has no predicate
        for that attribute category (fail-soft — the caller drops it with a warning).
    """
    iris = config.get(graph_name, {}).get(attribute_name)
    return iris[0] if iris else None
