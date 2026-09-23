"""Builds a compact, human-readable schema dump from BGW_graphs.xlsx.

This is the ablation baseline's only source of BioGateway schema knowledge:
unlike the full pipeline (which resolves predicates/graphs/IRIs
programmatically and hands the LLM only the exact facts it needs), the direct
generator gets one flat text description of the in-scope graphs up front and
must recall the right predicate from it on its own.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl

from ..entity_linker.label_config import find_graph_sheet, graph_names_from_classification

# Restricted to the same graph scope the pipeline is evaluated against
# (see queries/README.md / the paper's Methods section): six entity graphs
# and seven relation graphs.
IN_SCOPE_GRAPHS = [
    "gene", "prot", "taxon", "tad", "go", "omim",
    "gene2phen", "tfac2gene", "prot2prot", "prot2bp", "prot2mf", "prot2cc", "reg2targ",
]

GRAPH_IRI_PATTERN = "http://rdf.biogateway.eu/graph/{graph}"

# Common prefixes used throughout BioGateway's schema and every hand-written
# gold query in this project (kept identical to the ones the structured
# pipeline's own SPARQL-generation prompt uses, for a fair comparison).
COMMON_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.geneontology.org/formats/oboInOwl#",
    "obo": "http://purl.obolibrary.org/obo/",
    "sio": "http://semanticscience.org/resource/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}


def _short_iri(iri: str) -> str:
    """Shorten a predicate/class IRI to prefix:local form when a common prefix covers it.

    Parameters:
        iri (str):
            The full predicate or class IRI from the spreadsheet.

    Returns:
        The CURIE form (e.g. "skos:prefLabel") if a known prefix's namespace
        is a leading substring of iri, otherwise the IRI unchanged.
    """
    for prefix, ns in COMMON_PREFIXES.items():
        if iri.startswith(ns):
            return f"{prefix}:{iri[len(ns):]}"
    return iri


def build_schema_text(excel_path: Path, graph_names: list[str] = IN_SCOPE_GRAPHS) -> str:
    """Render a compact per-graph predicate listing from BGW_graphs.xlsx.

    For each in-scope graph, lists every predicate row from its sheet as
    "predicate (friendly label) -> object node type [: example]", skipping
    fully empty rows. This is deliberately the raw schema, not a curated
    subset — the ablation should have access to roughly the same information
    a person would from the spreadsheet, not a hand-picked cheat sheet.

    Parameters:
        excel_path (Path):
            Path to the BGW_graphs.xlsx spreadsheet.
        graph_names (list[str]):
            Short graph names to include, in output order. Defaults to
            IN_SCOPE_GRAPHS.

    Returns:
        A single text block, one section per graph, ready to embed in a
        prompt.
    """
    wb = openpyxl.load_workbook(excel_path, read_only=True)
    known = set(graph_names_from_classification(wb))

    sections = []
    for name in graph_names:
        if name not in known:
            continue
        ws = find_graph_sheet(wb, name)
        if ws is None:
            continue

        lines = [f"### Graph <{GRAPH_IRI_PATTERN.format(graph=name)}>"]
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue  # header row
            pred_iri, prop_label, obj_node_type, obj_desc = row[0], row[1], row[2], row[3]
            if pred_iri is None:
                continue
            pred = _short_iri(str(pred_iri))
            piece = f"- {pred}"
            if prop_label:
                piece += f" ({prop_label})"
            if obj_node_type:
                piece += f" -> {obj_node_type}"
            if obj_desc:
                piece += f": {obj_desc}"
            lines.append(piece)
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_prefix_block(prefixes: dict[str, str] = COMMON_PREFIXES) -> str:
    """Render PREFIX declarations for the schema's common namespaces.

    Parameters:
        prefixes (dict[str, str]):
            Mapping of prefix name to namespace IRI. Defaults to COMMON_PREFIXES.

    Returns:
        One "PREFIX name: <iri>" line per entry, newline-joined.
    """
    return "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in prefixes.items())
