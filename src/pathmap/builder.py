"""Path-map builder: reads BGW_graphs.xlsx and writes data/pathmap.json.

Run:  python -m src.pathmap.builder
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import openpyxl

# Entity types extracted from BGW_graphs.xlsx column D, normalized and resolved via ALIASES.
# Resolved types (after ALIASES) are checked against this set.
VALID_OBJECT_TYPES = {
    "protein",
    "gene",
    "organism",
    "biological process",
    "cellular component",
    "molecular function",
    "chromosome",
    "genome assembly",
    "phenotype",
    "CRM",
    "tad",
}

# Aliases for non-canonical type strings found in the spreadsheet, mapping to canonical types
# Applied before the validity check so edge detection never silently drops matches.
ALIASES = {
    "disease": "phenotype",
    "target gene": "gene",
    "protein (transcriptional factor) with which it interacts": "protein",
}

# Maps entity-graph name → canonical entity type (must be in VALID_OBJECT_TYPES
# or a known non-edge-producing placeholder).
ENTITY_GRAPH_TYPE = {
    "gene": "gene",
    "prot": "protein",
    "crm": "CRM",
    "tad": "tad",
    "omim": "phenotype",
    "go": "go_term",                    # ontology graph — no outgoing entity edges
    "mi": "molecular_interaction_method",  # ontology graph — no outgoing entity edges
    "taxon": "organism",
}

# Base IRI for BioGateway graphs
GRAPH_BASE_IRI = "http://rdf.biogateway.eu/graph/"

# RDF CONSTANTS
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDF_SUBJECT = _RDF + "subject"
_RDF_OBJECT = _RDF + "object"

# To avoid repeats in relation graphs
STRUCTURAL_PREDICATES = {_RDF_SUBJECT, _RDF_OBJECT}

# Auxiliary functions
def normalize(type_str: str) -> str:
    """Lowercase + spaces/dashes to underscores for use as SPARQL variable base."""
    return type_str.lower().replace(" ", "_").replace("-", "_")


def resolve_type(raw: str | None) -> str | None:
    """Auxiliary function to return canonical type if raw (after aliasing) is in VALID_OBJECT_TYPES. Otherwise, returns None.
    
    Parameters:
        raw (str | None): The raw type string extracted from the spreadsheet, which may be None or a non-canonical string.
    
    Returns:
        str | None: The canonical type string if the raw input is valid and recognized. Otherwise, None.
    """
    if not raw:
        return None
    # Obtain the canonical type if it is found in ALIASES, otherwise keep the original raw string.
    canonical = ALIASES.get(raw, raw)
    return canonical if canonical in VALID_OBJECT_TYPES else None


def _sheet_rows(ws) -> list[dict]:
    """Auxiliary function to transform a worksheet into a list of dicts with keys: predicate, label, obj_node_type, entity_type.
    
    Parameters:
        ws: An openpyxl worksheet object representing a graph sheet.
    
    Returns:
        list[dict]: A list of dictionaries, each representing a row in the worksheet (excluding the header), with keys:
            - "predicate": The predicate IRI as a string.
            - "label": The label of the edge (empty string if not provided).
            - "obj_node_type": The type of the object node (e.g., "resource", "literal"), empty string if not provided.
            - "entity_type": The entity type specified in column D (after normalization), empty string if not provided.
    """
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        # Skip header
        if i == 0:
            continue
        pred = row[0]
        if pred is None:
            continue
        rows.append({
            "predicate": str(pred),
            "label": str(row[1]) if row[1] is not None else "",
            "obj_node_type": str(row[2]) if row[2] is not None else "",
            "entity_type": str(row[3]) if row[3] is not None else "",
        })
    return rows


def _make_edge(
    frm: str, to: str, predicate: str, label: str,
    graph: str, subject_type: str, object_type: str,
) -> dict:
    return {
        "from": frm,
        "to": to,
        "predicate": predicate,
        "label": label,
        "graph": graph,
        "subject_type": subject_type,
        "object_type": object_type,
    }


def _reverse_edge(edge: dict) -> dict:
    """Flip from/to while preserving the true subject_type and object_type."""
    return {**edge, "from": edge["to"], "to": edge["from"]}


def _build_entity_edges(graph_name: str, rows: list[dict]) -> list[dict]:
    """Auxiliary function to build edges for an entity graph, given the graph name
    and the rows of its sheet.
    
    Parameters:
        graph_name (str): The name of the graph (e.g., "gene graph", "taxon").
        rows (list[dict]): A list of dictionaries representing the rows of the graph's sheet,
            where each dictionary has keys: "predicate", "label", "obj_node_type", "entity_type".

    Returns:
        list[dict]: A list of edge dictionaries, where each dictionary has keys:
            - "from": The normalized source type (same for all edges in an entity graph).
            - "to": The normalized object type (must be in VALID_OBJECT_TYPES).
            - "predicate": The predicate IRI as a string.
            - "label": The label of the edge (empty string if not provided).
            - "graph": The name of the graph.
            - "subject_type": The canonical subject type (same for all edges in an entity graph).
            - "object_type": The canonical object type (must be in VALID_OBJECT_TYPES).
    """
    src_raw = ENTITY_GRAPH_TYPE.get(graph_name)
    src_type = resolve_type(src_raw)
    if src_type is None:
        return []

    src_norm = normalize(src_type)
    edges = []
    for row in rows:
        # We are not interested in literals here
        if row["obj_node_type"] != "resource":
            continue
        obj_type = resolve_type(row["entity_type"])
        if obj_type is None:
            continue
        obj_norm = normalize(obj_type)
        edges.append(_make_edge(
            src_norm, obj_norm,
            row["predicate"], row["label"],
            graph_name, src_norm, obj_norm,
        ))
    return edges


def _build_relation_edges(graph_name: str, rows: list[dict]) -> list[dict]:
    """Auxiliary function to build edges for a relation graph, given the graph name
    and the rows of its sheet.
    
    Parameters:
        graph_name (str): The name of the graph (e.g., "crm2gen graph", "ortho").
        rows (list[dict]): A list of dictionaries representing the rows of the graph's sheet,
            where each dictionary has keys: "predicate", "label", "obj_node_type", "entity_type".
    
    Returns:
        list[dict]: A list of edge dictionaries, where each dictionary has keys:
            - "from": The normalized subject type (must be in VALID_OBJECT_TYPES).
            - "to": The normalized object type (must be in VALID_OBJECT_TYPES).
            - "predicate": The predicate IRI as a string.
            - "label": The label of the edge (empty string if not provided).
            - "graph": The name of the graph.
            - "subject_type": The canonical subject type (must be in VALID_OBJECT_TYPES).
            - "object_type": The canonical object type (must be in VALID_OBJECT_TYPES).
    """
    subj_row = next((r for r in rows if r["predicate"] == _RDF_SUBJECT), None)
    obj_row = next((r for r in rows if r["predicate"] == _RDF_OBJECT), None)
    if subj_row is None or obj_row is None:
        return []

    # Check if the types for the relation are valid
    subj_type = resolve_type(subj_row["entity_type"])
    obj_type = resolve_type(obj_row["entity_type"])
    if subj_type is None or obj_type is None:
        return []

    subj_norm = normalize(subj_type)
    obj_norm = normalize(obj_type)

    edges = []
    for row in rows:
        if row["obj_node_type"] != "resource":
            continue
        # Skip RDF/ontology scaffolding rows
        # otherwise rdf:subject/rdf:object rows would be mistaken for the semantic
        # predicate whenever subject_type == object_type (self-loop relations).
        if row["predicate"] in STRUCTURAL_PREDICATES:
            continue
        # Semantic predicate: its resolved entity_type must match the object type.
        row_obj_type = resolve_type(row["entity_type"])
        if row_obj_type != obj_type:
            continue
        edges.append(_make_edge(
            subj_norm, obj_norm,
            row["predicate"], row["label"],
            graph_name, subj_norm, obj_norm,
        ))
    return edges

def build_pathmap(excel_path: Path) -> dict:
    """Method to parse the excel spreadsheet to generate a path-map dict.
    
    Parameters:
        excel_path (Path): The path to the BGW_graphs.xlsx file.

    Returns:
        dict: A dictionary containing 'nodes', 'edges', 'graph_base_iri', and 'source_hash' keys.
    """
    wb = openpyxl.load_workbook(excel_path)

    # Read graph classifications from the 'graphs' sheet.
    graph_types = {}
    for row in wb["graphs"].iter_rows(min_row=2, values_only=True):
        iri, gtype = row[0], row[1]
        if iri and gtype and gtype in ("Entity graph", "Relation graph"):
            name = str(iri).rstrip("/").split("/")[-1]
            graph_types[name] = str(gtype)

    source_hash = hashlib.sha256(excel_path.read_bytes()).hexdigest()

    all_edges = []
    for graph_name, graph_type in graph_types.items():
        # Obtain the sheet, which is named "<name> graph" or "<name>" (taxon, ortho, or reg2targ).
        ws = None
        candidate_name = f"{graph_name} graph"
        if candidate_name in wb.sheetnames:
            ws = wb[candidate_name]
        else:
            candidate_name = graph_name
            if candidate_name in wb.sheetnames:
                ws = wb[candidate_name]
        
        if ws is None:
            continue

        rows = _sheet_rows(ws)
        if graph_type == "Entity graph":
            edges = _build_entity_edges(graph_name, rows)
        else:
            edges = _build_relation_edges(graph_name, rows)

        for edge in edges:
            all_edges.append(edge)
            all_edges.append(_reverse_edge(edge))

    # Deduplicate on (from, to, predicate, graph) preserving insertion order.
    seen = set()
    deduped = []
    for e in all_edges:
        key = (e["from"], e["to"], e["predicate"], e["graph"])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    nodes = sorted({e["from"] for e in deduped} | {e["to"] for e in deduped})
    return {
        "nodes": nodes,
        "edges": deduped,
        "graph_base_iri": GRAPH_BASE_IRI,
        "source_hash": source_hash,
    }


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    xlsx = root / "data" / "BGW_graphs.xlsx"
    out = root / "data" / "pathmap.json"

    print(f"Reading {xlsx} ...")
    pm = build_pathmap(xlsx)
    out.write_text(json.dumps(pm, indent=2), encoding="utf-8")
    print(f"Wrote {out}  ({len(pm['edges'])} edges, {len(pm['nodes'])} nodes)")

    # Quick structural sanity checks
    edges = pm["edges"]

    def has_edge(frm: str, to: str, pred_fragment: str | None = None) -> bool:
        for e in edges:
            if e["from"] == frm and e["to"] == to:
                if pred_fragment is None or pred_fragment in e["predicate"]:
                    return True
        return False

    checks = [
        ("gene->organism via in taxon (RO_0002162)",
         has_edge("gene", "organism", "RO_0002162")),
        ("organism->gene (reverse in taxon)",
         has_edge("organism", "gene", "RO_0002162")),
        ("gene->protein via encodes (SIO_010078)",
         has_edge("gene", "protein", "SIO_010078")),
        ("protein->gene (reverse encodes)",
         has_edge("protein", "gene", "SIO_010078")),
        ("protein->biological_process via involved in (RO_0002331, prot2bp)",
         has_edge("protein", "biological_process", "RO_0002331")),
        ("biological_process->protein (reverse)",
         has_edge("biological_process", "protein", "RO_0002331")),
        ("gene->phenotype via gene2phen (alias disease->phenotype)",
         has_edge("gene", "phenotype")),
        ("phenotype->gene (reverse gene2phen)",
         has_edge("phenotype", "gene")),
        ("no structural predicates in edge list",
         not any(e["predicate"] in STRUCTURAL_PREDICATES for e in edges)),
    ]

    print("\n=== Path-map validation ===")
    all_ok = True
    for desc, ok in checks:
        status = "OK" if ok else "FAIL"
        print(f"  {status}  {desc}")
        if not ok:
            all_ok = False

    print(f"\n  Nodes ({len(pm['nodes'])}): {pm['nodes']}")
    print(f"  Edges: {len(pm['edges'])} (including reverses)")
    print(f"\n  {'All checks passed.' if all_ok else 'Some checks FAILED.'}")