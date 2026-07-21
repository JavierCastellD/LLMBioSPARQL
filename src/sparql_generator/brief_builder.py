"""Assembly-brief builder for the SPARQL generator.

Code resolves every fact (predicates, graphs, IRIs) the LLM will need to author
the query — the LLM never invents them, it only assembles structure. This
module turns the outputs of the planner, entity linker, and output designation
into the single JSON "brief" handed to the LLM.

Resolution is fail-soft: an element that cannot be resolved (unknown type,
unresolved span, unmapped attribute name) is dropped with a warning rather than
raising, so a partial brief can still be produced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..entity_linker.label_config import LabelConfig, build_label_config, get_predicates
from ..entity_linker.linker import TYPE_TO_GRAPH
from .attribute_resolver import AttributeConfig, build_attribute_config, normalize_attribute_name, resolve_attribute

_DEFAULT_EXCEL = Path(__file__).resolve().parents[2] / "data" / "BGW_graphs.xlsx"

# Emit verbatim in every brief; "owl:" is bound to oboInOwl (not the real OWL
# namespace) since that's the prefix BioGateway's gold queries use for
# hasExactSynonym and friends.
PREFIXES: dict[str, str] = {
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "obo":  "http://purl.obolibrary.org/obo/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "sio":  "http://semanticscience.org/resource/",
    "owl":  "http://www.geneontology.org/formats/oboInOwl#",
}

GRAPH_IRI_PATTERN = "http://rdf.biogateway.eu/graph/{graph}"

# Class anchors (for rdfs:subClassOf), only defined for these four types.
CLASS_ANCHORS: dict[str, str] = {
    "gene":    "sio:SIO_010035",
    "protein": "sio:SIO_010043",
    "CRM":     "obo:SO_0000727",
    "tad":     "obo:SO_0002304",
}


def to_curie(iri: str) -> str:
    """Shorten a full IRI to a "prefix:local" CURIE using PREFIXES.

    Parameters:
        iri (str):
            The full predicate/class IRI to shorten.

    Returns:
        The CURIE form if iri starts with a known namespace, otherwise the IRI
        wrapped in angle brackets (a syntactically valid SPARQL absolute IRI).
    """
    for prefix, namespace in PREFIXES.items():
        if iri.startswith(namespace):
            return f"{prefix}:{iri[len(namespace):]}"
    return f"<{iri}>"


def _name_predicate(label_cfg: LabelConfig, graph: str) -> str:
    """Resolve a graph's default name predicate, as a CURIE.

    Tries "preferred label" (skos:prefLabel-style) first, falling back to
    "label" (rdfs:label-style) — matching what entity graphs vs. ontology
    graphs actually carry.

    Parameters:
        label_cfg (LabelConfig):
            The per-graph label-predicate mapping built by build_label_config().
        graph (str):
            Short graph name (e.g. "gene", "go").

    Returns:
        The name predicate as a CURIE.
    """
    iris = get_predicates(label_cfg, graph, ["preferred label"]) or get_predicates(label_cfg, graph, ["label"])
    return to_curie(iris[0])


def _resolve_attribute_iri(
    label_cfg: LabelConfig,
    attr_cfg: AttributeConfig,
    graph: str,
    attribute_name: str,
) -> str | None:
    """Resolve an output-designation attribute name to a predicate IRI for a graph.

    "synonym" is delegated to label_config (which already covers it with a
    sensible fallback); every other category is looked up in attr_cfg, which
    has no cross-graph fallback since these are graph-specific facts.

    Parameters:
        label_cfg (LabelConfig):
            The per-graph label-predicate mapping built by build_label_config().
        attr_cfg (AttributeConfig):
            The per-graph attribute-predicate mapping built by build_attribute_config().
        graph (str):
            Short graph name (e.g. "gene", "omim").
        attribute_name (str):
            The attribute name as emitted by output designation.

    Returns:
        The resolved predicate IRI, or None if unresolvable (fail-soft).
    """
    category = normalize_attribute_name(attribute_name)
    if category == "synonym":
        iris = get_predicates(label_cfg, graph, ["synonym"])
        return iris[0] if iris else None
    return resolve_attribute(attr_cfg, graph, category)


def _edges_from_resolved(resolved: list[dict]) -> list[dict]:
    """Turn the planner's type-level "resolved" paths into node-id-level edges.

    Picks the first (shortest) alternative path per resolved triple and mints
    fresh intermediate node ids (x1, x2, ...) for any bridge hops. Every edge is
    emitted in its TRUE RDF subject->object direction (never flipped), using
    each hop's "reverse" flag to tell which side of the traversal is the real
    subject.

    Parameters:
        resolved (list[dict]):
            The "resolved" list from PathPlanner.resolve_relations(), each with
            keys subject, object (original node ids) and segments (alternative
            paths, each a list of edge dicts with from/to/predicate/label/graph/
            subject_type/object_type/reverse).

    Returns:
        A list of edge dicts, each {"subject", "predicate" (CURIE), "object",
        "subject_graph", and "note" (only present when the edge is a bridge hop)}.
    """
    edges: list[dict] = []
    intermediate_count = 0

    for entry in resolved:
        segment = entry["segments"][0]
        current_id = entry["subject"]
        bridged = len(segment) > 1

        for i, hop in enumerate(segment):
            is_last = i == len(segment) - 1
            if is_last:
                next_id = entry["object"]
            else:
                intermediate_count += 1
                next_id = f"x{intermediate_count}"

            if hop["reverse"]:
                # Traversal direction is object->subject; the true RDF triple
                # runs the other way (next_id is the true subject).
                subj_id, obj_id = next_id, current_id
            else:
                subj_id, obj_id = current_id, next_id

            edge = {
                "subject": subj_id,
                "predicate": to_curie(hop["predicate"]),
                "object": obj_id,
                "subject_graph": hop["graph"],
            }
            if bridged:
                edge["note"] = f"{entry['subject_type']} to {entry['object_type']} bridge via {hop['label']!r}"
            edges.append(edge)

            current_id = next_id

    return edges


def _grounding_from_linker(linker_input: list[dict], linker_output: list[dict]) -> tuple[dict, list[str]]:
    """Build the brief's "grounding" section from an entity-linker run.

    Aligns linker_input and linker_output by POSITION (EntityLinker.link()
    guarantees a same-length, same-order result list), not by span text, since
    duplicate spans on different node ids would make text-matching ambiguous.

    Parameters:
        linker_input (list[dict]):
            The span nodes passed to EntityLinker.link() (node_id, span, type).
        linker_output (list[dict]):
            The corresponding results from EntityLinker.link().

    Returns:
        A tuple (grounding, warnings): grounding maps node_id -> a compact
        mode/graph/term/predicate(s) dict; warnings lists any span that
        couldn't be grounded (dropped, fail-soft).
    """
    grounding: dict[str, dict] = {}
    warnings: list[str] = []

    for inp, out in zip(linker_input, linker_output):
        node_id = inp["node_id"]
        graph = TYPE_TO_GRAPH.get(inp["type"])
        mode = out.get("mode")

        if graph is None:
            warnings.append(f"node {node_id!r}: unknown type {inp['type']!r} — grounding omitted")
            continue

        if mode in ("bind", "many"):
            grounding[node_id] = {
                "mode": mode,
                "graph": graph,
                "term": out["term"],
                "label_predicate": to_curie(out["label_predicate"]),
            }
        elif mode == "filter":
            grounding[node_id] = {
                "mode": "filter",
                "graph": graph,
                "term": out["term"],
                "predicates": [to_curie(p) for p in out["predicates"]],
            }
        else:
            warnings.append(f"span {inp['span']!r} (node {node_id!r}) is {mode} — grounding omitted")

    return grounding, warnings


def _projection_outputs(outputs: list[str], nodes: dict[str, dict], label_cfg: LabelConfig) -> tuple[dict, list[str]]:
    """Build the brief's "projection.outputs" section.

    Parameters:
        outputs (list[str]):
            Node ids to return, from concept extraction.
        nodes (dict[str, dict]):
            Node id -> {"type": ..., "span": optional}, from concept extraction.
        label_cfg (LabelConfig):
            The per-graph label-predicate mapping built by build_label_config().

    Returns:
        A tuple (projection_outputs, warnings): maps node_id -> {"graph",
        "name_predicate", "class_anchor" (only if the type has one)}; warnings
        lists any output whose type has no known graph (dropped, fail-soft).
    """
    result: dict[str, dict] = {}
    warnings: list[str] = []

    for node_id in outputs:
        node = nodes.get(node_id)
        if node is None:
            warnings.append(f"output {node_id!r}: not found in nodes — omitted")
            continue

        node_type = node["type"]
        graph = TYPE_TO_GRAPH.get(node_type)
        if graph is None:
            warnings.append(f"output {node_id!r}: unknown type {node_type!r} — omitted")
            continue

        entry = {"graph": graph, "name_predicate": _name_predicate(label_cfg, graph)}
        anchor = CLASS_ANCHORS.get(node_type)
        if anchor:
            entry["class_anchor"] = anchor
        result[node_id] = entry

    return result, warnings


def _projection_attributes(
    attributes: list[dict],
    nodes: dict[str, dict],
    label_cfg: LabelConfig,
    attr_cfg: AttributeConfig,
) -> tuple[list[dict], list[str]]:
    """Build the brief's "projection.attributes" list, resolving names to IRIs.

    Parameters:
        attributes (list[dict]):
            The output-designation "attributes" list: [{"on": node_id, "attribute": name}, ...].
        nodes (dict[str, dict]):
            Node id -> {"type": ..., "span": optional}, from concept extraction.
        label_cfg (LabelConfig):
            The per-graph label-predicate mapping built by build_label_config().
        attr_cfg (AttributeConfig):
            The per-graph attribute-predicate mapping built by build_attribute_config().

    Returns:
        A tuple (projection_attributes, warnings): a list of {"on", "predicate"
        (CURIE), "label"} dicts; warnings lists any attribute that couldn't be
        resolved (dropped, fail-soft).
    """
    result: list[dict] = []
    warnings: list[str] = []

    for attr in attributes:
        node_id, name = attr["on"], attr["attribute"]
        node = nodes.get(node_id)
        if node is None:
            warnings.append(f"attribute {name!r} on {node_id!r}: node not found — omitted")
            continue

        graph = TYPE_TO_GRAPH.get(node["type"])
        if graph is None:
            warnings.append(f"attribute {name!r} on {node_id!r}: unknown type {node['type']!r} — omitted")
            continue

        iri = _resolve_attribute_iri(label_cfg, attr_cfg, graph, name)
        if iri is None:
            warnings.append(f"attribute {name!r} on {node_id!r} (graph {graph!r}): no matching predicate — omitted")
            continue

        result.append({"on": node_id, "predicate": to_curie(iri), "label": name})

    return result, warnings


def build_brief(
    question: str,
    extraction: dict,
    planner_result: dict,
    linker_input: list[dict],
    linker_output: list[dict],
    designation: dict,
    label_cfg: LabelConfig | None = None,
    attr_cfg: AttributeConfig | None = None,
    excel_path: Path | None = None,
) -> tuple[dict, list[str]]:
    """Assemble the LLM-facing SPARQL-generation brief from every upstream step's output.

    Parameters:
        question (str):
            The natural-language question.
        extraction (dict):
            Concept-extraction result: {"outputs": [...], "nodes": {...}, "triples": [...]}.
        planner_result (dict):
            PathPlanner.resolve_relations() result (uses its "resolved" list).
        linker_input (list[dict]):
            The span nodes passed to EntityLinker.link().
        linker_output (list[dict]):
            The corresponding EntityLinker.link() results.
        designation (dict):
            Output-designation result: {"attributes": [{"on", "attribute"}, ...]}.
        label_cfg (LabelConfig):
            Pre-built label config; built from excel_path if omitted.
        attr_cfg (AttributeConfig):
            Pre-built attribute config; built from excel_path if omitted.
        excel_path (Path):
            Path to BGW_graphs.xlsx, used only if label_cfg/attr_cfg are omitted.

    Returns:
        A tuple (brief, warnings): brief is the JSON-serializable assembly brief;
        warnings lists every element dropped during fail-soft resolution.
    """
    xlsx = excel_path or _DEFAULT_EXCEL
    if label_cfg is None:
        label_cfg = build_label_config(xlsx) if xlsx.exists() else {}
    if attr_cfg is None:
        attr_cfg = build_attribute_config(xlsx) if xlsx.exists() else {}

    nodes = extraction["nodes"]

    edges = _edges_from_resolved(planner_result.get("resolved", []))
    grounding, grounding_warnings = _grounding_from_linker(linker_input, linker_output)
    proj_outputs, output_warnings = _projection_outputs(extraction["outputs"], nodes, label_cfg)
    proj_attributes, attr_warnings = _projection_attributes(
        designation.get("attributes", []), nodes, label_cfg, attr_cfg
    )

    brief: dict[str, Any] = {
        "question": question,
        "outputs": extraction["outputs"],
        "nodes": nodes,
        "edges": edges,
        "grounding": grounding,
        "projection": {"outputs": proj_outputs, "attributes": proj_attributes},
        "prefixes": PREFIXES,
        "graph_iri_pattern": GRAPH_IRI_PATTERN,
    }
    warnings = grounding_warnings + output_warnings + attr_warnings
    return brief, warnings
