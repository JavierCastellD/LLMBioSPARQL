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
from .path_resolver import PathResolver, hop_orientation

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

    "synonym" is delegated to label_config: tries the graph's generic "synonym"
    predicate first, falling back to "exact synonym" for graphs (like taxon,
    go) that only expose the subtyped synonym variants — the same default
    preference the entity linker itself uses. Every other category is looked
    up in attr_cfg, which has no cross-graph fallback since these are
    graph-specific facts.

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
        iris = get_predicates(label_cfg, graph, ["synonym"]) or get_predicates(label_cfg, graph, ["exact synonym"])
        return iris[0] if iris else None
    return resolve_attribute(attr_cfg, graph, category)


def _bridge_intermediate_type(hop: dict) -> str:
    """Return the resolved type of the node a bridge hop lands on (not the side it came from).

    Parameters:
        hop (dict):
            A single segment hop with subject_type, object_type, and reverse.

    Returns:
        hop["object_type"] if the hop runs subject->object in traversal order,
        or hop["subject_type"] if "reverse" (traversal runs the other way).
    """
    return hop["subject_type"] if hop["reverse"] else hop["object_type"]


def _seed_bridge_cache(resolved: list[dict]) -> dict[tuple[str, str, tuple[str, ...]], str]:
    """Pre-seed the bridge-variable cache from every non-bridged edge's resolved object.

    A bridge hop that passes through the same (source, type, predicate-chain) as an
    already-named node (e.g. "n1 encodes n0") should promote to that node's own id
    instead of minting a fresh one — see docs/planner_variable_identity_spec.md. This
    must run as a separate pass BEFORE any bridge hop is expanded, since the relevant
    non-bridged entry can appear anywhere in "resolved" relative to the bridged ones
    it should inform (observed in practice: the direct "encodes" edge often comes
    after the bridged edges that should reuse it).

    Parameters:
        resolved (list[dict]):
            The "resolved" list from PathPlanner.resolve_relations().

    Returns:
        A dict mapping (source_node_id, landed_on_type, predicate_chain) to the
        already-named node id, for every single-hop (non-bridged) resolved entry.
    """
    cache: dict[tuple[str, str, tuple[str, ...]], str] = {}
    for entry in resolved:
        segment = entry["segments"][0]
        if len(segment) > 1:
            continue  # bridged; nothing named to promote to here
        hop = segment[0]
        key = (entry["subject"], _bridge_intermediate_type(hop), (to_curie(hop["predicate"]),))
        cache.setdefault(key, entry["object"])
    return cache


def _resolve_segments(
    resolved: list[dict], question: str, path_resolver: PathResolver, nodes: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    """Narrow each resolved edge's candidate "segments" down to the one PathResolver chose.

    The planner may return several equal-length candidate paths per edge (see
    docs/path_resolver_spec.md); this must run BEFORE _seed_bridge_cache/
    _edges_from_resolved so both of them can keep reading "segments[0]" as
    before and always get the chosen path, not an arbitrary first alternative.

    Parameters:
        resolved (list[dict]):
            The "resolved" list from PathPlanner.resolve_relations().
        question (str):
            The full natural-language question, given to the resolver as
            disambiguating context for its LLM tiebreak tiers.
        path_resolver (PathResolver):
            Chooses one segment per edge; see path_resolver.PathResolver.resolve().
        nodes (dict[str, dict]):
            Concept-extraction's node id -> {"type", "span"?} mapping, passed
            through so the resolver's LLM tiebreak can show which phrase in
            the question each edge's subject/object refer to.

    Returns:
        A tuple (resolved, records): "resolved" is a copy of the input with
        every entry's "segments" replaced by a single-element list containing
        only the chosen candidate; "records" mirrors PathResolver's
        per-edge resolution info (tier, validated, etc.) for logging/
        inspection — never sent to the LLM as part of the brief.
    """
    narrowed: list[dict] = []
    records: list[dict] = []
    for entry in resolved:
        result = path_resolver.resolve(question, entry, nodes)
        narrowed.append({**entry, "segments": [result["segment"]]})
        records.append({
            "subject": entry["subject"],
            "relation": entry["relation"],
            "object": entry["object"],
            **result["record"],
        })
    return narrowed, records


def _edges_from_resolved(resolved: list[dict]) -> list[dict]:
    """Turn the planner's type-level "resolved" paths into node-id-level edges.

    Expects "resolved" to already carry exactly one candidate per entry (see
    _resolve_segments) — this function just reads segments[0]. Bridge hops
    (multi-hop segments) reuse a variable for their intermediate node rather than
    always minting a fresh one: a cache keyed by (source_node_id, intermediate_type,
    predicate_chain_from_source) is shared across every resolved entry, so two
    independent bridged paths starting from the same node via the same predicate
    chain (e.g. "this gene's protein" reached via "encodes" for two different
    onward constraints) collapse onto one variable instead of each inventing its
    own — and a bridge hop that matches an already-NAMED node's own (source, type,
    chain) promotes to that node's id instead of minting an "x" var at all. Two
    same-typed intermediates only stay distinct when they differ in source and/or
    predicate chain (e.g. an "encoded protein" vs. a separate "interactor protein"
    reached via a different predicate) — see docs/planner_variable_identity_spec.md.

    Every edge is emitted in its TRUE RDF subject->object direction (never
    flipped), using each hop's "reverse" flag to tell which side of the
    traversal is the real subject.

    Parameters:
        resolved (list[dict]):
            The "resolved" list from PathPlanner.resolve_relations() (already
            narrowed to one candidate per entry by _resolve_segments), each with
            keys subject, object (original node ids) and segments (a single
            chosen path, a list of edge dicts with from/to/predicate/label/graph/
            subject_type/object_type/reverse).

    Returns:
        A list of edge dicts, each {"subject", "predicate" (CURIE), "object",
        "subject_graph", and "note" (only present when the edge is a bridge hop)}.
    """
    edges: list[dict] = []
    intermediate_count = 0
    bridge_cache = _seed_bridge_cache(resolved)

    for entry in resolved:
        segment = entry["segments"][0]
        current_id = entry["subject"]
        bridged = len(segment) > 1
        chain: list[str] = []  # predicate CURIEs from entry["subject"] up to the current hop

        for i, hop in enumerate(segment):
            is_last = i == len(segment) - 1
            pred_curie = to_curie(hop["predicate"])
            chain.append(pred_curie)

            if is_last:
                next_id = entry["object"]
            else:
                key = (entry["subject"], _bridge_intermediate_type(hop), tuple(chain))
                next_id = bridge_cache.get(key)
                if next_id is None:
                    intermediate_count += 1
                    next_id = f"x{intermediate_count}"
                    bridge_cache[key] = next_id

            subj_id, obj_id = hop_orientation(hop, current_id, next_id)

            edge = {
                "subject": subj_id,
                "predicate": pred_curie,
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
        couldn't be grounded (dropped, fail-soft). A "bind" entry also carries
        "uri": the linker's already-resolved URI, copied verbatim (not
        CURIE-shortened) so the generator can inline it directly instead of
        re-emitting a label-lookup triple for an entity that's already known.
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

        if mode == "bind":
            grounding[node_id] = {
                "mode": mode,
                "graph": graph,
                "term": out["term"],
                "label_predicate": to_curie(out["label_predicate"]),
                "uri": out["uri"],
            }
        elif mode == "many":
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
    path_resolver: PathResolver | None = None,
) -> tuple[dict, list[str], list[dict]]:
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
        path_resolver (PathResolver):
            Chooses one candidate segment per resolved edge when the planner
            found several equal-length alternatives; built with defaults if
            omitted (see docs/path_resolver_spec.md).

    Returns:
        A tuple (brief, warnings, path_resolutions): brief is the
        JSON-serializable assembly brief; warnings lists every element dropped
        during fail-soft resolution; path_resolutions is one record per
        resolved edge (tier, validated, etc.) for logging/inspection only —
        never sent to the LLM as part of the brief.
    """
    xlsx = excel_path or _DEFAULT_EXCEL
    if label_cfg is None:
        label_cfg = build_label_config(xlsx) if xlsx.exists() else {}
    if attr_cfg is None:
        attr_cfg = build_attribute_config(xlsx) if xlsx.exists() else {}
    if path_resolver is None:
        path_resolver = PathResolver()

    nodes = extraction["nodes"]

    resolved, path_resolutions = _resolve_segments(planner_result.get("resolved", []), question, path_resolver, nodes)
    edges = _edges_from_resolved(resolved)
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
    return brief, warnings, path_resolutions
