"""Relation-aware path planner: resolves typed relation triples (from concept
extraction) to predicate paths over pathmap.json.

Run validation:  python -m src.pathmap.planner
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .builder import ALIASES, normalize

HOP_LIMIT = 6

# Cap on how many alternative paths are returned per triple
MAX_ALTERNATIVES = 20

# RELATION_VOCAB maps relation NAMES that concept extraction may emit to the
# path-map edge labels they resolve to
_REAL_LABELS = [
    "directly regulates",
    "enables",
    "encodes",
    "has version",
    "in taxon",
    "indirectly regulates",
    "involved in",
    "involved in negative regulation of",
    "involved in positive regulation of",
    "involved in regulation of",
    "is orthologous to",
    "molecularly interacts with",
    "part of",
    "has phenotype",
]

RELATION_VOCAB = {lab: [lab] for lab in _REAL_LABELS}

# 1:1 synonyms
RELATION_VOCAB.update({
    "encoded by":         ["encodes"],
    "interacts with":     ["molecularly interacts with"],
    "forms complex with": ["molecularly interacts with"],
    "in organism":        ["in taxon"],
    "found in organism":  ["in taxon"],
    "located on":         ["part of"],
    "located in":         ["part of"],
    "has function":       ["enables"]
})

# coarse names
RELATION_VOCAB.update({
    "regulates": [
        "involved in",
        "involved in regulation of",
        "involved in positive regulation of",
        "involved in negative regulation of",
        "directly regulates",
        "indirectly regulates",
    ]
})

# List of allowed relation names for the extraction step to import.
RELATION_NAMES = sorted(RELATION_VOCAB)

def resolve_relation(name: str) -> list[str] | None:
    """Resolve a relation name (as emitted by concept extraction) to the
    set of path-map "label"s it matches.

    Parameters:
        name (str): The coarse relation name (case-insensitive, surrounding
            whitespace is ignored).

    Returns:
        list[str] | None: The list of path-map labels this relation resolves
        to, or None if name is not a recognized relation.
    """
    return RELATION_VOCAB.get(name.strip().lower())


# Auxiliary functions

_DEFAULT_PATHMAP = Path(__file__).resolve().parents[2] / "data" / "pathmap.json"

def _resolve(type_str: str) -> str:
    """Apply alias, then normalise to the node-id form used in the path map."""
    canonical = ALIASES.get(type_str, type_str)
    return normalize(canonical)

def _format_path(seg: list[dict]) -> str:
    """Render a single path (list of edges) as 'node -[label]-> node -> ...'.

    The chain of nodes follows each edge's traversal "from"/"to" (guaranteed
    connected, since that's how the path was built), but the arrow for each
    hop is drawn using the "reverse" flag: "-[label]->" when "from"/"to"
    matches "subject_type"/"object_type", or "<-[label]-" when the hop
    traverses the predicate backwards (object -> subject).
    """
    parts = [seg[0]["from"]]
    for e in seg:
        arrow = f"<-[{e['label']}]-" if e["reverse"] else f"-[{e['label']}]->"
        parts.append(f"{arrow} {e['to']}")
    return " ".join(parts)


def print_resolved(result: dict[str, Any]) -> None:
    """Print a human-readable summary of a resolve_relations() result."""
    for entry in result["resolved"]:
        print(f"  {entry['subject']} --[{entry['relation']}]--> {entry['object']}"
              f"  ({entry['subject_type']} -> {entry['object_type']}, predicates={entry['predicates']})")
        for i, seg in enumerate(entry["segments"]):
            tag = f"[{i + 1}] " if len(entry["segments"]) > 1 else ""
            print(f"      {tag}{_format_path(seg)}")
    for entry in result["no_path"]:
        print(f"  NO PATH: {entry['subject']} --[{entry['relation']}]--> {entry['object']}"
              f"  ({entry['reason']})")


class PathPlanner:
    """Load pathmap.json and answer path queries."""

    def __init__(self, pathmap_path: Path | None = None) -> None:
        path = pathmap_path or _DEFAULT_PATHMAP
        with open(path, encoding="utf-8") as f:
            self._pm = json.load(f)
        self._edges = self._pm["edges"]

        # Adjacency list: node -> [(neighbour, edge_index), ...]
        self._adj: dict[str, list[tuple[str, int]]] = {}
        for i, edge in enumerate(self._edges):
            self._adj.setdefault(edge["from"], []).append((edge["to"], i))

        # Set of node ids actually present in the path map (the normalized
        # projection of builder.VALID_OBJECT_TYPES that ended up with edges).
        self._known_nodes = set(self._pm["nodes"])

        # Base IRI for BioGateway named graphs (see builder.GRAPH_BASE_IRI),
        # used to turn an edge's short "graph" name into a full graph IRI.
        self._graph_base_iri = self._pm.get("graph_base_iri", "")

    def _edge_to_triple(self, edge: dict) -> dict:
        """Turn a path-map edge into a SPARQL-ready triple pattern.

        Always uses "subject_type"/"object_type" (the true RDF triple direction),
        so the consumer never has to reason about traversal direction or the
        "reverse" flag: the result can be emitted directly as
        "GRAPH <graph> { ?<subject> <predicate> ?<object> . }".
        """
        return {
            "subject": edge["subject_type"],
            "predicate": edge["predicate"],
            "predicate_label": edge["label"],
            "object": edge["object_type"],
            "graph": self._graph_base_iri + edge["graph"],
        }

    def _simple_paths_of_length(self, start: str, goal: str, length: int) -> list[list[dict]]:
        """Method that finds all simple paths of exactly "length" edges from start to goal.

        A node may be visited at most twice, but only if the two entry edges carry
        different labels (e.g. entering protein via "encodes" and again via
        "molecularly interacts with"). The goal may additionally be revisited on the
        final step regardless of its entry labels, to handle same-type start==goal queries.

        Parameters:
            start (str): The starting node of the search.
            goal (str): The target node of the search.
            length (int): The exact number of edges the returned paths must have.
                Must be >= 1 (a 0-length path carries no relation).

        Returns:
            list[list[dict]]: Up to MAX_ALTERNATIVES paths, each a list of "length"
            edge dictionaries (in order from start to goal, each annotated with
            "reverse": True if "from"/"to" are flipped relative to
            "subject_type"/"object_type" for this hop).
        """
        results = []
        # Each stack entry is (node, depth, path-so-far, visited-so-far).
        # visited maps each node to the frozenset of edge labels used to enter it
        # (empty frozenset for the start node, which has no entry edge).
        stack = [(start, 0, [], {start: frozenset()})]

        while stack and len(results) < MAX_ALTERNATIVES:
            node, depth, path, visited = stack.pop()

            if depth == length:
                if node == goal:
                    results.append(path)
                continue

            for neighbour, eidx in reversed(self._adj.get(node, [])):
                edge = self._edges[eidx]
                edge_label = edge["label"]
                is_final_step = depth + 1 == length

                if neighbour in visited:
                    entry_labels = visited[neighbour]
                    can_revisit = (is_final_step and neighbour == goal) or (
                        len(entry_labels) == 1 and edge_label not in entry_labels
                    )
                    if not can_revisit:
                        continue

                next_path = path + [{**edge, "reverse": edge["from"] != edge["subject_type"]}]
                next_visited = dict(visited)
                if neighbour not in visited:
                    next_visited[neighbour] = frozenset({edge_label})
                elif len(visited[neighbour]) == 1 and edge_label not in visited[neighbour]:
                    next_visited[neighbour] = visited[neighbour] | {edge_label}
                stack.append((neighbour, depth + 1, next_path, next_visited))

        return results

    def _relation_paths(self, start: str, goal: str, allowed_predicates: list[str], hop_limit: int) -> list[list[dict]]:
        """Method that finds the shortest path from start to goal whose TERMINAL edge's "label" is in allowed_predicates.

        Tries path lengths up to hop_limit and returns the alternatives at the first length where at least one path's
        last edge matches. If no such path is found, falls back once: retries the same length range but matches on the
        second-to-last edge instead (paths of length >= 2 only), to handle cases where the relation sits one hop before
        the target (e.g. gene -[encodes]-> protein -[molecularly interacts with]-> protein <-[encodes]- gene).

        Parameters:
            start (str): The subject node's resolved type.
            goal (str): The object node's resolved type.
            allowed_predicates (list[str]): The path-map labels resolve_relation()
                returned for this triple's relation.
            hop_limit (int): The maximum path length (in edges) to consider.

        Returns:
            list[list[dict]]: Up to MAX_ALTERNATIVES alternative paths (each a list of edge dicts, in order from start to goal),
            or [] if no path of length <= hop_limit has a matching terminal or second-to-last label.
        """
        for length in range(1, hop_limit + 1):
            candidates = self._simple_paths_of_length(start, goal, length)
            matched = [p for p in candidates if p[-1]["label"] in allowed_predicates]
            if matched:
                return matched

        for length in range(2, hop_limit + 1):
            candidates = self._simple_paths_of_length(start, goal, length)
            matched = [p for p in candidates if p[-2]["label"] in allowed_predicates]
            if matched:
                return matched

        return []

    def resolve_relations(self, nodes: dict[str, dict], triples: list[dict]) -> dict[str, Any]:
        """Method that performs relation path resolution from concept-extraction output: a set of typed nodes 
        and a list of relation triples over them.

        For each triple, the relation name is resolved to its allowed path-map labels, then the shortest path 
        between the subject's and object's resolved types whose terminal edge label is one of those is found. 
        There might be multiple paths, which are returned.

        Parameters:
            nodes (dict[str, dict]): Maps node id -> {"type": <entity type>, "span": <optional>}. Entity types are as 
                                     emitted by concept extraction.
            triples (list[dict]): Each item is {"subject": <node id>, "relation": <relation name>, "object": <node id>}.

        Returns:
            dict[str, Any]: A dictionary with the following keys:
                - "nodes": A sorted list of all unique resolved entity-type nodes involved in any matched path, across all triples.
                - "edges": A list of unique edges (across all matched paths for all triples), each a dict with keys "from", "to", 
                           "predicate", "label", "graph", "subject_type", "object_type", and "reverse".
                - "triples": A deduplicated list of SPARQL-ready triple patterns, across all matched paths for all triples.
                - "resolved": A list with one entry per input triple that resolved to at least one path, each a dict with 
                              keys "subject", "relation", "object", "subject_type"/"object_type" (resolved types), "predicates" (the
                              allowed path-map labels for "relation"), "segments" (the alternative matched paths, and "bridged" 
                              (True if any segment has more than one hop).
                - "no_path": A list of dicts, one per input triple that could not be resolved, each with keys "subject", 
                             "relation", "object", and "reason".
        """
        all_nodes = set()
        all_edges = []
        seen_edge_keys = set()
        all_triples = []
        seen_triple_keys = set()
        resolved = []
        no_path = []

        
        for triple in triples:
            subj_id, rel_name, obj_id = triple["subject"], triple["relation"], triple["object"]

            # Resolve the relation name to path-map labels
            allowed_predicates = resolve_relation(rel_name)
            if allowed_predicates is None:
                no_path.append({**triple, "reason": "unknown relation"})
                continue

            subj_node = nodes.get(subj_id)
            obj_node = nodes.get(obj_id)
            if subj_node is None or obj_node is None:
                no_path.append({**triple, "reason": "unknown node id"})
                continue

            # Resolve the subject's and object's types, and check they're known in the path map
            subj_type = _resolve(subj_node["type"])
            obj_type = _resolve(obj_node["type"])
            if subj_type not in self._known_nodes or obj_type not in self._known_nodes:
                no_path.append({**triple, "reason": "unrecognized entity type"})
                continue

            # Obtains all possible paths from subj_type to obj_type whose terminal edge matches one of the allowed predicates
            segments = self._relation_paths(subj_type, obj_type, allowed_predicates, HOP_LIMIT)
            if not segments:
                no_path.append({
                    **triple,
                    "reason": f"no path from {subj_type!r} to {obj_type!r} matches "
                              f"predicate(s) {allowed_predicates!r} within {HOP_LIMIT} hops",
                })
                continue

            # Adds the resolved triple
            resolved.append({
                "subject": subj_id,
                "relation": rel_name,
                "object": obj_id,
                "subject_type": subj_type,
                "object_type": obj_type,
                "predicates": allowed_predicates,
                "segments": segments,
                "bridged": any(len(seg) > 1 for seg in segments) # This means that an intermediary node was needed to connect subject and object
            })


            for seg in segments:
                for e in seg:
                    all_nodes.add(e["from"])
                    all_nodes.add(e["to"])
                    edge_key = (e["from"], e["to"], e["predicate"], e["graph"])
                    if edge_key not in seen_edge_keys:
                        seen_edge_keys.add(edge_key)
                        all_edges.append(e)

                    triple_key = (e["subject_type"], e["predicate"], e["object_type"], e["graph"])
                    if triple_key not in seen_triple_keys:
                        seen_triple_keys.add(triple_key)
                        all_triples.append(self._edge_to_triple(e))

        return {
            "nodes": sorted(all_nodes),
            "edges": all_edges,
            "triples": all_triples,
            "resolved": resolved,
            "no_path": no_path,
        }


if __name__ == "__main__":

    planner = PathPlanner()

    # --- Task 1 validation ---------------------------------------------------

    print("\n=== Task 1 validation ===")

    # 1) RELATION_VOCAB sanity check: every label in RELATION_VOCAB's values exists
    #    in pathmap.json's labels, and every map label is reachable by at least one relation.
    map_labels = {e["label"] for e in planner._edges}
    vocab_labels = {label for labels in RELATION_VOCAB.values() for label in labels}
    missing_from_map = vocab_labels - map_labels
    missing_from_vocab = map_labels - vocab_labels
    print(f"  RELATION_VOCAB labels not in pathmap: {missing_from_map or 'none'}")
    print(f"  pathmap labels not covered by RELATION_VOCAB: {missing_from_vocab or 'none'}")

    # 2) "Which genes are involved in DNA repair in humans?"
    nodes_dna = {
        "OUTPUT": {"type": "gene"},
        "n1": {"type": "biological process", "span": "DNA repair"},
        "n2": {"type": "organism", "span": "humans"},
    }
    triples_dna = [
        {"subject": "OUTPUT", "relation": "involved in", "object": "n1"},
        {"subject": "OUTPUT", "relation": "in taxon", "object": "n2"},
    ]
    res_dna = planner.resolve_relations(nodes_dna, triples_dna)
    print("\n--- DNA repair in humans ---")
    print_resolved(res_dna)

    # 3) "Which proteins are involved in DNA repair and form complexes with FOXL1 in humans?"
    nodes_foxl1 = {
        "OUTPUT": {"type": "protein"},
        "n1": {"type": "biological process", "span": "DNA repair"},
        "x1": {"type": "protein"},
        "n2": {"type": "gene", "span": "FOXL1"},
        "n3": {"type": "organism", "span": "humans"},
    }
    triples_foxl1 = [
        {"subject": "OUTPUT", "relation": "involved in", "object": "n1"},
        {"subject": "OUTPUT", "relation": "interacts with", "object": "x1"},
        {"subject": "x1", "relation": "encoded by", "object": "n2"},
        {"subject": "OUTPUT", "relation": "in taxon", "object": "n3"},
    ]
    res_foxl1 = planner.resolve_relations(nodes_foxl1, triples_foxl1)
    print("\n--- FOXL1 complexes ---")
    print_resolved(res_foxl1)
