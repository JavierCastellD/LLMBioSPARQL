"""Path planner: BFS over pathmap.json to find traversal paths between entity types.

Run validation:  python -m src.pathmap.planner
"""
from __future__ import annotations

import json
import warnings
from collections import deque
from pathlib import Path
from typing import Any

from .builder import ALIASES, normalize

HOP_LIMIT = 6

# Cap on how many equally-short alternative paths are returned per constraint.
# The graph has several node pairs connected by multiple parallel edges with
# different predicates (e.g. gene<->protein via "encodes" AND three tfac2gene
# regulation predicates), so without a cap a multi-hop tie could combine these
# combinatorially. 20 is generous for an 11-node graph but keeps results bounded.
MAX_ALTERNATIVES = 20

# Auxiliary functions

_DEFAULT_PATHMAP = Path(__file__).resolve().parents[2] / "data" / "pathmap.json"

def _resolve(type_str: str) -> str:
    """Apply alias, then normalise to the node-id form used in the path map."""
    canonical = ALIASES.get(type_str, type_str)
    return normalize(canonical)

def _format_path(seg: list[dict]) -> str:
    """Render a single path (list of edges) as 'node -[label]-> node -> ...'.

    The chain of nodes follows each edge's traversal "from"/"to" (guaranteed
    connected, since that's how _all_shortest_paths built the path), but the
    arrow for each hop is drawn using the "reverse" flag: "-[label]->" when
    "from"/"to" matches "subject_type"/"object_type", or "<-[label]-" when
    the hop traverses the predicate backwards (object -> subject).
    """
    parts = [seg[0]["from"]]
    for e in seg:
        arrow = f"<-[{e['label']}]-" if e["reverse"] else f"-[{e['label']}]->"
        parts.append(f"{arrow} {e['to']}")
    return " ".join(parts)


def print_plan(path_plan: dict[str, Any]) -> None:
    """Print a human-readable summary of a path plan.

    "segments[orig]" is a list of equally-short alternative paths (see
    PathPlanner._all_shortest_paths); when there is more than one, each is
    printed on its own line, numbered.
    """
    print(f"Nodes: {path_plan['nodes']}")
    for orig, alt_paths in path_plan["segments"].items():
        if not alt_paths:
            continue
        # Drop empty paths (output type == constraint type itself: nothing to traverse).
        non_empty = [seg for seg in alt_paths if seg]
        if not non_empty:
            continue
        if len(non_empty) == 1:
            print(f"  {orig}: {_format_path(non_empty[0])}")
        else:
            print(f"  {orig}: ({len(non_empty)} equally-short alternatives)")
            for i, seg in enumerate(non_empty, start=1):
                print(f"    [{i}] {_format_path(seg)}")

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

    def plan(self, output: str, constraints: list[str], hop_limit: int = HOP_LIMIT) -> dict[str, Any]:
        """Method that returns a dictionary representing the path plan to the output type considering
        the constraints. For each constraint, ALL shortest paths from the output node are found
        (see PathPlanner._all_shortest_paths) -- ties are not resolved arbitrarily, every equally-short
        alternative is kept. All per-constraint paths (across all alternatives) are merged into a single
        node set and deduplicated edge/triple lists.

        Parameters:
            output (str): The target output type for which the path plan is to be generated.
            constraints (list[str]): A list of constraint types that need to be connected to the output type.
            hop_limit (int): The maximum number of hops to consider when searching for paths. Defaults to HOP_LIMIT.

        Returns:
            dict[str, Any]: A dictionary containing the output type, resolved constraints, nodes, edges, segments for each constraint,
            and any constraints for which no path was found. It contains the following keys:
                - "output_type": The resolved output node type.
                - "constraints": A list of resolved constraint node types.
                - "nodes": A sorted list of all unique nodes involved in any alternative path for any constraint.
                - "edges": A list of unique edges involved in any alternative path, where each edge is represented as a dictionary
                  with keys "from", "to", "predicate", "label", "graph", "subject_type", "object_type", and "reverse" (True if
                  "from"/"to" are flipped relative to "subject_type"/"object_type" for this hop -- see
                  PathPlanner._all_shortest_paths).
                - "triples": A deduplicated list of SPARQL-ready triple patterns derived from "edges" (across all alternatives),
                  intended for an LLM (or template) to turn directly into a query. Each item is a dict with keys "subject",
                  "predicate", "predicate_label", "object", and "graph" (a full named-graph IRI). "subject"/"object" are entity-type
                  node ids that double as SPARQL variable names (e.g. "?gene"); the correct triple is always
                  "GRAPH <graph> { ?<subject> <predicate> ?<object> . }" -- no direction reasoning ("reverse"/"from"/"to") is
                  needed when using this list. NOTE: when a constraint has multiple alternative paths (e.g. several distinct
                  predicates between the same two types), "triples"/"edges" include ALL of them -- pick the one(s) matching the
                  question's semantics rather than including every alternative in the final SPARQL query.
                - "segments": A dictionary mapping each original constraint to a list of equally-short alternative paths (each a
                  list of edges, in order from output to constraint), or None if no path was found within hop_limit. Usually a
                  single-element list; more than one element means there was a tie (see PathPlanner._all_shortest_paths).
                - "no_path": A list of original constraints for which no path was found to the output type.
        """
        if not self._is_known_type(output):
            warnings.warn(
                f"plan(): output type {output!r} is not a recognized entity type "
                f"(see VALID_OBJECT_TYPES in builder.py); no paths can be found.",
                stacklevel=2,
            )

        output_node = _resolve(output)

        all_nodes = {output_node}
        all_edges = []
        seen_edge_keys = set()
        all_triples = []
        seen_triple_keys = set()
        segments = {}
        no_path = []

        for orig in constraints:
            if not self._is_known_type(orig):
                warnings.warn(
                    f"plan(): constraint {orig!r} is not a recognized entity type "
                    f"(see VALID_OBJECT_TYPES in builder.py); it will get no_path.",
                    stacklevel=2,
                )

            target = _resolve(orig)

            # Find ALL shortest paths from output to target (handles ties explicitly).
            # If output == target, this returns [[]] (a single, trivial empty path).
            alt_paths = self._all_shortest_paths(output_node, target, hop_limit)
            # If no path is found, we add the original constraint to the no_path list and set its segment to None
            if alt_paths is None:
                no_path.append(orig)
                segments[orig] = None
                continue

            # If (a) path(s) is/are found, we add them to the segments and update the
            # all_nodes, all_edges and all_triples collections with edges from EVERY alternative.
            segments[orig] = alt_paths
            for path_edges in alt_paths:
                for e in path_edges:
                    all_nodes.add(e["from"])
                    all_nodes.add(e["to"])
                    key = (e["from"], e["to"], e["predicate"], e["graph"])
                    if key not in seen_edge_keys:
                        seen_edge_keys.add(key)
                        all_edges.append(e)

                    # Triples are keyed on the true (subject, predicate, object, graph) so a
                    # forward and a reverse traversal of the same edge collapse to one triple.
                    triple_key = (e["subject_type"], e["predicate"], e["object_type"], e["graph"])
                    if triple_key not in seen_triple_keys:
                        seen_triple_keys.add(triple_key)
                        all_triples.append(self._edge_to_triple(e))

        return {
            "output_type": output_node,
            "constraints": [_resolve(c) for c in constraints],
            "nodes": sorted(all_nodes),
            "edges": all_edges,
            "triples": all_triples,
            "segments": segments,
            "no_path": no_path,
        }

    def _bfs_distances(self, start: str, hop_limit: int) -> dict[str, int]:
        """Method that performs a breadth-first search (BFS) to find the minimum hop-distance
        from start to every node reachable within hop_limit.

        Parallel edges between the same pair of nodes don't affect distance (they're all
        the same number of hops), so this BFS works over nodes, not edges.

        Parameters:
            start (str): The starting node of the search.
            hop_limit (int): The maximum number of hops to explore.

        Returns:
            dict[str, int]: Maps each node reachable from start (within hop_limit hops, inclusive)
            to its minimum hop-distance from start. start itself maps to 0.
        """
        dist = {start: 0}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if dist[node] >= hop_limit:
                continue
            for neighbour, _ in self._adj.get(node, []):
                if neighbour not in dist:
                    dist[neighbour] = dist[node] + 1
                    queue.append(neighbour)

        return dist

    def _all_shortest_paths(self, start: str, goal: str, hop_limit: int) -> list[list[dict]] | None:
        """Method that finds ALL shortest paths from start to goal, handling ties.

        The graph is a multigraph: a node pair can be connected by several edges with
        different predicates (e.g. gene<->protein via "encodes" and three tfac2gene
        regulation predicates). When more than one path of the minimum length exists --
        whether because of such parallel edges or because multiple distinct intermediate
        nodes lead to the goal in the same number of hops -- ALL of them are returned
        (up to MAX_ALTERNATIVES), so the caller can pick the one matching the question's
        semantics instead of getting an arbitrary one silently.

        Parameters:
            start (str): The starting node of the search.
            goal (str): The target node of the search.
            hop_limit (int): The maximum number of hops to consider.

        Returns:
            list[list[dict]] | None: A list of alternative paths, each a list of edge dictionaries
            (in order from start to goal, each annotated with "reverse" -- see plan()'s docstring),
            or None if goal is unreachable from start within hop_limit. If start == goal, returns
            [[]] (a single, empty-path alternative).
        """
        dist = self._bfs_distances(start, hop_limit)
        if goal not in dist:
            return None

        target_len = dist[goal]
        results: list[list[dict]] = []
        path: list[dict] = []

        def dfs(node: str, depth: int) -> None:
            if len(results) >= MAX_ALTERNATIVES:
                return
            if depth == target_len:
                if node == goal:
                    results.append(list(path))
                return
            for neighbour, eidx in self._adj.get(node, []):
                # Only follow edges that make progress towards goal along a shortest path.
                if dist.get(neighbour) != depth + 1:
                    continue
                edge = self._edges[eidx]
                path.append({**edge, "reverse": edge["from"] != edge["subject_type"]})
                dfs(neighbour, depth + 1)
                path.pop()
                if len(results) >= MAX_ALTERNATIVES:
                    return

        dfs(start, 0)
        return results

    def _is_known_type(self, type_str: str) -> bool:
        """Method that checks whether type_str resolves (via alias + normalize) to a node that
        actually exists in the loaded path map.
        
        Parameters:
            type_str (str): The input type string to check.
        
        Returns:
            bool: True if the resolved type string is a known node in the path map, False otherwise.
        """
        return _resolve(type_str) in self._known_nodes


if __name__ == "__main__":

    # Demo: Show the full merged path for the main validation query
    planner = PathPlanner()
    r = planner.plan(output="gene", constraints=["organism", "biological_process"])
    print("\n--- Demo plan(gene, [organism, biological_process]) ---")
    print_plan(r)

    # Demo: a query whose output sits on the "object" side of its only edge to
    # the constraint, exercising the "reverse" traversal flag.
    r2 = planner.plan(output="protein", constraints=["gene"])
    print("\n--- Demo plan(protein, [gene]) ---")
    print_plan(r2)
