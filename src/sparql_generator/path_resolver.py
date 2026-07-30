"""Path resolver: choose one candidate segment among a resolved edge's
equal-length alternatives, before the brief is built (see
docs/path_resolver_spec.md).

The planner already enumerates every equal-length candidate path for a
relational edge in resolved[*]["segments"]; brief_builder previously always
used segments[0] and silently discarded the rest, which picks the wrong path
whenever the first alternative isn't the semantically intended one (e.g. a
CRM-mediated route instead of the direct gene2phen route).

Division of labour:
  Code — builds an entity-stripped, LIMIT-1 probe per candidate to classify it
         populated/dead (Tier 2).
  LLM  — 1-of-N tiebreak among semantically valid candidates, when probing
         can't settle it alone or isn't available (Tiers 3/4/5).

The LLM client is an injectable callable (str -> str) so any backend works.
"""
from __future__ import annotations

from pathlib import Path

import requests
import urllib3

from ..config import (
    BIOGATEWAY_GRAPH_BASE,
    BIOGATEWAY_SPARQL_ENDPOINT,
    PATH_RESOLVER_MAX_RETRIES,
    PATH_RESOLVER_TIEBREAK_PROMPT,
    SPARQL_TIMEOUT,
    SPARQL_VERIFY_SSL,
)
from ..utils.llm import LLMCallable, OllamaLLM, llm_call_with_retry
from ..utils.prompts_loader import render_prompt

if not SPARQL_VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def hop_orientation(hop: dict, before_id: str, after_id: str) -> tuple[str, str]:
    """Return (subject_id, object_id) for one hop, honoring its "reverse" flag.

    Shared by brief_builder's edge expansion and this module's probe
    construction, so both wire a hop's direction identically: traversal runs
    before_id -> after_id, but the TRUE RDF triple runs the other way when
    "reverse" is set.

    Parameters:
        hop (dict):
            A single segment hop with a "reverse" flag.
        before_id (str):
            The node id/variable on the "from" side of the traversal.
        after_id (str):
            The node id/variable on the "to" side of the traversal.

    Returns:
        (subject_id, object_id) in true RDF subject->object order.
    """
    return (after_id, before_id) if hop["reverse"] else (before_id, after_id)


def _describe_segment(segment: list[dict]) -> str:
    """Render a one-line-per-hop description of a candidate path for the LLM tiebreak.

    Uses the same reverse-arrow convention as pathmap.planner.print_resolved(),
    plus each hop's graph name, e.g.:
      protein <-[involved in]- biological_process [prot2bp]
      gene <-[encodes]- protein [gene]
      gene -[involved in]-> phenotype [gene2phen]

    Parameters:
        segment (list[dict]):
            One candidate path: a list of hops with from/to/label/reverse/graph.

    Returns:
        The multi-line description string.
    """
    lines = []
    for hop in segment:
        arrow = f"<-[{hop['label']}]-" if hop["reverse"] else f"-[{hop['label']}]->"
        lines.append(f"{hop['from']} {arrow} {hop['to']} [{hop['graph']}]")
    return "\n".join(lines)


def _build_probe_query(segment: list[dict]) -> str:
    """Build an entity-stripped, LIMIT-1 SPARQL probe for one candidate path.

    Tests only whether the path is structurally populated anywhere in the
    database — no grounding literals, no other edges, no ontology regex — so
    a "dead" result means the CHAIN itself is unused, not that one specific
    entity lacks this association.

    Parameters:
        segment (list[dict]):
            One candidate path: a list of hops with predicate/graph/reverse.

    Returns:
        A SPARQL SELECT query string with one GRAPH block per hop, LIMIT 1.
    """
    var_names = [f"v{i}" for i in range(len(segment) + 1)]
    triples = []
    for i, hop in enumerate(segment):
        subj, obj = hop_orientation(hop, var_names[i], var_names[i + 1])
        graph_iri = BIOGATEWAY_GRAPH_BASE + hop["graph"]
        triples.append(f"  GRAPH <{graph_iri}> {{ ?{subj} <{hop['predicate']}> ?{obj} . }}")
    body = "\n".join(triples)
    return f"SELECT ?{var_names[0]} ?{var_names[-1]} WHERE {{\n{body}\n}} LIMIT 1"


def _node_span(nodes: dict[str, dict], node_id: str) -> str:
    """Return a node's quoted span for display, or a fallback for unnamed nodes.

    Parameters:
        nodes (dict[str, dict]):
            Concept-extraction's node id -> {"type", "span"?} mapping.
        node_id (str):
            The node id to look up (may be a bridge-internal id with no span).

    Returns:
        '"<span>"' if the node names an explicit phrase from the question,
        otherwise a fallback noting it's the question's own unnamed output.
    """
    span = nodes.get(node_id, {}).get("span")
    return f'"{span}"' if span else "(unnamed — this is the question's own output, not a phrase in the text)"


def build_tiebreak_prompt(
    question: str,
    entry: dict,
    candidates: dict[int, str],
    nodes: dict[str, dict],
) -> str:
    """Render the 1-of-N path tiebreak prompt.

    Parameters:
        question (str):
            The full natural-language question (for semantic context only).
        entry (dict):
            The resolved edge being disambiguated: subject, object (node ids),
            subject_type, object_type, and relation (the relation name concept
            extraction assigned to this specific edge).
        candidates (dict[int, str]):
            Surviving candidate index -> its readable path description.
            Deliberately excludes probe row counts — see docs/path_resolver_spec.md.
        nodes (dict[str, dict]):
            Concept-extraction's node id -> {"type", "span"?} mapping, used to
            show which phrase in the question each end of this edge refers to
            (the question may name several entities; this identifies which
            ones THIS specific edge connects).

    Returns:
        The fully rendered prompt string ready to send to the LLM.
    """
    candidates_text = "\n\n".join(f"Candidate {i}:\n{desc}" for i, desc in candidates.items())
    return render_prompt(
        PATH_RESOLVER_TIEBREAK_PROMPT,
        prompts_dir=_PROMPTS_DIR,
        QUESTION=question,
        RELATION=entry["relation"],
        SUBJECT_TYPE=entry["subject_type"],
        SUBJECT_SPAN=_node_span(nodes, entry["subject"]),
        OBJECT_TYPE=entry["object_type"],
        OBJECT_SPAN=_node_span(nodes, entry["object"]),
        CANDIDATES=candidates_text,
        CANDIDATE_INDICES=", ".join(str(i) for i in candidates),
    )


def _validate_tiebreak(parsed, candidate_indices: list[int]) -> list[str]:
    """Validate the LLM's 1-of-N tiebreak response.

    Parameters:
        parsed (Any):
            The JSON value returned by extract_json (possibly None).
        candidate_indices (list[int]):
            The indices the LLM was allowed to choose from.

    Returns:
        A list of human-readable problems found, or an empty list if valid.
    """
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]
    chosen = parsed.get("chosen_index")
    if not isinstance(chosen, int) or chosen not in candidate_indices:
        return [f'"chosen_index" must be one of {candidate_indices}']
    return []


class PathResolver:
    """Chooses one candidate segment among a resolved edge's equal-length alternatives.

    Attributes:
        llm_client (LLMCallable):
            Callable (prompt: str) -> str. Defaults to OllamaLLM.
        endpoint (str):
            SPARQL endpoint URL for Tier 2 probing.
        timeout (int):
            Per-probe request timeout in seconds.
        verify_ssl (bool):
            Whether to verify SSL certificates for probing.
    """

    def __init__(
        self,
        llm_client: LLMCallable | None = None,
        endpoint: str = BIOGATEWAY_SPARQL_ENDPOINT,
        timeout: int = SPARQL_TIMEOUT,
        verify_ssl: bool = SPARQL_VERIFY_SSL,
    ) -> None:
        """Initializes the attributes of the class.

        Parameters:
            llm_client (LLMCallable):
                Callable (prompt: str) -> str. Defaults to OllamaLLM.
            endpoint (str):
                SPARQL endpoint URL for Tier 2 probing.
            timeout (int):
                Per-probe request timeout in seconds.
            verify_ssl (bool):
                Whether to verify SSL certificates for probing.
        """
        self._llm = llm_client or OllamaLLM()
        self._endpoint = endpoint
        self._timeout = timeout
        self._verify_ssl = verify_ssl

    def resolve(self, question: str, entry: dict, nodes: dict[str, dict]) -> dict:
        """Choose one segment among a resolved edge's candidate alternatives.

        Parameters:
            question (str):
                The full natural-language question (context for the LLM tiebreak).
            entry (dict):
                One item from PathPlanner.resolve_relations()["resolved"]: a dict
                with subject, object, subject_type, object_type, relation, and
                "segments" (a list of candidate paths, each a list of hops).
            nodes (dict[str, dict]):
                Concept-extraction's node id -> {"type", "span"?} mapping, so
                an LLM tiebreak can show which phrase in the question this
                specific edge's subject/object refer to (see
                path_resolver.build_tiebreak_prompt).

        Returns:
            A dict {"segment": the chosen candidate (list of hops), "record":
            {"tier": 1-5, "candidates": N, ...}} — the record is for
            logging/inspection only, never sent to the LLM as part of the brief.
        """
        segments = entry["segments"]

        if len(segments) == 1:
            return {"segment": segments[0], "record": {"tier": 1, "candidates": 1, "validated": True}}

        populated: list[int] = []
        probe_failed = False
        for i, seg in enumerate(segments):
            try:
                count = self._probe_count(seg)
            except Exception:
                probe_failed = True
                break
            if count > 0:
                populated.append(i)

        if probe_failed:
            # The endpoint is reachable in principle, but this specific probe
            # genuinely failed (timeout/connection error). Never count an
            # error as "dead" — degrade the WHOLE edge to an unvalidated
            # tiebreak instead of silently mis-eliminating a candidate.
            return self._llm_tiebreak(
                question, entry, segments, list(range(len(segments))), nodes, validated=False
            )

        if len(populated) == 1:
            i = populated[0]
            return {
                "segment": segments[i],
                "record": {
                    "tier": 2,
                    "candidates": len(segments),
                    "populated": populated,
                    "chosen": i,
                    "validated": True,
                },
            }

        if len(populated) > 1:
            return self._llm_tiebreak(question, entry, segments, populated, nodes, validated=True)

        # len(populated) == 0: possible data gap (empty != wrong), not a crash.
        return self._llm_tiebreak(
            question, entry, segments, list(range(len(segments))), nodes, validated=False, all_empty=True
        )

    def _probe_count(self, segment: list[dict]) -> int:
        """Run a Tier-2 probe for one candidate and return its row count (0 or 1).

        Parameters:
            segment (list[dict]):
                One candidate path.

        Returns:
            1 if the path is populated anywhere in the database, 0 if dead.
            Raises on a genuine request failure (timeout, connection error).
        """
        query = _build_probe_query(segment)
        resp = requests.post(
            self._endpoint,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=self._timeout,
            verify=self._verify_ssl,
        )
        resp.raise_for_status()
        return len(resp.json()["results"]["bindings"])

    def _llm_tiebreak(
        self,
        question: str,
        entry: dict,
        segments: list[list[dict]],
        candidate_indices: list[int],
        nodes: dict[str, dict],
        validated: bool,
        all_empty: bool = False,
    ) -> dict:
        """Ask the LLM to choose 1-of-N among the given candidate indices.

        Parameters:
            question (str):
                The full natural-language question.
            entry (dict):
                The resolved edge (subject/object/subject_type/object_type/relation).
            segments (list[list[dict]]):
                All candidate paths for this edge.
            candidate_indices (list[int]):
                Which candidates the LLM may choose among (populated ones for
                Tier 3, all of them for Tiers 4/5).
            nodes (dict[str, dict]):
                Concept-extraction's node id -> {"type", "span"?} mapping, used
                to show which phrase in the question this edge connects.
            validated (bool):
                Whether the chosen candidate is data-confirmed populated.
            all_empty (bool):
                True for Tier 5 (every probe came back empty).

        Returns:
            A dict {"segment": ..., "record": {...}}, falling back to the
            first candidate in candidate_indices if the LLM never returns a
            valid choice (never crashes, never yields an empty brief).
        """
        descriptions = {i: _describe_segment(segments[i]) for i in candidate_indices}
        prompt = build_tiebreak_prompt(question, entry, descriptions, nodes)
        parsed = llm_call_with_retry(
            self._llm, prompt, lambda p: _validate_tiebreak(p, candidate_indices), PATH_RESOLVER_MAX_RETRIES
        )
        chosen = parsed["chosen_index"] if parsed is not None else candidate_indices[0]

        record = {
            "tier": 5 if all_empty else (3 if validated else 4),
            "candidates": len(segments),
            "considered": candidate_indices,
            "chosen": chosen,
            "validated": validated,
        }
        if all_empty:
            record["all_candidates_empty"] = True
            record["uncertain"] = True

        return {"segment": segments[chosen], "record": record}
