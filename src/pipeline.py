"""End-to-end orchestrator: question -> concept extraction -> output designation
-> entity linking -> path planning -> SPARQL generation.

Each stage is an existing, independently-built/tested class; this module only
wires them together and converts between their input/output shapes. No new LLM
prompts or fact-resolution logic live here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .concept_extraction.extractor import ConceptExtractor
from .entity_linker.linker import EntityLinker
from .output_designation.designator import OutputDesignator
from .pathmap.planner import PathPlanner
from .sparql_generator.generator import SPARQLGenerator
from .utils.llm import LLMCallable


def _span_nodes_from_extraction(extraction: dict) -> list[dict]:
    """Build the entity-linker's span_nodes list: every node that carries a span.

    Parameters:
        extraction (dict):
            Concept-extraction result with a "nodes" key.

    Returns:
        List of {"node_id", "span", "type"} dicts, one per spanned node
        (output or not — an output node with a span, e.g. a named attribute
        query, needs linking too).
    """
    return [
        {"node_id": nid, "span": node["span"], "type": node["type"]}
        for nid, node in extraction["nodes"].items()
        if "span" in node
    ]


def _output_nodes_from_extraction(extraction: dict) -> list[dict]:
    """Build output-designation's output_nodes list from "outputs" + "nodes".

    Parameters:
        extraction (dict):
            Concept-extraction result with "outputs" and "nodes" keys.

    Returns:
        List of {"node_id", "type", "span"?} dicts, one per output node.
    """
    nodes = extraction["nodes"]
    result = []
    for nid in extraction["outputs"]:
        node = nodes[nid]
        entry = {"node_id": nid, "type": node["type"]}
        if "span" in node:
            entry["span"] = node["span"]
        result.append(entry)
    return result


class Text2SPARQLPipeline:
    """Runs a natural-language question through every pipeline stage end to end.

    Attributes:
        extractor (ConceptExtractor):
            Stage 1 — question -> typed nodes/triples.
        designator (OutputDesignator):
            Stage 2 — which literal attributes the question asks for.
        linker (EntityLinker):
            Stage 3 — grounds spans to BioGateway URIs/filters.
        planner (PathPlanner):
            Stage 4 — resolves relation triples to predicate paths.
        generator (SPARQLGenerator):
            Stage 5 — authors the SPARQL query from an assembly brief.
    """

    def __init__(
        self,
        llm_client: LLMCallable | None = None,
        excel_path: Path | None = None,
        pathmap_path: Path | None = None,
    ) -> None:
        """Initializes one instance of each pipeline stage.

        Parameters:
            llm_client (LLMCallable):
                Callable (prompt: str) -> str, shared by every LLM-backed stage.
                If None, each stage constructs its own default OllamaLLM.
            excel_path (Path):
                Path to BGW_graphs.xlsx, shared by the linker and generator.
                If None, each stage uses its own default project-relative path.
            pathmap_path (Path):
                Path to pathmap.json for the planner. If None, uses the default
                project-relative path.
        """
        self.extractor = ConceptExtractor(llm_client=llm_client)
        self.designator = OutputDesignator(llm_client=llm_client)
        self.linker = EntityLinker(llm_client=llm_client, excel_path=excel_path)
        self.planner = PathPlanner(pathmap_path)
        self.generator = SPARQLGenerator(llm_client=llm_client, excel_path=excel_path)

    def run(self, question: str) -> dict[str, Any]:
        """Run a question through every stage and return the full trace.

        Parameters:
            question (str):
                The natural-language question to convert to SPARQL.

        Returns:
            A dict with "question" plus one key per stage: "extraction",
            "output_designation", "linker", "plan", "generation" — so callers
            can inspect any intermediate result. If concept extraction fails,
            every later key is absent and "skipped" explains why. Output
            designation failing does not stop the pipeline (falls back to no
            requested attributes, fail-soft, matching that stage's own contract).
        """
        result: dict[str, Any] = {"question": question}

        extraction = self.extractor.extract(question)
        result["extraction"] = extraction
        if "error" in extraction:
            result["skipped"] = "concept extraction failed"
            return result

        output_nodes = _output_nodes_from_extraction(extraction)
        designation = self.designator.designate(question, output_nodes)
        result["output_designation"] = designation
        if "error" in designation:
            designation = {"attributes": []}

        span_nodes = _span_nodes_from_extraction(extraction)
        linker_output = self.linker.link(question, span_nodes)
        result["linker"] = {"input": span_nodes, "output": linker_output}

        plan = self.planner.resolve_relations(extraction["nodes"], extraction["triples"])
        result["plan"] = plan

        result["generation"] = self.generator.generate(
            question, extraction, plan, span_nodes, linker_output, designation
        )

        return result
