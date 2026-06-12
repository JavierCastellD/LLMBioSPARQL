# Text2SPARQL for BioGateway

A natural-language to SPARQL pipeline for [BioGateway](http://biogateway.eu), a biomedical RDF knowledge graph that integrates data about human genes, proteins, cis-regulatory modules (CRMs/enhancers), transcription factors, phenotypes/diseases, and Gene Ontology biological processes.

---

## Project structure

```
Text2SPARQL/
├── src/
│   ├── config.py                  # Ollama settings, retry limits, SPARQL endpoint config
│   ├── concept_extraction/
│   │   ├── extractor.py           # ConceptExtractor — stage 1 of the pipeline (LLM-based)
│   │   ├── prompts_loader.py       # Loads prompts/*.txt as string.Template templates
│   │   └── prompts/
│   │       └── concept_extraction.txt
│   └── pathmap/
│       ├── builder.py             # Reads data/BGW_graphs.xlsx, writes data/pathmap.json
│       └── planner.py             # PathPlanner — stage 2 of the pipeline (deterministic)
├── tests/
│   ├── run_concept_extraction.py  # Batch-runs ConceptExtractor over a list of questions
│   └── run_planner.py             # Batch-runs PathPlanner over concept-extraction results
├── data/
│   ├── BGW_graphs.xlsx            # Spreadsheet describing BioGateway's named graphs
│   └── pathmap.json               # Generated path map consumed by PathPlanner
├── queries/                        # Example questions and reference SPARQL queries
└── requirements.txt
```

---

## Pipeline

The pipeline converts a natural-language question into a SPARQL query through several stages. Each stage produces a structured intermediate representation consumed by the next.

### Stage 1 — Concept extraction (`src/concept_extraction/extractor.py`)

`ConceptExtractor.extract(question)` uses an LLM (via Ollama) to turn a plain-English question into a small typed graph:

```json
{
  "nodes": {
    "OUTPUT": {"type": "gene"},
    "n1": {"type": "biological process", "span": "DNA repair"},
    "n2": {"type": "organism", "span": "humans"}
  },
  "triples": [
    {"subject": "OUTPUT", "relation": "involved in", "object": "n1"},
    {"subject": "OUTPUT", "relation": "in taxon", "object": "n2"}
  ]
}
```

- Exactly one node has the id `"OUTPUT"`: the entity the question asks to retrieve.
- Other nodes are named entities from the question (with a `"span"`) or unnamed intermediate connector nodes (e.g. `"x1"`, no `"span"`).
- Each triple's `"relation"` is one of the coarse relation names in `RELATION_NAMES` (imported from `src.pathmap.planner`).
- The response is validated (node types, relation names, connectivity to `OUTPUT`); an invalid response triggers up to `EXTRACTOR_MAX_RETRIES` repair attempts before returning a structured `{"error": ...}`.

### Stage 2 — Path planning (`src/pathmap/planner.py`)

`PathPlanner.resolve_relations(nodes, triples)` is fully deterministic (no LLM). For each triple, it:

1. Resolves the relation name to one or more path-map predicate labels (`RELATION_VOCAB`).
2. Resolves the subject/object entity types to path-map node types.
3. Finds the shortest path between those types whose terminal edge matches one of the resolved labels — inserting intermediate "hidden bridge" nodes if needed (e.g. `gene -[encodes]-> protein -[involved in]-> biological_process`).

It returns the merged set of nodes/edges/SPARQL-ready triples across all input triples, plus per-triple `resolved`/`no_path` diagnostics.

Further stages (SPARQL query assembly) are not yet implemented.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally (only needed for Stage 1)

```powershell
pip install -r requirements.txt

ollama serve                 # start the Ollama daemon
ollama pull qwen2.5:7b       # pull the model (requires the daemon to be running)
```

---

## Configuration

Edit `src/config.py` to change the model, endpoint, or retry behaviour:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"
EXTRACTOR_MAX_RETRIES = 3
```

---

## Running

Validate the planner (deterministic, no LLM needed):

```powershell
python -m src.pathmap.planner
```

Rebuild `data/pathmap.json` from `data/BGW_graphs.xlsx`:

```powershell
python -m src.pathmap.builder
```

Run concept extraction over a batch of questions (requires Ollama running), then feed the results into the planner:

```powershell
python tests/run_concept_extraction.py queries/examples_simple.json extraction_results.json
python tests/run_planner.py extraction_results.json planner_results.json
```

`run_concept_extraction.py` expects a JSON file containing a list of `{"question": ...}` objects and writes a list of `{"question", "extractor"}` objects. `run_planner.py` consumes that output directly (each item's `"extractor"` field) and writes a list of `{"question", "result"}` objects with the planner's `resolve_relations()` output.
