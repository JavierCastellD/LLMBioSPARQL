# LLMBioSPARQL

A natural-language to SPARQL pipeline for [BioGateway](http://biogateway.eu), a biomedical RDF knowledge graph that integrates data about human genes, proteins, cis-regulatory modules (CRMs/enhancers), transcription factors, phenotypes/diseases, and Gene Ontology biological processes.

An LLM (served locally via [Ollama](https://ollama.com)) is used only for the well-scoped sub-tasks that need natural-language understanding — identifying concepts, choosing an attribute, proposing a search term, picking among candidate paths, assembling the final query. Everything that can be resolved deterministically against the graph's schema (path finding) or validated live against the endpoint (candidate resolution, query execution) is not left to the LLM.

---

## Project structure

```
LLMBioSPARQL/
├── src/
│   ├── config.py                  # Ollama/model settings, prompt names, retry limits, SPARQL endpoint config
│   ├── pipeline.py                # Text2SPARQLPipeline — wires all six stages together
│   ├── concept_extraction/
│   │   ├── extractor.py           # ConceptExtractor — stage 1 (LLM)
│   │   └── prompts/
│   ├── output_designation/
│   │   ├── designator.py          # OutputDesignator — stage 2 (LLM)
│   │   └── prompts/
│   ├── entity_linker/
│   │   ├── linker.py              # EntityLinker — stage 3 (LLM proposal + live SPARQL validation)
│   │   ├── sparql_probe.py        # Runs candidate lookups against the live endpoint
│   │   ├── label_config.py        # Which label predicates to search per entity type
│   │   └── prompts/
│   ├── pathmap/
│   │   ├── builder.py             # Reads data/BGW_graphs.xlsx, writes data/pathmap.json
│   │   └── planner.py             # PathPlanner — stage 4 (deterministic, no LLM)
│   ├── sparql_generator/
│   │   ├── path_resolver.py       # PathResolver — stage 5 (live validation + LLM tiebreak)
│   │   ├── attribute_resolver.py  # Resolves output-designation attributes to predicates
│   │   ├── brief_builder.py       # Assembles the generation brief from all prior stages
│   │   ├── generator.py           # SPARQLGenerator — stage 6 (LLM + live validation/repair)
│   │   └── prompts/
│   └── utils/
│       ├── llm.py                 # OllamaLLM client
│       └── prompts_loader.py      # Loads prompts/*.txt as string.Template templates
├── tests/
│   ├── run_question.py            # Run one question through the full pipeline, print every stage
│   ├── run_concept_extraction.py  # Batch-run stage 1 over a list of questions
│   ├── run_output_designation.py  # Batch-run stage 2
│   ├── run_linker.py              # Batch-run stage 3
│   ├── run_planner.py             # Batch-run stage 4 (deterministic)
│   ├── run_sparql_generator.py    # Batch-run stages 4-6 from gold stage 1-3 inputs (isolated)
│   └── run_pipeline.py            # Batch-run the full pipeline end to end (chained)
├── data/
│   ├── BGW_graphs.xlsx            # Spreadsheet describing BioGateway's named graphs/predicates
│   └── pathmap.json               # Generated path map consumed by PathPlanner
├── queries/                        # Example/gold question-SPARQL pairs — see queries/README.md
└── requirements.txt
```

---

## Pipeline

The pipeline converts a natural-language question into a SPARQL query through six stages (`src/pipeline.py`). Each stage produces a structured intermediate representation consumed by the next.

### 1 — Concept extraction (`src/concept_extraction/extractor.py`)

An LLM turns the question into a small typed graph: nodes (one of 11 fixed entity types, e.g. `gene`, `protein`, `biological process`) linked by triples drawn from a fixed relation vocabulary (`RELATION_VOCAB` in `src/pathmap/planner.py`). One or more nodes are marked as `outputs`. Nodes without a `"span"` are unnamed bridge concepts the LLM introduced to connect two others (e.g. the protein between a gene and a biological process). The response is validated and retried up to `EXTRACTOR_MAX_RETRIES` times on malformed output.

### 2 — Output designation (`src/output_designation/designator.py`)

For each output node, an LLM picks which literal attribute the question is actually asking for (name, definition, synonym, position, etc.) from a small suggested vocabulary.

### 3 — Entity linking (`src/entity_linker/linker.py`)

For every non-output, spanned node, an LLM proposes a search term, whether a regex filter is needed, and which label predicates to try. The proposal is validated live against BioGateway (`sparql_probe.py`): a single match resolves to a fixed URI (`bind` mode), several matches are left as a filter for the generator to apply (`filter`/`many` mode), and zero matches trigger a reformulation loop (up to `LINKER_MAX_ITERATIONS`).

### 4 — Path planning (`src/pathmap/planner.py`)

Fully deterministic, no LLM. For each triple, resolves the relation name and node types to BioGateway's schema (loaded from `data/pathmap.json`) and finds the shortest path connecting them, inserting hidden bridge nodes where the graph requires an intermediate hop.

### 5 — Path resolution (`src/sparql_generator/path_resolver.py`)

When more than one equal-length candidate path satisfies a triple, each is validated live against the endpoint (discarding empty ones) and an LLM breaks the tie among the remaining populated candidates.

### 6 — SPARQL generation (`src/sparql_generator/generator.py`)

An LLM assembles the outputs of every previous stage (via `brief_builder.py`) into an executable SPARQL query. The query is run against the live endpoint; syntactic or structural errors trigger up to `SPARQL_GENERATOR_MAX_REPAIR` repair attempts.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally, with the configured model pulled

```powershell
pip install -r requirements.txt

ollama serve                 # start the Ollama daemon
ollama pull qwen3:32b        # pull the model (requires the daemon to be running)
```

---

## Configuration

Edit `src/config.py` to change the model, prompts, retry behaviour, or SPARQL endpoint. The Ollama host and model can also be overridden via environment variables without touching the file:

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
$env:OLLAMA_MODEL = "qwen3:32b"
```

---

## Running

Run a single question through the full pipeline and print every stage's output:

```powershell
python tests/run_question.py "Which genes are involved in DNA repair in humans?"
```

Rebuild `data/pathmap.json` from `data/BGW_graphs.xlsx` (only needed after editing the spreadsheet):

```powershell
python -m src.pathmap.builder
```

Batch-run the full pipeline over a set of questions (each item just needs a `"question"` key):

```powershell
python tests/run_pipeline.py queries/test_examples.json results/pipeline/my_run_pipeline_out.json
```

Batch-run an individual stage, or the tail of the pipeline (stages 4-6) fed with gold intermediate results to evaluate it in isolation — see `queries/test_examples.json` for the expected input schema and `tests/run_*.py` docstrings for each script's exact input/output shape.

### Evaluation

`evaluation.ipynb` scores each stage independently against `queries/test_examples.json` (concept extraction, output designation, entity linking, path planning/resolution), then scores SPARQL generation both in isolation and as the full chained pipeline — plus a separate section against the harder, hand-authored `queries/manual_examples.json`. Only `test_examples.json`/`manual_examples.json` are used for evaluation; `train_examples.json` supplies the in-prompt few-shot examples and is never scored.
