# Text2SPARQL for BioGateway

A natural-language to SPARQL pipeline for [BioGateway](http://biogateway.eu), a biomedical RDF knowledge graph that integrates data about human genes, proteins, cis-regulatory modules (CRMs/enhancers), transcription factors, phenotypes/diseases, and Gene Ontology biological processes.

---

## Project structure

```
Text2SPARQL/
├── src/
│   ├── config.py        # Ollama model and endpoint settings
│   ├── prompts.py       # System prompt and few-shot examples fed to the LLM
│   └── extractor.py     # QueryExtractor — stage 1 of the pipeline
├── tests/
│   └── test_extractor.py
├── queries/             # Example questions and reference SPARQL queries
│   └── README.md
├── schemas/             # RDF/Turtle schemas for each BioGateway named graph
└── requirements.txt
```

---

## Pipeline

The pipeline converts a natural-language question into a SPARQL query through several stages. Each stage produces a structured intermediate representation consumed by the next.

### Stage 1 — Query Extractor (`src/extractor.py`)

Takes a plain-English question and returns a dict with three keys:

| Key | Type | Description |
|---|---|---|
| `entities` | `list[str]` | Named biomedical entities mentioned verbatim (gene symbols, protein accessions, disease names, chromosomal positions, …) |
| `output` | `list[str]` | What the user wants to retrieve (e.g. `"protein"`, `"biological process"`, `"start position"`) |
| `relationships` | `list[dict]` | Semantic triples `{subject, predicate, object}` implied by the question |

Further stages (schema selection, SPARQL generation) are not yet implemented.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally

```powershell
pip install -r requirements.txt

ollama serve                 # start the Ollama daemon
ollama pull qwen2.5:7b       # pull the model (requires the daemon to be running)
```

---

## Configuration

Edit `src/config.py` to change the model or endpoint:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:7b"
EXTRACTOR_MAX_RETRIES = 3
```
