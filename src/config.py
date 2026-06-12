OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Whether to enable "thinking"/reasoning traces for models that support it
# (e.g. Qwen3). Disabled by default since the reasoning trace can take
# several minutes and is not needed for this extraction task.
OLLAMA_THINK = False

# How many times to retry if the model returns malformed JSON
EXTRACTOR_MAX_RETRIES = 3

# Name of the prompt template (under src/concept_extraction/prompts/, without
# the ".txt" extension) used for concept extraction.
CONCEPT_EXTRACTION_PROMPT = "concept_extraction"

# --- Entity linker ---------------------------------------------------------

# BioGateway SPARQL endpoint. SSL verification is disabled because the server
# certificate does not validate against the default CA bundle.
BIOGATEWAY_SPARQL_ENDPOINT = "https://semantics.inf.um.es/biogateway"
SPARQL_VERIFY_SSL = False
SPARQL_TIMEOUT = 60  # seconds

# Named graphs live under this base; e.g. the "gene" hint -> .../graph/gene
BIOGATEWAY_GRAPH_BASE = "http://rdf.biogateway.eu/graph/"

# Max tool-calling iterations in the linker's LLM loop before giving up.
LINKER_MAX_ITERATIONS = 5
