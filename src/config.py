import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "qwen3:32b"
#"qwen3:32b" "qwen2.5:7b"

# How many times to retry if the model returns malformed JSON
EXTRACTOR_MAX_RETRIES = 3

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
