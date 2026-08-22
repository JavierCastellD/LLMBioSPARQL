import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:32b")
#"qwen3:32b" "qwen2.5:7b"

# Whether to enable "thinking"/reasoning traces for models that support it
# (e.g. Qwen3). Disabled by default since the reasoning trace can take
# several minutes and is not needed for this extraction task.
OLLAMA_THINK = False

# How many times to retry if the model returns malformed JSON
EXTRACTOR_MAX_RETRIES = 3

# Name of the prompt template (under src/concept_extraction/prompts/, without
# the ".txt" extension) used for concept extraction.
CONCEPT_EXTRACTION_PROMPT = "concept_extraction_v3"

# --- Output designation -------------------------------------------------------

# Name of the prompt template (under src/output_designation/prompts/, without
# the ".txt" extension) used for output designation.
OUTPUT_DESIGNATION_PROMPT = "output_designation"

# How many times to retry if the model returns malformed JSON.
OUTPUT_DESIGNATION_MAX_RETRIES = 2

# --- Entity linker ---------------------------------------------------------

# Names of the prompt templates (under src/entity_linker/prompts/, without the
# ".txt" extension) used for the two LLM calls in the entity linker.
LINKER_PROPOSAL_PROMPT = "linker_proposal_v3"
LINKER_REFORMULATION_PROMPT = "linker_reformulation_v3"

# BioGateway SPARQL endpoint. SSL verification is disabled because the server
# certificate does not validate against the default CA bundle.
BIOGATEWAY_SPARQL_ENDPOINT = "https://semantics.inf.um.es/biogateway"
SPARQL_VERIFY_SSL = False
SPARQL_TIMEOUT = 60  # seconds

# Named graphs live under this base; e.g. the "gene" hint -> .../graph/gene
BIOGATEWAY_GRAPH_BASE = "http://rdf.biogateway.eu/graph/"

# Max reformulation attempts per zero-result span before marking it unresolved.
LINKER_MAX_ITERATIONS = 3

# When True, a single SPARQL match triggers an extra LLM call to confirm it is
# the intended entity before binding.  False (default) binds automatically.
LLM_CONFIRM_SINGLE_MATCH = False

# --- SPARQL generator --------------------------------------------------------

# Name of the prompt template (under src/sparql_generator/prompts/, without the
# ".txt" extension) used to author the SPARQL query from the assembly brief.
SPARQL_GENERATOR_PROMPT = "sparql_generator"

# Max repair attempts after a parse/execution error, feeding (query, error)
# back to the same LLM. 0 disables repair (validate once, return as-is).
SPARQL_GENERATOR_MAX_REPAIR = 2

# --- Path resolver (equal-length candidate path selection) ------------------

# Name of the prompt template (under src/sparql_generator/prompts/, without the
# ".txt" extension) used for the LLM's 1-of-N tiebreak among candidate paths.
PATH_RESOLVER_TIEBREAK_PROMPT = "path_resolver_tiebreak"

# How many times to retry if the model returns malformed JSON.
PATH_RESOLVER_MAX_RETRIES = 2
