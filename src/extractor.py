import json
import logging
from typing import Any

import ollama
from pydantic import BaseModel, ValidationError

from src import config
from src.prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger(__name__)

# Pydantic models to ensure the extracted JSON has the expected structure and types
class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractionResult(BaseModel):
    entities: list[str]
    output: list[str]
    relationships: list[Relationship]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": self.entities,
            "output": self.output,
            "relationships": [r.model_dump() for r in self.relationships],
        }


class QueryExtractor:
    """Class to extract structured information from natural-language questions about BioGateway using an LLM.
    
    Attributes:
        model_name (str): The name of the Ollama model to use for extraction.
        max_retries (int): The maximum number of times to retry extraction if it fails.
    """

    def __init__(self, model_name: str = config.OLLAMA_MODEL, base_url: str = config.OLLAMA_BASE_URL, 
                 max_retries: int = config.EXTRACTOR_MAX_RETRIES):
        """Initialize the QueryExtractor with the specified model and Ollama client.
        
        Parameters:
            model_name (str): The name of the Ollama model to use for extraction.
            base_url (str): The base URL of the Ollama server.
            max_retries (int): The maximum number of times to retry extraction if it fails.
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self._client = ollama.Client(host=base_url)

    def extract(self, question: str) -> dict[str, Any]:
        """Extract entities, desired outputs, and relationships from a NL question.

        Parameters:
            question (str): A natural-language question about BioGateway.

        Returns:
            dict[str, Any]: A dictionary with keys: entities (list[str]), output (list[str]), relationships (list[dict])
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question)},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat(
                    model=self.model_name,
                    messages=messages,
                    format="json",
                    options={"temperature": 0},
                )
                raw = response.message.content
                parsed = json.loads(raw)
                result = ExtractionResult.model_validate(parsed)
                return result.to_dict()

            except (json.JSONDecodeError, ValidationError, KeyError) as exc:
                last_error = exc
                logger.warning("Attempt %d/%d failed: %s", attempt, self.max_retries, exc)

        raise ValueError(
            f"Extraction failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )
