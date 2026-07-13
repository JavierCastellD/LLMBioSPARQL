"""Shared LLM utilities: JSON extraction, retry/repair loop, Ollama client factory."""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import ollama

from ..config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_THINK

LLMCallable = Callable[[str], str]


def extract_json(text: str) -> Any | None:
    """Parse JSON from raw LLM output, tolerating markdown fences and surrounding prose.

    Parameters:
        text (str):
            Raw LLM completion to parse.

    Returns:
        The parsed JSON value, or None if no valid JSON object could be recovered.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: extract the outermost {...} block.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _build_repair_prompt(original_prompt: str, previous_response: str, errors: list[str]) -> str:
    """Append the previous invalid response and a list of problems to the original prompt.

    Parameters:
        original_prompt (str):
            The prompt that produced the invalid response.
        previous_response (str):
            The raw LLM output that failed validation.
        errors (list[str]):
            Human-readable descriptions of the validation failures.

    Returns:
        A new prompt asking the LLM to correct the identified issues.
    """
    issues = "\n".join(f"- {e}" for e in errors)
    return (
        f"{original_prompt}\n\n"
        f"Your previous answer was:\n{previous_response}\n\n"
        f"That answer is INVALID for these reasons:\n{issues}\n\n"
        f"Output ONLY a corrected JSON object that fixes these issues."
    )


def llm_call_with_retry(
    llm: LLMCallable,
    prompt: str,
    validate_fn: Callable[[Any], list[str]],
    max_retries: int = 3,
) -> Any | None:
    """Call the LLM, validate the response, and repair up to max_retries times.

    Validates up to max_retries times, making at most max_retries-1 repair calls.

    Parameters:
        llm (LLMCallable):
            Callable that takes a prompt string and returns the model's response.
        prompt (str):
            The initial prompt to send.
        validate_fn (Callable[[Any], list[str]]):
            Function that receives the parsed JSON and returns a list of error strings,
            or an empty list if the response is valid.
        max_retries (int):
            Maximum number of validation attempts. Defaults to 3.

    Returns:
        The parsed JSON on success, or None if all attempts fail.
    """
    raw = llm(prompt)
    for attempt in range(max_retries):
        parsed = extract_json(raw)
        errors = validate_fn(parsed)
        if not errors:
            return parsed
        # Only repair if there are more attempts left.
        if attempt < max_retries - 1:
            raw = llm(_build_repair_prompt(prompt, raw, errors))
    return None


class OllamaLLM:
    """Ollama-backed LLM callable that satisfies LLMCallable via __call__.

    Attributes:
        model (str):
            Name of the Ollama model to use.
        base_url (str):
            URL of the Ollama server.
        temperature (float):
            Sampling temperature. 0.0 gives deterministic output.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.0,
    ) -> None:
        """Initializes the attributes of the class.

        Parameters:
            model (str):
                Name of the Ollama model to use.
            base_url (str):
                URL of the Ollama server.
            temperature (float):
                Sampling temperature. 0.0 gives deterministic output.
        """
        self._client = ollama.Client(host=base_url)
        self._model = model
        self._temperature = temperature

    def __call__(self, prompt: str) -> str:
        """Send a prompt to the Ollama model and return the generated text.

        Parameters:
            prompt (str):
                The input prompt to send to the model.

        Returns:
            The model's response as a plain string.
        """
        resp = self._client.generate(
            model=self._model,
            prompt=prompt,
            think=OLLAMA_THINK,
            options={"temperature": self._temperature},
        )
        return resp["response"]
