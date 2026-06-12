"""Loads prompt templates from prompts/ and fills them with string.Template.

Templates use $VAR placeholders (not str.format()/f-strings), since prompts
are full of literal "{" and "}" from JSON examples that .format() would choke on.
"""
from __future__ import annotations

from pathlib import Path
from string import Template

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def render_prompt(name: str, **values: str) -> str:
    """Read prompts/<name>.txt and substitute $-prefixed placeholders.

    Parameters:
        name (str): The template's filename without the ".txt" extension.
        **values (str): Values to substitute for each $VAR placeholder in the template.

    Returns:
        str: The rendered prompt text.
    """
    template_path = _PROMPTS_DIR / f"{name}.txt"
    template = Template(template_path.read_text(encoding="utf-8"))
    return template.substitute(**values)
