"""Loads prompt templates from a given directory and fills in $-placeholders.

Templates use $VAR syntax (not str.format()/f-strings), since prompts often
contain literal "{" and "}" from JSON examples that .format() would choke on.
"""
from __future__ import annotations

from pathlib import Path
from string import Template


def render_prompt(name: str, prompts_dir: Path, **values: str) -> str:
    """Read <prompts_dir>/<name>.txt and substitute $-prefixed placeholders.

    Parameters:
        name (str): Template filename without the ".txt" extension.
        prompts_dir (Path): Directory that contains the template file.
        **values (str): Values to substitute for each $VAR placeholder.

    Returns:
        str: The rendered prompt text.
    """
    template = Template((prompts_dir / f"{name}.txt").read_text(encoding="utf-8"))
    return template.substitute(**values)
