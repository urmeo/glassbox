"""Prompt templates live in ``data/prompts/`` — content, not code."""

from __future__ import annotations

import os
from typing import Optional


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prompts_dir(base: Optional[str] = None) -> str:
    return os.path.join(base or _repo_root(), "data", "prompts")


def load_prompt(name: str) -> str:
    """Load a prompt template by name (``.txt`` optional)."""
    filename = name if name.endswith(".txt") else name + ".txt"
    with open(os.path.join(prompts_dir(), filename), "r", encoding="utf-8") as fh:
        return fh.read()
