"""Load packaged prompt text without normalizing authored escape sequences."""

from __future__ import annotations

import os
from typing import Optional

from .resources import resource_text


def prompts_dir(base: Optional[str] = None) -> str:
    if base is not None:
        return os.path.join(base, "data", "prompts")
    return os.path.join(os.path.dirname(__file__), "_data", "prompts")


def load_prompt(name: str) -> str:
    return resource_text("prompts", name)
