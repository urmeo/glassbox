"""The stimulus — what a reader actually consumes.

A stimulus wraps a :class:`~glassbox.interfaces.Presentation` in one of two forms:

* ``image``  — a rendered PNG (what a real vision-language reader sees), plus a text
  fallback.
* ``spec``   — no image; the reader works from the presentation's structured values
  (simulated readers) or its text serialization (real readers in ``--skip-render``).

Keeping both forms behind one type means the readers don't care how the interface
was produced — only what it presents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .interfaces import Presentation


@dataclass(frozen=True)
class Stimulus:
    presentation: Presentation
    kind: str                      # "image" or "spec"
    text: str                      # text serialization (always present)
    image_path: Optional[str] = None
    media_type: str = "image/png"

    @property
    def has_image(self) -> bool:
        return self.kind == "image" and self.image_path is not None
