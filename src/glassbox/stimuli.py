"""Wrap a visible text specification or rendered image with its provenance."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .interfaces import Presentation


@dataclass(frozen=True)
class Stimulus:
    presentation: Presentation
    kind: str
    text: str
    image_path: Optional[str] = None
    media_type: str = "image/png"
    image_sha256: Optional[str] = None
    render_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("image", "spec") or not isinstance(self.text, str):
            raise ValueError("stimulus requires an image/spec kind and text")
        if not isinstance(self.render_metadata, dict):
            raise ValueError("render_metadata must be a dict")
        object.__setattr__(self, "render_metadata", copy.deepcopy(self.render_metadata))

    @property
    def has_image(self) -> bool:
        return self.kind == "image" and self.image_path is not None

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()
