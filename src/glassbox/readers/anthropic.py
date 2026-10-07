"""Lazy anthropic reader payloads and reply extraction."""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from .._config import positive_integer, text as config_text
from ..families import resolve_family
from ..schema import Question, Scenario
from ..stimuli import Stimulus
from ._http import encode_png_base64, post_json, require_key
from .base import Answer, Reader, build_mcq_prompt, parse_choice

ENDPOINT = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicReader(Reader):
    simulated = False
    deterministic = False

    def __init__(self, model: str, max_tokens: int = 512):
        model = config_text(model, "anthropic model")
        positive_integer(max_tokens, "max_tokens")
        if any(c.isspace() for c in model):
            raise ValueError("model id must not contain whitespace")
        self.model = model
        self.family = "anthropic"
        self.provider = "anthropic"
        self.name = "anthropic:" + model
        self.max_tokens = max_tokens
        identity = resolve_family(self.name)
        self.model_family = identity.family
        self.family_resolved = identity.resolved
        self.family_basis = identity.basis

    def build_payload(self, question: Question, stimulus: Stimulus) -> Dict[str, Any]:
        content = []
        if stimulus.has_image:  # Preserve the image-first wire format.
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": stimulus.media_type,
                        "data": encode_png_base64(stimulus.image_path),
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": build_mcq_prompt(question, stimulus, image=stimulus.has_image),
            }
        )
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }

    def answer(
        self, scenario: Scenario, question: Question, stimulus: Stimulus
    ) -> Answer:
        payload = self.build_payload(question, stimulus)
        prompt = next(
            block["text"]
            for block in payload["messages"][0]["content"]
            if block["type"] == "text"
        )
        key = require_key("ANTHROPIC_API_KEY")
        resp = post_json(
            ENDPOINT, {"x-api-key": key, "anthropic-version": API_VERSION}, payload
        )
        text = extract_text(resp)
        return Answer(
            choice_id=parse_choice(text, question),
            method="api",
            raw=text,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )


def extract_text(resp: Dict[str, Any]) -> str:
    """The first text block of a Messages response."""
    if not isinstance(resp, dict) or not isinstance(resp.get("content", []), list):
        return ""
    for block in resp.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            return text if isinstance(text, str) else ""
    return ""
