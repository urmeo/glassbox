"""Anthropic Messages API vision reader.

Image block carries **raw** base64 (no ``data:`` prefix — that is an OpenAI-ism), the
image precedes the text, ``max_tokens`` is required, and the reply is the first text
block at ``content[0].text``. The key is read from ``ANTHROPIC_API_KEY`` at call time.
"""

from __future__ import annotations

from typing import Any, Dict

from ..schema import Question, Scenario
from ..stimuli import Stimulus
from ._http import encode_png_base64, post_json, require_key
from .base import Answer, Reader, build_mcq_prompt, parse_choice

ENDPOINT = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicReader(Reader):
    simulated = False

    def __init__(self, model: str, max_tokens: int = 512):
        if not model:
            raise ValueError("anthropic reader needs a model id, e.g. anthropic:claude-sonnet-5")
        self.model = model
        self.family = "anthropic"
        self.name = "anthropic:" + model
        self.max_tokens = max_tokens

    def build_payload(self, question: Question, stimulus: Stimulus) -> Dict[str, Any]:
        content = []
        if stimulus.has_image:  # image first — Claude reads images best before text
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": stimulus.media_type,
                    "data": encode_png_base64(stimulus.image_path),
                },
            })
        content.append({"type": "text",
                        "text": build_mcq_prompt(question, stimulus, image=stimulus.has_image)})
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }

    def answer(self, scenario: Scenario, question: Question, stimulus: Stimulus) -> Answer:
        key = require_key("ANTHROPIC_API_KEY")
        resp = post_json(ENDPOINT,
                         {"x-api-key": key, "anthropic-version": API_VERSION},
                         self.build_payload(question, stimulus))
        text = extract_text(resp)
        return Answer(choice_id=parse_choice(text, question), method="api", raw=text)


def extract_text(resp: Dict[str, Any]) -> str:
    """The first text block of a Messages response."""
    for block in resp.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "") or ""
    return ""
