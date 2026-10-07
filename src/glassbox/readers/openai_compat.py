"""Lazy openai_compat reader payloads and reply extraction."""

from __future__ import annotations

import os
import hashlib
from typing import Any, Dict

from .._config import positive_integer, text as config_text
from ..families import resolve_family
from ..schema import Question, Scenario
from ..stimuli import Stimulus
from ._http import encode_png_base64, post_json, require_key
from .base import Answer, Reader, build_mcq_prompt, parse_choice

_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class OpenAICompatReader(Reader):
    simulated = False
    deterministic = False

    def __init__(self, family: str, model: str, max_tokens: int = 512):
        if not isinstance(family, str) or family not in _ENDPOINTS:
            raise ValueError(
                "OpenAI-compatible family must be one of %s" % ", ".join(_ENDPOINTS)
            )
        model = config_text(model, "model")
        positive_integer(max_tokens, "max_tokens")
        if any(c.isspace() for c in model):
            raise ValueError("model id must not contain whitespace")
        self.family = family
        self.provider = family
        self.model = model
        self.name = family + ":" + model
        self.max_tokens = max_tokens
        identity = resolve_family(self.name)
        self.model_family = identity.family
        self.family_resolved = identity.resolved
        self.family_basis = identity.basis

    def build_payload(self, question: Question, stimulus: Stimulus) -> Dict[str, Any]:
        content = [
            {
                "type": "text",
                "text": build_mcq_prompt(question, stimulus, image=stimulus.has_image),
            }
        ]
        if stimulus.has_image:
            uri = "data:%s;base64,%s" % (
                stimulus.media_type,
                encode_png_base64(stimulus.image_path),
            )
            content.append({"type": "image_url", "image_url": {"url": uri}})
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
        }

    def _headers(self, key: str) -> Dict[str, str]:
        headers = {"authorization": "Bearer " + key}
        if self.family == "openrouter":
            headers["X-Title"] = "Glass Box"
            app_url = os.environ.get("GLASSBOX_APP_URL")
            if app_url:  # optional attribution only; omitted unless set
                headers["HTTP-Referer"] = app_url
        return headers

    def answer(
        self, scenario: Scenario, question: Question, stimulus: Stimulus
    ) -> Answer:
        payload = self.build_payload(question, stimulus)
        prompt = next(
            block["text"]
            for block in payload["messages"][0]["content"]
            if block["type"] == "text"
        )
        key = require_key(_KEY_ENV[self.family])
        resp = post_json(_ENDPOINTS[self.family], self._headers(key), payload)
        text = extract_text(resp)
        return Answer(
            choice_id=parse_choice(text, question),
            method="api",
            raw=text,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )


def extract_text(resp: Dict[str, Any]) -> str:
    try:
        text = resp["choices"][0]["message"]["content"]
        return text if isinstance(text, str) else ""
    except (KeyError, IndexError, TypeError):
        return ""
