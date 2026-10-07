"""Resolve known model-name families independently from API routing."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FAMILY_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "gpt": "openai",
    "qwen": "qwen",
    "llama": "llama",
    "gemini": "gemini",
    "mistral": "mistral",
    "mixtral": "mistral",
    "deepseek": "deepseek",
    "simulated": "simulated",
}
_PREFIXES = (
    ("anthropic", r"claude(?:$|[-_.\d])"),
    ("openai", r"(?:gpt(?:$|[-_.\d])|o[134](?:$|[-_.]))"),
    ("qwen", r"qwen(?:$|[-_.\d])"),
    ("llama", r"llama(?:$|[-_.\d])"),
    ("gemini", r"gemini(?:$|[-_.\d])"),
    ("mistral", r"(?:mistral|mixtral)(?:$|[-_.\d])"),
    ("deepseek", r"deepseek(?:$|[-_.\d])"),
)


@dataclass(frozen=True)
class FamilyIdentity:
    provider: str
    model: str
    family: str
    resolved: bool
    basis: str


def canonical_family(value: str) -> str:
    """Normalize explicit family labels; unresolved labels remain unknown."""
    return (
        _FAMILY_ALIASES.get(value.strip().lower(), "unknown")
        if isinstance(value, str)
        else "unknown"
    )


def resolve_family(reader_spec: str) -> FamilyIdentity:
    """Recognize model-name prefixes; this does not verify model availability."""
    if not isinstance(reader_spec, str) or not reader_spec.strip():
        return FamilyIdentity("unknown", "unknown", "unknown", False, "unknown")
    spec = reader_spec.strip()
    provider, separator, model = spec.partition(":")
    if not separator:
        provider, model = "unknown", spec
    provider, model = provider.lower(), model.strip()
    if provider == "simulated" and model in {"literal", "diligent", "careless"}:
        return FamilyIdentity(provider, model, "simulated", True, "simulated-fixture")
    leaf = model.lower().rsplit("/", 1)[-1]
    for family, pattern in _PREFIXES:
        if re.match(pattern, leaf):
            return FamilyIdentity(provider, model, family, True, "model-name")
    return FamilyIdentity(provider, model, "unknown", False, "unknown")


def model_family(reader_spec: str) -> str:
    """Return the resolved model family or unknown."""
    return resolve_family(reader_spec).family
