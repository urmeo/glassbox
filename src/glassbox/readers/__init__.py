"""Reader construction and pure specification validation."""

from __future__ import annotations

from typing import List

from .._config import strings, text
from .base import Answer, Reader
from .simulated import SIMULATED_PERSONAS, SimulatedReader


def validate_reader_spec(spec: str, *, shorthand: bool = False) -> str:
    """Validate provider/model syntax without accessing credentials or the network."""
    spec = text(spec, "reader spec")
    if shorthand and spec == "simulated":
        return spec
    provider, separator, model = spec.partition(":")
    provider, model = provider.lower(), model.strip()
    if not separator or provider not in {
        "simulated",
        "anthropic",
        "openai",
        "openrouter",
    }:
        raise ValueError("unknown reader provider in spec %r" % spec)
    if not model or any(character.isspace() for character in model):
        raise ValueError(
            "reader spec needs a nonempty model/persona without whitespace"
        )
    if provider == "simulated" and model not in SIMULATED_PERSONAS:
        raise ValueError("unknown simulated persona %r" % model)
    return provider + ":" + model


def build_reader(spec: str) -> Reader:
    spec = validate_reader_spec(spec)
    provider, model = spec.split(":", 1)
    if provider == "simulated":
        return SimulatedReader(model)
    if provider == "anthropic":
        from .anthropic import AnthropicReader

        return AnthropicReader(model)
    from .openai_compat import OpenAICompatReader

    return OpenAICompatReader(provider, model)


def expand_reader_specs(specs: List[str]) -> List[str]:
    """Expand simulated shorthand; repetition is controlled through replicates."""
    specs = strings(specs, "readers")
    out = []
    for spec in specs:
        spec = validate_reader_spec(spec, shorthand=True)
        out.extend(
            ["simulated:" + persona for persona in SIMULATED_PERSONAS]
            if spec == "simulated"
            else [spec]
        )
    if len(set(out)) != len(out):
        raise ValueError("readers contain duplicate expanded specs")
    return out


def build_readers(specs: List[str]) -> List[Reader]:
    return [build_reader(spec) for spec in expand_reader_specs(specs)]


__all__ = [
    "Answer",
    "Reader",
    "SimulatedReader",
    "SIMULATED_PERSONAS",
    "build_reader",
    "build_readers",
    "expand_reader_specs",
    "validate_reader_spec",
]
