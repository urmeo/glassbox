"""Readers — the models (real or simulated) that answer questions from a stimulus.

A reader is named ``family:model`` (for example ``simulated:literal``,
``anthropic:claude-sonnet-5``, ``openrouter:qwen/qwen3-vl-8b-instruct``). Simulated
readers are deterministic fixtures that make the whole pipeline verifiable offline;
real readers call a vision API, read no keys from disk, and are only exercised when a
key is present in the environment.
"""

from __future__ import annotations

from typing import List

from .base import Answer, Reader
from .simulated import SIMULATED_PERSONAS, SimulatedReader


def _split(spec: str):
    family, _, model = spec.partition(":")
    return family, model


def build_reader(spec: str) -> Reader:
    """Construct a reader from a ``family:model`` spec string.

    ``simulated`` (with no persona) is rejected here — expand it with
    :func:`expand_reader_specs` first so the caller controls the persona set.
    """
    family, model = _split(spec)
    if family == "simulated":
        if not model:
            raise ValueError("use 'simulated:<persona>' (one of %s)"
                             % ", ".join(SIMULATED_PERSONAS))
        return SimulatedReader(model)
    if family == "anthropic":
        from .anthropic import AnthropicReader
        return AnthropicReader(model)
    if family in ("openrouter", "openai"):
        from .openai_compat import OpenAICompatReader
        return OpenAICompatReader(family, model)
    raise ValueError("unknown reader family %r in spec %r" % (family, spec))


def expand_reader_specs(specs: List[str]) -> List[str]:
    """Expand shorthand specs. Bare ``simulated`` becomes every simulated persona."""
    out: List[str] = []
    for spec in specs:
        if spec == "simulated":
            out.extend("simulated:" + p for p in SIMULATED_PERSONAS)
        else:
            out.append(spec)
    # de-duplicate, preserving order
    seen = set()
    unique = []
    for s in out:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def build_readers(specs: List[str]) -> List[Reader]:
    return [build_reader(s) for s in expand_reader_specs(specs)]


__all__ = [
    "Answer", "Reader", "SimulatedReader", "SIMULATED_PERSONAS",
    "build_reader", "build_readers", "expand_reader_specs",
]
