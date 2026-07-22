"""M4 training configs — declared, validated, and gated (never run here).

Training the generator (SFT seed -> GRPO on the comprehension reward) needs paid compute
(Tinker credits + a GPU), so it is out of scope for offline verification. These configs
are declared as data and validated for the things that must hold before any run — chiefly
that the reader pool excludes the generator's own model family (anti-gaming) and that the
reward is the comprehension signal. The offline reward + search (``glassbox.reward``)
prove the objective is optimizable without training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .reward import model_family, pool_excludes_family

_METHODS = {"sft", "grpo"}


class TrainingConfigError(ValueError):
    """A training config is malformed."""


@dataclass(frozen=True)
class TrainingConfig:
    id: str
    description: str
    framework: str
    base_model: str
    generator_family: str
    methods: List[str]
    reward: str
    reader_pool: List[str]
    status: str
    params: Dict[str, Any] = field(default_factory=dict)


def parse_training_config(raw: Dict[str, Any]) -> TrainingConfig:
    def req(key: str, typ):
        v = raw.get(key)
        if not isinstance(v, typ) or (isinstance(v, (str, list)) and not v):
            raise TrainingConfigError("training config needs a non-empty %r" % key)
        return v

    return TrainingConfig(
        id=req("id", str), description=req("description", str),
        framework=req("framework", str), base_model=req("base_model", str),
        generator_family=req("generator_family", str),
        methods=list(req("methods", list)), reward=req("reward", str),
        reader_pool=list(req("reader_pool", list)),
        status=raw.get("status", "not run"), params=dict(raw.get("params", {})))


def load_training_config(path: str) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError("%s: invalid JSON: %s" % (path, exc))
    return parse_training_config(raw)


def validate_training_config(config: TrainingConfig) -> List[str]:
    """Return the errors that would block a real run (empty means ready-to-run)."""
    errors: List[str] = []
    if config.reward != "comprehension_lift":
        errors.append("reward must be 'comprehension_lift' (the score is the reward); got %r"
                      % config.reward)
    for m in config.methods:
        if m not in _METHODS:
            errors.append("unknown training method %r (have: %s)" % (m, sorted(_METHODS)))
    if not config.reader_pool:
        errors.append("reader_pool is empty — need readers to score the reward")
    if not pool_excludes_family(config.reader_pool, config.generator_family):
        errors.append("reader pool must exclude the generator's family %r — a generator "
                      "must not be scored by a sibling model (anti-gaming)"
                      % config.generator_family)
    unresolved = [r for r in config.reader_pool if model_family(r) == "unknown"]
    if unresolved:
        errors.append("cannot resolve the model family of reader(s) %s — use a recognized "
                      "model slug so the anti-gaming exclusion can be verified (fail-closed)"
                      % unresolved)
    return errors
