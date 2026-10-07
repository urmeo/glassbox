"""Training declarations and family-exclusion preflight. Training is not run here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ._config import (
    config_input_path,
    finite_json,
    load_config,
    object_fields,
    strings,
    text,
)
from .families import canonical_family, model_family
from .readers import expand_reader_specs

_METHODS = {"sft", "grpo"}


class TrainingConfigError(ValueError):
    """A training configuration is malformed."""


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

    def __post_init__(self) -> None:
        for name in (
            "id",
            "description",
            "framework",
            "base_model",
            "generator_family",
            "reward",
            "status",
        ):
            object.__setattr__(
                self, name, text(getattr(self, name), name, TrainingConfigError)
            )
        for name in ("methods", "reader_pool"):
            object.__setattr__(
                self, name, strings(getattr(self, name), name, TrainingConfigError)
            )
        try:
            expand_reader_specs(self.reader_pool)
        except ValueError as exc:
            raise TrainingConfigError(str(exc)) from None
        if not isinstance(self.params, dict):
            raise TrainingConfigError("params must be an object")
        finite_json(self.params, TrainingConfigError)


def parse_training_config(raw: Dict[str, Any]) -> TrainingConfig:
    object_fields(raw, TrainingConfig.__dataclass_fields__, TrainingConfigError)
    values = dict(raw)
    values.setdefault("status", "unrun")
    try:
        return TrainingConfig(**values)
    except TypeError as exc:
        raise TrainingConfigError(str(exc)) from None


def load_training_config(source: str) -> TrainingConfig:
    config = parse_training_config(load_config("training", source, TrainingConfigError))
    object.__setattr__(config, "_source_path", config_input_path("training", source))
    return config


def validate_training_config(config: TrainingConfig) -> List[str]:
    """Return configuration errors; an empty list does not verify model availability."""
    if not isinstance(config, TrainingConfig):
        raise TrainingConfigError("config must be a TrainingConfig")
    config.__post_init__()
    errors: List[str] = []
    if config.reward != "comprehension_lift":
        errors.append("reward must be 'comprehension_lift'; got %r" % config.reward)
    for method in config.methods:
        if method not in _METHODS:
            errors.append("unknown training method %r" % method)
    family = canonical_family(config.generator_family)
    base_family = model_family(config.base_model)
    if family == "unknown":
        errors.append(
            "cannot resolve the generator's family %r" % config.generator_family
        )
    if base_family == "unknown":
        errors.append("cannot resolve the base model's family %r" % config.base_model)
    elif family != "unknown" and base_family != family:
        errors.append(
            "base model family %r differs from declared generator family %r"
            % (base_family, family)
        )
    reader_specs = expand_reader_specs(config.reader_pool)
    unresolved = [spec for spec in reader_specs if model_family(spec) == "unknown"]
    if unresolved:
        errors.append("cannot resolve the model family of reader(s) %s" % unresolved)
    siblings = [spec for spec in reader_specs if model_family(spec) == family]
    if siblings and family != "unknown":
        errors.append(
            "reader pool must exclude the generator's family %r: %s"
            % (family, siblings)
        )
    return errors
