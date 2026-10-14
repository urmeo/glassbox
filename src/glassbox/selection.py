"""Freeze searched features and check that evaluation scenarios are disjoint."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

from . import __version__
from ._config import finite_json, strict_json
from .answer_parsing import PARSER_VERSION
from .output_paths import validate_output_files
from .prompts import load_prompt
from .readers.base import Reader
from .reward import EvalResult, GENERATOR_FEATURE_POOL, _inputs
from .schema import Scenario, parse_scenario, scenario_payload, scenario_sha256
from .validate import validate_scenario

FORMAT = "glassbox.selection.v1"
_FIELDS = {
    "format", "glassbox_version", "features", "selection_reward", "candidate_count",
    "selection_protocol", "baseline_sampling", "skip_render", "readers", "scenarios",
    "reader_prompt_template_sha256", "answer_parser_version", "sha256",
}
_READER_FIELDS = {
    "name", "provider", "model", "model_family", "family_resolved", "family_basis",
    "simulated", "deterministic",
}
_SNAPSHOT_FIELDS = {
    "id", "title", "domain", "description", "data", "questions", "resolved_presentation",
}


class SelectionError(ValueError):
    """A frozen selection or its evaluation split is invalid."""


@dataclass(frozen=True)
class SelectionArtifact:
    """Canonical JSON keeps nested selection metadata immutable."""

    payload_json: str
    source_path: Optional[str] = None

    @property
    def features(self) -> FrozenSet[str]:
        return frozenset(selection_payload(self)["features"])

    @property
    def scenario_ids(self) -> Tuple[str, ...]:
        return tuple(item["id"] for item in selection_payload(self)["scenarios"])

    @property
    def reader_names(self) -> Tuple[str, ...]:
        return tuple(item["name"] for item in selection_payload(self)["readers"])

    @property
    def skip_render(self) -> bool:
        return selection_payload(self)["skip_render"]

    @property
    def selection_reward(self) -> float:
        return selection_payload(self)["selection_reward"]

    @property
    def sha256(self) -> str:
        return selection_payload(self)["sha256"]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _fields(raw: Any, expected: set, label: str) -> None:
    if not isinstance(raw, dict) or set(raw) != expected:
        raise SelectionError("%s has missing or unknown fields" % label)


def _hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SelectionError("%s must be a SHA-256 digest" % label)


def scenario_content_sha256(scenario: Scenario) -> str:
    """Ignore scenario identity and descriptive text when checking reused tasks."""
    return _digest({"data": scenario.data, "questions": [asdict(q) for q in scenario.questions]})


def _source_sha256(scenario: Scenario) -> str:
    items = sorted(scenario.items, key=lambda item: item["id"])
    return _digest(_normalize_numbers({"items": items, "derived": scenario.derived}))


def _normalize_numbers(value: Any) -> Any:
    """Treat integral floats and integers as the same source values."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    return value


def _reader_metadata(reader: Reader) -> Dict[str, Any]:
    return {
        "name": reader.name,
        "provider": reader.provider,
        "model": getattr(reader, "model", ""),
        "model_family": reader.model_family,
        "family_resolved": reader.family_resolved,
        "family_basis": reader.family_basis,
        "simulated": reader.simulated,
        "deterministic": reader.deterministic,
    }


def freeze_selection(evaluation: EvalResult, scenarios: Sequence[Scenario],
                     readers: Sequence[Reader], skip_render: bool) -> SelectionArtifact:
    _inputs(scenarios, readers, skip_render)
    if not isinstance(evaluation, EvalResult):
        raise SelectionError("selection requires a feature-search result")
    if evaluation.scenario_count != len(scenarios) or evaluation.reader_count != len(readers):
        raise SelectionError("selection inputs do not match the feature-search result")
    raw = {
        "format": FORMAT,
        "glassbox_version": __version__,
        "features": sorted(evaluation.generator_features),
        "selection_reward": evaluation.generator_reward,
        "candidate_count": evaluation.candidate_count,
        "selection_protocol": evaluation.selection_protocol,
        "baseline_sampling": evaluation.baseline_sampling,
        "skip_render": skip_render,
        "readers": [_reader_metadata(reader) for reader in readers],
        "scenarios": [
            {"id": scenario.id, "sha256": scenario_sha256(scenario),
             "content_sha256": scenario_content_sha256(scenario),
             "source_sha256": _source_sha256(scenario),
             "snapshot": scenario_payload(scenario)}
            for scenario in scenarios
        ],
        "reader_prompt_template_sha256": hashlib.sha256(
            load_prompt("reader_mcq").encode("utf-8")).hexdigest(),
        "answer_parser_version": PARSER_VERSION,
    }
    raw["sha256"] = _digest(raw)
    return parse_selection(raw)


def parse_selection(raw: Any) -> SelectionArtifact:
    raw = finite_json(raw, SelectionError)
    _fields(raw, _FIELDS, "selection")
    if raw["format"] != FORMAT:
        raise SelectionError("unsupported selection artifact format")
    for key in ("glassbox_version", "answer_parser_version"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise SelectionError("%s must be nonempty text" % key)
    features = raw["features"]
    if (not isinstance(features, list)
            or any(not isinstance(name, str) or name not in GENERATOR_FEATURE_POOL for name in features)
            or len(set(features)) != len(features)):
        raise SelectionError("selected features must be distinct supported features")
    count = raw["candidate_count"]
    if (type(count) is not int or not 1 <= count <= 2**len(GENERATOR_FEATURE_POOL)
            or count & (count - 1) or len(features) > count.bit_length() - 1):
        raise SelectionError("candidate_count must match an exhaustive feature search")
    value = raw["selection_reward"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1 <= value <= 1:
        raise SelectionError("selection_reward must be finite and between -1 and 1")
    if raw["selection_protocol"] != "in_sample_exhaustive_feature_search.v1":
        raise SelectionError("unsupported selection protocol")
    if raw["baseline_sampling"] != "fresh_per_candidate_and_cards":
        raise SelectionError("unsupported selection baseline protocol")
    if type(raw["skip_render"]) is not bool:
        raise SelectionError("skip_render must be boolean")
    readers = raw["readers"]
    if not isinstance(readers, list) or not readers:
        raise SelectionError("selection requires source readers")
    names = set()
    for reader in readers:
        _fields(reader, _READER_FIELDS, "source reader")
        for key in ("name", "provider", "model_family", "family_basis"):
            if not isinstance(reader[key], str) or not reader[key].strip():
                raise SelectionError("source reader %s must be nonempty text" % key)
        if not isinstance(reader["model"], str) or reader["name"] in names:
            raise SelectionError("source reader models must be text and names distinct")
        if any(type(reader[key]) is not bool for key in ("simulated", "family_resolved")):
            raise SelectionError("source reader flags must be boolean")
        if reader["deterministic"] is not None and type(reader["deterministic"]) is not bool:
            raise SelectionError("source reader determinism must be boolean or null")
        names.add(reader["name"])
    scenarios = raw["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise SelectionError("selection requires source scenarios")
    ids, contents = set(), set()
    for item in scenarios:
        _fields(item, {"id", "sha256", "content_sha256", "source_sha256", "snapshot"},
                "selection scenario")
        _fields(item["snapshot"], _SNAPSHOT_FIELDS, "scenario snapshot")
        authored = {key: value for key, value in item["snapshot"].items()
                    if key != "resolved_presentation"}
        scenario = parse_scenario(authored)
        errors = validate_scenario(scenario)
        if errors:
            raise SelectionError("invalid selection scenario: %s" % "; ".join(errors))
        if (item["id"] != scenario.id or scenario.id in ids
                or scenario_payload(scenario) != item["snapshot"]
                or item["sha256"] != scenario_sha256(scenario)
                or item["content_sha256"] != scenario_content_sha256(scenario)
                or item["source_sha256"] != _source_sha256(scenario)
                or item["content_sha256"] in contents):
            raise SelectionError("selection scenario identity, snapshot or digest is invalid")
        ids.add(scenario.id)
        contents.add(item["content_sha256"])
    _hash(raw["reader_prompt_template_sha256"], "reader prompt hash")
    _hash(raw["sha256"], "selection hash")
    unsigned = {key: value for key, value in raw.items() if key != "sha256"}
    if raw["sha256"] != _digest(unsigned):
        raise SelectionError("selection artifact digest does not match its content")
    return SelectionArtifact(_canonical(raw))


def selection_payload(artifact: SelectionArtifact) -> Dict[str, Any]:
    if not isinstance(artifact, SelectionArtifact) or not isinstance(artifact.payload_json, str):
        raise SelectionError("expected a SelectionArtifact")
    return strict_json(artifact.payload_json)


def validate_selection(artifact: SelectionArtifact) -> SelectionArtifact:
    validated = parse_selection(selection_payload(artifact))
    return SelectionArtifact(validated.payload_json, artifact.source_path)


def validate_evaluation(artifact: SelectionArtifact, scenarios: Sequence[Scenario]) -> None:
    artifact = validate_selection(artifact)
    if (not isinstance(scenarios, Sequence) or not scenarios
            or any(not isinstance(scenario, Scenario) for scenario in scenarios)):
        raise SelectionError("evaluation requires explicit scenarios")
    raw = selection_payload(artifact)
    selected_ids = set(artifact.scenario_ids)
    selected_content = {item["content_sha256"] for item in raw["scenarios"]}
    selected_sources = {item["source_sha256"] for item in raw["scenarios"]}
    seen, seen_ids, seen_sources = set(), set(), set()
    for scenario in scenarios:
        errors = validate_scenario(scenario)
        if errors:
            raise SelectionError("invalid evaluation scenario: %s" % "; ".join(errors))
        content = scenario_content_sha256(scenario)
        source = _source_sha256(scenario)
        if (scenario.id in selected_ids or content in selected_content
                or source in selected_sources):
            raise SelectionError("evaluation overlaps selection scenario %r" % scenario.id)
        if content in seen or source in seen_sources:
            raise SelectionError("evaluation scenarios contain duplicate task content")
        if scenario.id in seen_ids:
            raise SelectionError("evaluation scenario IDs must be distinct")
        seen.add(content)
        seen_ids.add(scenario.id)
        seen_sources.add(source)


def load_selection(path: str) -> SelectionArtifact:
    try:
        source = Path(path).resolve()
        artifact = parse_selection(strict_json(source.read_text(encoding="utf-8")))
        return SelectionArtifact(artifact.payload_json, str(source))
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise SelectionError("cannot load selection: %s" % error) from None


def write_selection(path: str, artifact: SelectionArtifact) -> str:
    artifact = validate_selection(artifact)
    destination = Path(path)
    (destination,) = validate_output_files(destination.parent, (destination.name,))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(selection_payload(artifact), indent=2, ensure_ascii=False) + "\n")
    return str(destination)
