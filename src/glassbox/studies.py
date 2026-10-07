"""Validated study definitions and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from . import analysis, report, scoring
from ._config import (
    config_input_path,
    boolean,
    load_config,
    object_fields,
    positive_integer,
    strings,
    text,
)
from .interfaces import INTERFACE_VARIANTS, REFERENCE_VARIANTS
from .judge import build_judge, validate_judge_spec
from .output_paths import validate_output_files
from .readers import build_readers, expand_reader_specs
from .schema import Scenario, load_all_scenarios


class StudyError(ValueError):
    """A study configuration is malformed."""


@dataclass(frozen=True)
class StudyConfig:
    id: str
    description: str
    readers: List[str]
    scenarios: Optional[List[str]] = None
    interfaces: Optional[List[str]] = None
    replicates: int = 1
    judge: str = "polish"
    skip_render: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", text(self.id, "id", StudyError))
        text(self.description, "description", StudyError, empty=True)
        reader_specs = strings(self.readers, "readers", StudyError)
        try:
            expand_reader_specs(reader_specs)
            judge = validate_judge_spec(self.judge)
        except ValueError as exc:
            raise StudyError(str(exc)) from None
        object.__setattr__(self, "readers", reader_specs)
        object.__setattr__(self, "judge", judge)
        for field in ("scenarios", "interfaces"):
            values = strings(getattr(self, field), field, StudyError, optional=True)
            object.__setattr__(self, field, values)
        if self.interfaces is not None:
            for variant in self.interfaces:
                if variant not in INTERFACE_VARIANTS:
                    raise StudyError("unknown testable interface %r" % variant)
        positive_integer(self.replicates, "replicates", StudyError)
        boolean(self.skip_render, "skip_render", StudyError)

    def to_meta(self) -> Dict[str, Any]:
        return asdict(self)


def parse_study(raw: Dict[str, Any]) -> StudyConfig:
    object_fields(raw, StudyConfig.__dataclass_fields__, StudyError)
    try:
        return StudyConfig(**raw)
    except TypeError as exc:
        raise StudyError(str(exc)) from None


def load_study(source: str) -> StudyConfig:
    config = parse_study(load_config("studies", source, StudyError))
    object.__setattr__(config, "_source_path", config_input_path("studies", source))
    return config


@dataclass(frozen=True)
class StudyResult:
    config: StudyConfig
    book: "scoring.ScoreBook"
    h1: "analysis.H1Analysis"
    cross: "analysis.CrossFamilyAnalysis"
    paths: Dict[str, str]


def _resolve_scenarios(config: StudyConfig) -> List[Scenario]:
    everything = load_all_scenarios()
    if config.scenarios is None:
        return list(everything.values())
    chosen = []
    for sid in config.scenarios:
        if sid not in everything:
            raise StudyError(
                "unknown scenario %r (have: %s)" % (sid, ", ".join(everything))
            )
        chosen.append(everything[sid])
    return chosen


def run_study(config: StudyConfig, out_dir: str) -> StudyResult:
    """Validate, execute, and report the declared study."""
    if not isinstance(config, StudyConfig):
        raise StudyError("config must be a StudyConfig")
    config.__post_init__()
    protected = (
        (config._source_path,)
        if getattr(config, "_source_path", None) is not None
        else ()
    )
    validate_output_files(
        out_dir, ("report.md", "results.json"), protected_paths=protected
    )
    scenarios = _resolve_scenarios(config)
    interfaces = (
        list(INTERFACE_VARIANTS) if config.interfaces is None else config.interfaces
    )
    judge = build_judge(config.judge, variants=interfaces)
    readers = build_readers(config.readers)
    variants = list(REFERENCE_VARIANTS) + interfaces
    results = scoring.read_all(
        scenarios,
        readers,
        variants=variants,
        skip_render=config.skip_render,
        out_dir=out_dir,
        replicates=config.replicates,
    )
    book = scoring.ScoreBook(results)
    h1 = analysis.analyze_h1(book, scenarios, judge)
    cross = analysis.analyze_cross_family(book, scenarios, judge)
    paths = report.write_run(
        out_dir,
        scenarios,
        results,
        book,
        h1,
        cross,
        config.skip_render,
        study=config.to_meta(),
        protected_paths=protected,
    )
    return StudyResult(config=config, book=book, h1=h1, cross=cross, paths=paths)
