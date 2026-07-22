"""Studies — a reproducible, pre-registerable H1 run defined by a config file.

A study fixes everything that determines a result — scenarios, readers, interfaces
under test, replicates, and judge — in one JSON file, so the exact run can be declared
up front and re-run by anyone. ``run_study`` executes it end to end (read -> score ->
H1 -> cross-family -> report) and echoes the config into the results for provenance.
The CLI's quick ``run`` is the same path with an ad-hoc config.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from . import analysis, report, scoring
from .interfaces import INTERFACE_VARIANTS, REFERENCE_VARIANTS
from .judge import build_judge
from .readers import build_readers
from .schema import Scenario, load_all_scenarios


class StudyError(ValueError):
    """A study config is malformed."""


@dataclass(frozen=True)
class StudyConfig:
    id: str
    description: str
    readers: List[str]
    scenarios: Optional[List[str]] = None    # None means all shipped scenarios
    interfaces: Optional[List[str]] = None   # interface variants under test; None means all
    replicates: int = 1
    judge: str = "polish"
    skip_render: bool = True

    def to_meta(self) -> Dict[str, Any]:
        return asdict(self)


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise StudyError(message)


def parse_study(raw: Dict[str, Any]) -> StudyConfig:
    _require(isinstance(raw.get("id"), str) and raw["id"], "study needs a string id")
    _require(isinstance(raw.get("description"), str), "study needs a description")
    readers = raw.get("readers")
    _require(isinstance(readers, list) and readers, "study needs a non-empty 'readers' list")

    scenarios = raw.get("scenarios")
    _require(scenarios is None or isinstance(scenarios, list), "'scenarios' must be a list or absent")
    interfaces = raw.get("interfaces")
    _require(interfaces is None or isinstance(interfaces, list), "'interfaces' must be a list or absent")
    if interfaces is not None:
        for v in interfaces:
            _require(v in INTERFACE_VARIANTS,
                     "interface %r is not a testable variant %s" % (v, INTERFACE_VARIANTS))

    replicates = raw.get("replicates", 1)
    _require(isinstance(replicates, int) and replicates >= 1, "'replicates' must be an int >= 1")

    return StudyConfig(
        id=raw["id"], description=raw["description"], readers=list(readers),
        scenarios=list(scenarios) if scenarios is not None else None,
        interfaces=list(interfaces) if interfaces is not None else None,
        replicates=replicates, judge=raw.get("judge", "polish"),
        skip_render=raw.get("skip_render", True))


def load_study(path: str) -> StudyConfig:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise StudyError("%s: invalid JSON: %s" % (path, exc))
    return parse_study(raw)


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
            raise StudyError("unknown scenario %r (have: %s)" % (sid, ", ".join(everything)))
        chosen.append(everything[sid])
    return chosen


def run_study(config: StudyConfig, out_dir: str) -> StudyResult:
    """Execute a study end to end and write its report."""
    scenarios = _resolve_scenarios(config)
    readers = build_readers(config.readers)
    judge = build_judge(config.judge)
    interfaces = config.interfaces or list(INTERFACE_VARIANTS)
    variants = list(REFERENCE_VARIANTS) + interfaces

    results = scoring.read_all(scenarios, readers, variants=variants,
                               skip_render=config.skip_render, out_dir=out_dir,
                               replicates=config.replicates)
    book = scoring.ScoreBook(results)
    h1 = analysis.analyze_h1(book, scenarios, judge)
    cross = analysis.analyze_cross_family(book, scenarios, judge)
    paths = report.write_run(out_dir, scenarios, results, book, h1, cross,
                             config.skip_render, study=config.to_meta())
    return StudyResult(config=config, book=book, h1=h1, cross=cross, paths=paths)
