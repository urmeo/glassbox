from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations, groupby
from numbers import Integral, Rational, Real
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Union

from . import core


# Scoring

CEILING_VARIANT = "raw"
BASELINE_VARIANT = "baseline"

Cell = Tuple[str, str, str]


@dataclass(frozen=True)
class QuestionResult:
    scenario_id: str
    reader: str
    reader_family: str
    reader_simulated: bool
    variant: str
    question_id: str
    chosen: Optional[str]
    correct_answer: str
    correct: bool
    method: str
    replicate: int = 0
    reader_provider: str = ""
    reader_model: str = ""
    reader_model_family: str = "unknown"
    reader_family_resolved: bool = False
    reader_family_basis: str = "unknown"
    reader_deterministic: Optional[bool] = None
    stimulus_kind: str = "unknown"
    stimulus_text_sha256: Optional[str] = None
    stimulus_image_sha256: Optional[str] = None
    render_metadata: Dict[str, Any] = field(default_factory=dict)
    raw_answer: str = ""


def read_all(
    scenarios: Sequence[core.Scenario],
    readers: Sequence[core.Reader],
    variants: Optional[Sequence[str]] = None,
    skip_render: bool = False,
    out_dir: Optional[str] = None,
    replicates: int = 1,
) -> List[QuestionResult]:
    core.positive_integer(replicates, "replicates")
    if not isinstance(skip_render, bool):
        raise ValueError("skip_render must be a boolean")
    try:
        scenarios, readers = list(scenarios), list(readers)
    except TypeError as exc:
        raise ValueError("scenarios and readers must be iterable selections") from exc
    _inputs(scenarios, readers, skip_render, label="scoring")
    variants = _select_variants(variants, core.ALL_VARIANTS, "variants")
    stim_dir = os.path.join(out_dir, "stimuli") if out_dir else None
    results: List[QuestionResult] = []
    for scenario in scenarios:
        for variant in variants:
            presentation = core.build_variant(scenario, variant)
            stimulus = core.render_stimulus(presentation, out_dir=stim_dir, skip=skip_render)
            for reader in readers:
                identity = core.resolve_family(reader.name)
                model_family = getattr(reader, "model_family", identity.family)
                resolved = getattr(reader, "family_resolved", identity.resolved)
                basis = getattr(reader, "family_basis", identity.basis)
                for replicate in range(replicates):
                    for question in scenario.questions:
                        answer = reader.answer(scenario, question, stimulus)
                        results.append(
                            QuestionResult(
                                scenario_id=scenario.id,
                                reader=reader.name,
                                reader_family=reader.family,
                                reader_simulated=reader.simulated,
                                variant=variant,
                                question_id=question.id,
                                chosen=answer.choice_id,
                                correct_answer=question.answer,
                                correct=(answer.choice_id == question.answer),
                                method=answer.method,
                                replicate=replicate,
                                reader_provider=getattr(
                                    reader, "provider", identity.provider
                                ),
                                reader_model=getattr(reader, "model", identity.model),
                                reader_model_family=model_family,
                                reader_family_resolved=resolved,
                                reader_family_basis=basis,
                                reader_deterministic=getattr(
                                    reader, "deterministic", None
                                ),
                                stimulus_kind=stimulus.kind,
                                stimulus_text_sha256=stimulus.text_sha256,
                                stimulus_image_sha256=stimulus.image_sha256,
                                render_metadata=deepcopy(stimulus.render_metadata),
                                raw_answer=answer.raw,
                            )
                        )
    return results


class ScoreBook:
    def __init__(self, results: Iterable[QuestionResult]):
        self.results = list(results)
        self.readers = sorted({r.reader for r in self.results})
        self.scenarios = sorted({r.scenario_id for r in self.results})
        self.variants = _ordered_variants({r.variant for r in self.results})
        self._reader_rows: Dict[str, QuestionResult] = {}
        self._cell: Dict[Cell, List[QuestionResult]] = {}
        identity_fields = (
            "reader_family", "reader_simulated", "reader_provider", "reader_model",
            "reader_model_family", "reader_family_resolved", "reader_family_basis",
            "reader_deterministic",
        )
        for result in self.results:
            previous = self._reader_rows.setdefault(result.reader, result)
            if any(getattr(previous, name) != getattr(result, name) for name in identity_fields):
                raise ValueError("inconsistent identity for reader %r" % result.reader)
            cell = (result.reader, result.scenario_id, result.variant)
            self._cell.setdefault(cell, []).append(result)
        self._simulated = {name: row.reader_simulated for name, row in self._reader_rows.items()}
        self._deterministic = {name: row.reader_deterministic for name, row in self._reader_rows.items()}

    def accuracy(self, reader: str, scenario: str, variant: str) -> float:
        cell = self._cell.get((reader, scenario, variant), [])
        return _mean(qr.correct for qr in cell)

    def ceiling(self, reader: str, scenario: str) -> float:
        return self.accuracy(reader, scenario, CEILING_VARIANT)

    def baseline(self, reader: str, scenario: str) -> float:
        return self.accuracy(reader, scenario, BASELINE_VARIANT)

    def lift(self, reader: str, scenario: str, variant: str) -> float:
        return self.accuracy(reader, scenario, variant) - self.baseline(
            reader, scenario
        )

    def ceiling_gap(self, reader: str, scenario: str, variant: str) -> float:
        return self.accuracy(reader, scenario, variant) - self.ceiling(reader, scenario)

    def accuracy_by_replicate(
        self, reader: str, scenario: str, variant: str
    ) -> List[float]:
        cell = self._cell.get((reader, scenario, variant), [])
        by_rep: Dict[int, List[bool]] = {}
        for qr in cell:
            by_rep.setdefault(qr.replicate, []).append(qr.correct)
        return [statistics.fmean(v) for _, v in sorted(by_rep.items())]

    def accuracy_std(self, reader: str, scenario: str, variant: str) -> float:
        accs = self.accuracy_by_replicate(reader, scenario, variant)
        return statistics.pstdev(accs) if accs else float("nan")

    def answer_stability(self, reader: str, scenario: str, variant: str) -> float:
        cell = self._cell.get((reader, scenario, variant), [])
        by_q: Dict[str, List[Optional[str]]] = {}
        for qr in cell:
            by_q.setdefault(qr.question_id, []).append(qr.chosen)
        fractions = [max(Counter(chosen).values()) / len(chosen)
                     for chosen in by_q.values()]
        return _mean(fractions)

    def mean_lift(self, scenario: str, variant: str) -> float:
        return _mean(self.lift(reader, scenario, variant) for reader in self.readers)

    def mean_accuracy(self, scenario: str, variant: str) -> float:
        return _mean(self.accuracy(reader, scenario, variant) for reader in self.readers)

    def interface_variants(self) -> List[str]:
        return [v for v in self.variants if v in core.INTERFACE_VARIANTS]

    def replicate_count(self) -> int:
        return max((qr.replicate for qr in self.results), default=-1) + 1

    def all_simulated(self) -> bool:
        return bool(self.readers) and all(self._simulated.values())

    def any_simulated(self) -> bool:
        return any(self._simulated.values())

    def all_deterministic(self) -> bool:
        return bool(self.readers) and all(
            v is True for v in self._deterministic.values()
        )

    def any_real(self) -> bool:
        return any(not sim for sim in self._simulated.values())


def _mean(values: Iterable[float]) -> float:
    try:
        return statistics.fmean(values)
    except statistics.StatisticsError:
        return float("nan")


def _ordered_variants(present: set) -> List[str]:
    ordered = [v for v in core.ALL_VARIANTS if v in present]
    ordered += sorted(v for v in present if v not in core.ALL_VARIANTS)
    return ordered


# Judge

@dataclass(frozen=True)
class Preference:
    scenario_id: str
    variant: str
    score: float
    judge: str
    simulated: bool


@dataclass(frozen=True)
class PairwiseJudgment:
    outcome: str
    raw: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in ("A", "B", "tie", "invalid", "abstained"):
            raise ValueError("pairwise outcome must be A, B, tie, invalid or abstained")
        if not isinstance(self.raw, str):
            raise ValueError("pairwise raw reply must be text")

    @property
    def status(self) -> str:
        return "valid" if self.outcome in ("A", "B") else self.outcome


@dataclass(frozen=True)
class PairwiseComparison:
    scenario_id: str
    variant_a: str
    variant_b: str
    outcome: str
    status: str
    raw: str


@dataclass(frozen=True)
class VariantJudgeScore:
    variant: str
    credit: float
    denominator: int
    planned_count: int
    score: float


@dataclass(frozen=True)
class ScenarioJudgeEvidence:
    scenario_id: str
    scenario_sha256: str
    cache_key: str
    protocol: str
    comparisons: Tuple[PairwiseComparison, ...]
    variant_scores: Tuple[VariantJudgeScore, ...]
    valid_count: int
    invalid_count: int
    abstained_count: int
    tie_count: int


@dataclass(frozen=True)
class JudgeMetadata:
    name: str
    spec: str
    target: str
    modality: str
    protocol: str
    simulated: bool
    deterministic: bool
    variants: Tuple[str, ...] = ()
    comparison_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    abstained_count: int = 0
    tie_count: int = 0
    evidence: Tuple[ScenarioJudgeEvidence, ...] = ()

    @property
    def score_label(self) -> str:
        return "synthetic polish" if self.target == "synthetic_polish" else "judge score"


class Judge:
    name = "judge"
    spec = "custom"
    simulated = False
    deterministic = False
    target = "unknown"
    modality = "unknown"
    protocol = "custom"

    def preference(self, scenario: core.Scenario, presentation: core.Presentation) -> float:
        raise NotImplementedError

    def metadata(self) -> JudgeMetadata:
        return JudgeMetadata(
            self.name,
            self.spec,
            self.target,
            self.modality,
            self.protocol,
            self.simulated,
            self.deterministic,
        )


class SimulatedJudge(Judge):
    name = "simulated:polish"
    spec = "polish"
    simulated = True
    deterministic = True
    target = "synthetic_polish"
    modality = "presentation_features"
    protocol = "fixed_polish_weights.v1"

    def preference(self, scenario: core.Scenario, presentation: core.Presentation) -> float:
        features = presentation.features
        score = 0.30
        for feature, weight in (
            ("polished", 0.35),
            ("highlight", 0.08),
            ("derived", 0.05),
            ("sorted", 0.04),
            ("as_text", -0.15),
        ):
            if feature in features:
                score += weight
        if presentation.order:
            avg_fields = statistics.fmean(
                len(presentation.shown[item_id]) for item_id in presentation.order
            )
            score -= 0.015 * avg_fields
        return max(0.0, min(1.0, score))


def validate_judge_spec(spec: str) -> str:
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("judge must be a nonempty string")
    spec = spec.strip()
    if spec in ("polish", "simulated", "simulated:polish"):
        return "polish"
    if spec in ("pairwise", "pairwise:simulated", "pairwise:polish"):
        return "pairwise"
    raise ValueError(
        "unknown judge %r (have: polish, pairwise)" % spec
    )


def _select_variants(variants, allowed, label):
    if isinstance(variants, (str, bytes)):
        raise ValueError("%s must be a nonempty sequence" % label)
    try:
        selected = tuple(allowed if variants is None else variants)
    except TypeError as exc:
        raise ValueError("%s must be an iterable selection" % label) from exc
    if (not selected or any(not isinstance(value, str) or value not in allowed for value in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("%s must be nonempty, known and distinct" % label)
    return selected


def _variants(variants: Optional[Sequence[str]]) -> Tuple[str, ...]:
    return _select_variants(variants, core.INTERFACE_VARIANTS, "judge variants")


def build_judge(spec: str, variants: Optional[Sequence[str]] = None) -> Judge:
    spec = validate_judge_spec(spec)
    selected = _variants(variants)
    if spec == "polish":
        return SimulatedJudge()
    return PairwiseRatingJudge(SimulatedPairwiseJudge(), selected, spec=spec)


def judge_all(
    judge: Judge, scenario: core.Scenario, presentations: Sequence[core.Presentation]
) -> List[Preference]:
    return [
        Preference(
            scenario.id,
            presentation.variant,
            judge.preference(scenario, presentation),
            judge.name,
            judge.simulated,
        )
        for presentation in presentations
    ]


class PairwiseJudge:
    name = "pairwise"
    simulated = False
    deterministic = False
    target = "unknown"
    modality = "unknown"
    protocol = "ordered_pairs_winrate.v1"

    def compare(
        self, scenario: core.Scenario, a: core.Presentation, b: core.Presentation
    ) -> Union[PairwiseJudgment, str]:
        raise NotImplementedError


class SimulatedPairwiseJudge(PairwiseJudge):
    name = "simulated:pairwise-polish"
    simulated = True
    deterministic = True
    target = "synthetic_polish"
    modality = "presentation_features"

    def __init__(self):
        self._rating = SimulatedJudge()

    def compare(self, scenario: core.Scenario, a: core.Presentation, b: core.Presentation) -> str:
        first_score = self._rating.preference(scenario, a)
        second_score = self._rating.preference(scenario, b)
        if first_score != second_score:
            return "A" if first_score > second_score else "B"
        return "A" if a.variant <= b.variant else "B"


class PairwiseRatingJudge(Judge):
    def __init__(
        self,
        pairwise: PairwiseJudge,
        variants: Optional[Sequence[str]] = None,
        *,
        spec: str = "custom",
    ):
        self._pairwise = pairwise
        self._variants = _variants(variants)
        self.name = "winrate(%s)" % pairwise.name
        self.spec = spec
        self.simulated = pairwise.simulated
        self.deterministic = pairwise.deterministic
        self.target = pairwise.target
        self.modality = pairwise.modality
        self.protocol = "counterbalanced_pairs_winrate.v2"
        self.comparison_count = 0
        self._cache: Dict[str, ScenarioJudgeEvidence] = {}

    def metadata(self) -> JudgeMetadata:
        evidence = tuple(self._cache.values())
        return replace(
            super().metadata(),
            variants=self._variants,
            comparison_count=self.comparison_count,
            valid_count=sum(item.valid_count for item in evidence),
            invalid_count=sum(item.invalid_count for item in evidence),
            abstained_count=sum(item.abstained_count for item in evidence),
            tie_count=sum(item.tie_count for item in evidence),
            evidence=evidence,
        )

    def _evidence(self, scenario: core.Scenario) -> ScenarioJudgeEvidence:
        scenario_hash = core.scenario_sha256(scenario)
        presentations = {
            variant: core.build_variant(scenario, variant) for variant in self._variants
        }
        identity = (
            scenario_hash,
            self._variants,
            tuple(
                (
                    variant,
                    {**asdict(presentation), "features": sorted(presentation.features)},
                    presentation.to_text(),
                )
                for variant, presentation in presentations.items()
            ),
            self.name,
            self.protocol,
            self._pairwise.name,
            self._pairwise.protocol,
            self.target,
            self.modality,
        )
        key = hashlib.sha256(
            json.dumps(
                identity, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]
        credits = {variant: 0.0 for variant in self._variants}
        denominators = {variant: 0 for variant in self._variants}
        comparisons = []
        for first, second in combinations(self._variants, 2):
            for variant_a, variant_b in ((first, second), (second, first)):
                result = self._pairwise.compare(
                    scenario, presentations[variant_a], presentations[variant_b]
                )
                if isinstance(result, str):
                    result = PairwiseJudgment(result, raw=result)
                if not isinstance(result, PairwiseJudgment):
                    raise ValueError(
                        "pairwise judge must return a judgment or outcome string"
                    )
                comparisons.append(
                    PairwiseComparison(
                        scenario_id=scenario.id,
                        variant_a=variant_a,
                        variant_b=variant_b,
                        outcome=result.outcome,
                        status=result.status,
                        raw=result.raw,
                    )
                )
                if result.status in ("invalid", "abstained"):
                    continue
                for variant, label in ((variant_a, "A"), (variant_b, "B")):
                    denominators[variant] += 1
                    credit = 0.5 if result.outcome == "tie" else float(result.outcome == label)
                    credits[variant] += credit
        evidence = ScenarioJudgeEvidence(
            scenario_id=scenario.id,
            scenario_sha256=scenario_hash,
            cache_key=key,
            protocol=self.protocol,
            comparisons=tuple(comparisons),
            variant_scores=tuple(
                VariantJudgeScore(
                    variant=variant,
                    credit=credits[variant],
                    denominator=denominators[variant],
                    planned_count=2 * (len(self._variants) - 1),
                    score=(
                        credits[variant] / denominators[variant]
                        if denominators[variant] else float("nan")
                    ),
                )
                for variant in self._variants
            ),
            valid_count=sum(item.status in ("valid", "tie") for item in comparisons),
            invalid_count=sum(item.status == "invalid" for item in comparisons),
            abstained_count=sum(item.status == "abstained" for item in comparisons),
            tie_count=sum(item.status == "tie" for item in comparisons),
        )
        self._cache[key] = evidence
        self.comparison_count += len(comparisons)
        return evidence

    def preference(self, scenario: core.Scenario, presentation: core.Presentation) -> float:
        if presentation.variant not in self._variants:
            raise ValueError("presentation variant was not selected for the judge")
        return next(
            item.score for item in self._evidence(scenario).variant_scores
            if item.variant == presentation.variant
        )


# Analysis

def _fractional_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    offset = 0
    for _, indices in groupby(order, key=values.__getitem__):
        tied = list(indices)
        rank = offset + (len(tied) + 1) / 2
        for index in tied:
            ranks[index] = rank
        offset += len(tied)
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = _scaled_deltas(a), _scaled_deltas(b)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    a, b = [x - ma for x in a], [y - mb for y in b]
    cov = math.fsum(x * y for x, y in zip(a, b))
    va, vb = math.fsum(x * x for x in a), math.fsum(y * y for y in b)
    if va == 0 or vb == 0:
        return float("nan")
    return max(-1.0, min(1.0, cov / ((va**0.5) * (vb**0.5))))


def _scaled_deltas(values: Sequence[float]) -> List[float]:
    normalized = []
    for value in values:
        if isinstance(value, Rational) or type(value) is float:
            normalized.append(value)
        else:
            ratio = getattr(value, "as_integer_ratio", None)
            normalized.append(Fraction(*ratio()) if ratio is not None else float(value))
    values = normalized
    deltas = [x - values[0] for x in values]
    scale = max(map(abs, deltas)) or 1
    try:
        finite = math.isfinite(scale)
    except OverflowError:
        finite = False
    if finite:
        return [x / scale for x in deltas]
    scale = max(map(abs, values)) or 1
    return [x / scale for x in values]


def _vectors(
    xs: Sequence[float], ys: Sequence[float]
) -> Tuple[List[float], List[float]]:
    if len(xs) != len(ys):
        raise ValueError("correlation needs equal-length sequences")
    result = []
    for values in (xs, ys):
        vector = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("correlation values must be finite real numbers")
            try:
                number = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    "correlation values must be finite real numbers"
                ) from exc
            if not math.isfinite(number):
                raise ValueError("correlation values must be finite real numbers")
            vector.append(int(value) if isinstance(value, Integral) else value)
        result.append(vector)
    return result[0], result[1]


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    xs, ys = _vectors(xs, ys)
    if len(xs) < 2:
        return float("nan")
    return _pearson(_fractional_ranks(xs), _fractional_ranks(ys))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    xs, ys = _vectors(xs, ys)
    if len(xs) < 2:
        return float("nan")
    return _pearson(xs, ys)


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    xs, ys = _vectors(xs, ys)
    concordant = discordant = tx = ty = 0
    for i, j in combinations(range(len(xs)), 2):
        dx = int(xs[i] > xs[j]) - int(xs[i] < xs[j])
        dy = int(ys[i] > ys[j]) - int(ys[i] < ys[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif (dx > 0) == (dy > 0):
            concordant += 1
        else:
            discordant += 1
    denom = ((concordant + discordant + tx) * (concordant + discordant + ty)) ** 0.5
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


@dataclass(frozen=True)
class InterfaceRow:
    variant: str
    comp_lift: float
    mean_accuracy: float
    pref_score: float
    comp_rank: Optional[int]
    pref_rank: Optional[int]


@dataclass(frozen=True)
class Reversal:
    preferred: str
    understood: str
    pref_gap: float
    comp_gap: float


@dataclass(frozen=True)
class ScenarioH1:
    scenario_id: str
    rows: List[InterfaceRow]
    spearman: float
    reader_agreement: float
    reversals: List[Reversal]


@dataclass(frozen=True)
class H1Analysis:
    scenarios: List[ScenarioH1]
    mean_spearman: float
    simulated: bool
    readers: List[str]
    judge: JudgeMetadata = field(default_factory=lambda: JudgeMetadata(
        "unknown", "unknown", "unknown", "unknown", "unknown", False, False))
    any_reader_simulated: bool = False
    mixed_readers: bool = False
    all_reader_deterministic: bool = False

    @property
    def deterministic(self) -> bool:
        return self.all_reader_deterministic and self.judge.deterministic

    @property
    def divergence_found(self) -> bool:
        return any(s.reversals for s in self.scenarios)


def _rank_desc(scores: List[float]) -> List[Optional[int]]:
    order = sorted(
        (index for index, score in enumerate(scores) if math.isfinite(score)),
        key=lambda index: -scores[index],
    )
    ranks: List[Optional[int]] = [None] * len(scores)
    last_score = None
    last_rank = 0
    for pos, idx in enumerate(order, start=1):
        if last_score is None or scores[idx] != last_score:
            last_rank = pos
            last_score = scores[idx]
        ranks[idx] = last_rank
    return ranks


def _paired_spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [
        (first, second)
        for first, second in zip(xs, ys)
        if math.isfinite(first) and math.isfinite(second)
    ]
    if len(pairs) < 2:
        return float("nan")
    first, second = zip(*pairs)
    return spearman(first, second)


def _agreement(vectors: Sequence[Sequence[float]]) -> float:
    correlations = [_paired_spearman(first, second)
                    for first, second in combinations(vectors, 2)]
    return _mean(value for value in correlations if not math.isnan(value))


def _reader_agreement(book: ScoreBook, scenario: str, variants: List[str]) -> float:
    return _agreement([
        [book.accuracy(reader, scenario, variant) for variant in variants]
        for reader in book.readers
    ])


def analyze_h1(
    book: ScoreBook, scenarios: Sequence[core.Scenario], judge: Judge
) -> H1Analysis:
    variants = book.interface_variants()
    if not variants or not scenarios or not book.readers:
        raise ValueError("H1 needs scored interfaces, scenarios and readers")
    scenario_reports: List[ScenarioH1] = []

    for scenario in scenarios:
        sid = scenario.id
        presentations = [core.build_variant(scenario, v) for v in variants]
        prefs = {p.variant: p.score for p in judge_all(judge, scenario, presentations)}

        comp = [book.mean_lift(sid, v) for v in variants]
        pref = [prefs[v] for v in variants]
        comp_ranks = _rank_desc(comp)
        pref_ranks = _rank_desc(pref)

        rows = [
            InterfaceRow(
                variant=v,
                comp_lift=comp[i],
                mean_accuracy=book.mean_accuracy(sid, v),
                pref_score=pref[i],
                comp_rank=comp_ranks[i],
                pref_rank=pref_ranks[i],
            )
            for i, v in enumerate(variants)
        ]

        comprehension = dict(zip(variants, comp))
        reversals = [
            Reversal(
                preferred=preferred,
                understood=understood,
                pref_gap=prefs[preferred] - prefs[understood],
                comp_gap=comprehension[understood] - comprehension[preferred],
            )
            for preferred, understood in _reversal_pairs(variants, prefs, comprehension)
        ]

        scenario_reports.append(
            ScenarioH1(
                scenario_id=sid,
                rows=rows,
                spearman=_paired_spearman(comp, pref),
                reader_agreement=_reader_agreement(book, sid, variants),
                reversals=reversals,
            )
        )

    mean_rho = _mean(s.spearman for s in scenario_reports if not math.isnan(s.spearman))
    return H1Analysis(
        scenarios=scenario_reports,
        mean_spearman=mean_rho,
        simulated=book.all_simulated(),
        readers=book.readers,
        judge=judge.metadata(),
        any_reader_simulated=book.any_simulated(),
        mixed_readers=book.any_simulated() and book.any_real(),
        all_reader_deterministic=book.all_deterministic(),
    )


def _reversal_pairs(
    variants: List[str], pref: Dict[str, float], comp: Dict[str, float]
) -> List[Tuple[str, str]]:
    return [
        (i, j)
        for i in variants
        for j in variants
        if all(math.isfinite(value) for value in (pref[i], pref[j], comp[i], comp[j]))
        and pref[i] > pref[j] and comp[i] < comp[j]
    ]


# Repair

START_VARIANT = "cards"
MAX_TURNS = 4


@dataclass(frozen=True)
class RepairTurn:
    turn: int
    variant: str
    features: List[str]
    chosen: Optional[str]
    correct: bool
    transform: str


@dataclass(frozen=True)
class RepairResult:
    scenario_id: str
    reader: str
    question_id: str
    turns: List[RepairTurn]
    converged: bool
    turns_to_understanding: Optional[int]


def _next_transform(presentation: core.Presentation, question: core.Question):
    metric = question.target_field
    if not all(presentation.shows_field(iid, metric) for iid in presentation.order):
        return "show the target %s" % (presentation.value_label or metric), frozenset(
            {"detail", "derived"}
        )
    op = question.compute["op"]
    if op.startswith("count_"):
        if "sorted" not in presentation.features:
            return "sort the target values for the count comparison", frozenset({"sorted"})
    elif presentation.highlight_item is None:
        return "highlight and sort by the target %s" % op, frozenset({"highlight", "sorted"})
    return None


def repair(
    scenario: core.Scenario,
    reader: core.Reader,
    question: core.Question,
    max_turns: int = MAX_TURNS,
    skip_render: bool = False,
    out_dir: Optional[str] = None,
) -> RepairResult:
    if type(max_turns) is not int or max_turns < 0:
        raise ValueError("max_turns must be a nonnegative integer")
    if not isinstance(skip_render, bool):
        raise ValueError("skip_render must be a boolean")
    presentation = core.build_variant(scenario, START_VARIANT, target=question)
    turns: List[RepairTurn] = []
    description = ""
    first_correct_turn = None
    for turn_index in range(max_turns + 1):
        stimulus = core.render_stimulus(presentation, out_dir=out_dir, skip=skip_render)
        answer = reader.answer(scenario, question, stimulus)
        correct = answer.choice_id == question.answer
        turns.append(
            RepairTurn(
                turn_index,
                presentation.variant,
                sorted(presentation.features),
                answer.choice_id,
                correct,
                description,
            )
        )
        if correct:
            first_correct_turn = turn_index
            break
        if turn_index == max_turns:
            break
        next_step = _next_transform(presentation, question)
        if next_step is None:
            break
        description, added = next_step
        presentation = core.with_features(scenario, presentation, added, target=question)
    return RepairResult(
        scenario.id, reader.name, question.id, turns,
        first_correct_turn is not None, first_correct_turn,
    )


# Reward

GENERATOR_FEATURE_POOL = ["detail", "derived", "highlight", "sorted", "polished"]

CARDS_VARIANT = "cards"


def _accuracy(
    scenario: core.Scenario, presentation: core.Presentation, readers: Sequence[core.Reader], skip_render: bool
) -> float:
    stimulus = core.render_stimulus(presentation, skip=skip_render)
    return statistics.fmean(
        sum(reader.answer(scenario, question, stimulus).choice_id == question.answer
            for question in scenario.questions) / len(scenario.questions)
        for reader in readers
    )


def interface_reward(
    scenario: core.Scenario,
    features: FrozenSet[str],
    readers: Sequence[core.Reader],
    skip_render: bool = True,
) -> float:
    return scenario_set_reward([scenario], features, readers, skip_render)


def scenario_set_reward(
    scenarios: Sequence[core.Scenario],
    features: FrozenSet[str],
    readers: Sequence[core.Reader],
    skip_render: bool = True,
) -> float:
    _inputs(scenarios, readers, skip_render)
    return _scenario_set_reward(scenarios, features, readers, skip_render)


def _scenario_set_reward(scenarios, features, readers, skip_render):
    rewards = []
    for scenario in scenarios:
        candidate = core.build_presentation(scenario, frozenset(features), variant="candidate")
        baseline = core.build_variant(scenario, BASELINE_VARIANT)
        rewards.append(_accuracy(scenario, candidate, readers, skip_render)
                       - _accuracy(scenario, baseline, readers, skip_render))
    return statistics.fmean(rewards)


def _subsets(pool: Sequence[str]):
    for size in range(len(pool) + 1):
        for combination in combinations(pool, size):
            yield frozenset(combination)


def search_best_interface(
    scenarios: Sequence[core.Scenario],
    readers: Sequence[core.Reader],
    pool: Optional[Sequence[str]] = None,
    skip_render: bool = True,
) -> List[Tuple[FrozenSet[str], float]]:
    _inputs(scenarios, readers, skip_render)
    pool = _pool(pool)
    scored = [
        (features, _scenario_set_reward(scenarios, features, readers, skip_render))
        for features in _subsets(pool)
    ]
    scored.sort(key=lambda candidate: candidate[1], reverse=True)
    return scored


@dataclass(frozen=True)
class EvalResult:
    generator_features: FrozenSet[str]
    generator_reward: float
    cards_reward: float
    beats_preference_tuned: bool
    beats_plaintext: bool
    candidate_count: int = 32
    scenario_count: int = 0
    reader_count: int = 0
    selection_protocol: str = "in_sample_exhaustive_feature_search.v1"
    baseline_sampling: str = "fresh_per_candidate_and_cards"
    any_reader_simulated: bool = False
    all_readers_simulated: bool = False
    selection_inputs_sha256: str = ""


def _reader_metadata(reader: core.Reader) -> Dict[str, Any]:
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


def _selection_inputs_sha256(
    scenarios: Sequence[core.Scenario], readers: Sequence[core.Reader], skip_render: bool
) -> str:
    inputs = {
        "scenarios": [core.scenario_sha256(scenario) for scenario in scenarios],
        "readers": [_reader_metadata(reader) for reader in readers],
        "skip_render": skip_render,
    }
    encoded = json.dumps(
        inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_generator(
    scenarios: Sequence[core.Scenario],
    readers: Sequence[core.Reader],
    pool: Optional[Sequence[str]] = None,
    skip_render: bool = True,
) -> EvalResult:
    _inputs(scenarios, readers, skip_render)
    pool = _pool(pool)
    inputs_sha256 = _selection_inputs_sha256(scenarios, readers, skip_render)
    ranked = search_best_interface(
        scenarios, readers, pool=pool, skip_render=skip_render
    )
    best_features, best_reward = ranked[0]
    cards_reward = scenario_set_reward(
        scenarios, core.VARIANT_FEATURES[CARDS_VARIANT], readers, skip_render
    )
    return EvalResult(
        generator_features=best_features,
        generator_reward=best_reward,
        cards_reward=cards_reward,
        beats_preference_tuned=best_reward > cards_reward,
        beats_plaintext=best_reward > 0.0,
        candidate_count=len(ranked),
        scenario_count=len(scenarios),
        reader_count=len(readers),
        any_reader_simulated=any(reader.simulated for reader in readers),
        all_readers_simulated=all(reader.simulated for reader in readers),
        selection_inputs_sha256=inputs_sha256,
    )


@dataclass(frozen=True)
class EvaluationMetrics:
    selected_accuracy: float
    cards_accuracy: float
    baseline_accuracy: float
    selected_lift: float
    cards_lift: float


@dataclass(frozen=True)
class ScenarioEvaluation(EvaluationMetrics):
    scenario_id: str
    scenario_sha256: str
    question_count: int


@dataclass(frozen=True)
class HeldOutResult:
    selection_sha256: str
    selected_features: FrozenSet[str]
    selection_scenario_ids: Tuple[str, ...]
    selection_reward: float
    scenarios: Tuple[ScenarioEvaluation, ...]
    aggregate: EvaluationMetrics
    reader_names: Tuple[str, ...]
    skip_render: bool
    all_readers_simulated: bool
    any_reader_simulated: bool
    evaluation_protocol: str = "frozen_features_disjoint_scenarios.v1"
    baseline_sampling: str = "shared_per_scenario"


def evaluate_selection(
    artifact: "SelectionArtifact",
    scenarios: Sequence[core.Scenario],
    readers: Sequence[core.Reader],
    skip_render: Optional[bool] = None,
) -> HeldOutResult:
    artifact = validate_selection(artifact)
    skip_render = artifact.skip_render if skip_render is None else skip_render
    _inputs(scenarios, readers, skip_render)
    validate_evaluation(artifact, scenarios)
    features = artifact.features
    presentations = [
        (scenario, core.build_presentation(scenario, features, variant="selected"),
         core.build_variant(scenario, CARDS_VARIANT),
         core.build_variant(scenario, BASELINE_VARIANT))
        for scenario in scenarios
    ]
    results = []
    for scenario, selected, cards, baseline in presentations:
        baseline_accuracy = _accuracy(scenario, baseline, readers, skip_render)
        selected_accuracy = _accuracy(scenario, selected, readers, skip_render)
        cards_accuracy = _accuracy(scenario, cards, readers, skip_render)
        results.append(
            ScenarioEvaluation(
                selected_accuracy=selected_accuracy,
                cards_accuracy=cards_accuracy,
                baseline_accuracy=baseline_accuracy,
                selected_lift=selected_accuracy - baseline_accuracy,
                cards_lift=cards_accuracy - baseline_accuracy,
                scenario_id=scenario.id,
                scenario_sha256=core.scenario_sha256(scenario),
                question_count=len(scenario.questions),
            )
        )
    aggregate = EvaluationMetrics(
        **{
            name: statistics.fmean(getattr(result, name) for result in results)
            for name in EvaluationMetrics.__dataclass_fields__
        }
    )
    return HeldOutResult(
        selection_sha256=artifact.sha256,
        selected_features=features,
        selection_scenario_ids=artifact.scenario_ids,
        selection_reward=artifact.selection_reward,
        scenarios=tuple(results),
        aggregate=aggregate,
        reader_names=tuple(reader.name for reader in readers),
        skip_render=skip_render,
        all_readers_simulated=all(reader.simulated for reader in readers),
        any_reader_simulated=any(reader.simulated for reader in readers),
    )


def _pool(pool: Optional[Sequence[str]]) -> List[str]:
    if pool is None:
        return list(GENERATOR_FEATURE_POOL)
    if not isinstance(pool, Sequence) or isinstance(pool, (str, bytes)):
        raise ValueError("feature pool must be a sequence")
    result = list(pool)
    if any(
        not isinstance(feature, str) or feature not in GENERATOR_FEATURE_POOL
        for feature in result
    ) or len(set(result)) != len(result):
        raise ValueError("feature pool must contain distinct supported features")
    return result


def _inputs(
    scenarios: Sequence[core.Scenario], readers: Sequence[core.Reader], skip_render: bool,
    *, label: str = "reward",
) -> None:
    if not isinstance(scenarios, Sequence) or not scenarios or any(
        not isinstance(scenario, core.Scenario) or not scenario.questions for scenario in scenarios
    ):
        raise ValueError("%s needs scenarios with questions" % label)
    if not isinstance(readers, Sequence) or not readers or any(
        not isinstance(reader, core.Reader) for reader in readers
    ):
        raise ValueError("%s needs readers" % label)
    for identities, label in (
        ([scenario.id for scenario in scenarios], "scenario IDs"),
        ([reader.name for reader in readers], "reader names"),
    ):
        if (
            any(not isinstance(value, str) or not value.strip() for value in identities)
            or len(set(identities)) != len(identities)
        ):
            raise ValueError("%s must be nonempty and distinct" % label)
    if not isinstance(skip_render, bool):
        raise ValueError("skip_render must be a boolean")


# Selection

FORMAT = "glassbox.selection.v2"
_FIELDS = {
    "format", "glassbox_version", "features", "selection_reward", "candidate_count",
    "selection_protocol", "baseline_sampling", "skip_render", "readers", "scenarios",
    "sha256",
}
_READER_FIELDS = {
    "name", "provider", "model", "model_family", "family_resolved", "family_basis",
    "simulated", "deterministic",
}
_SNAPSHOT_FIELDS = {
    "id", "title", "domain", "description", "data", "questions", "resolved_presentation",
}


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class SelectionArtifact:
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


def scenario_content_sha256(scenario: core.Scenario) -> str:
    return _digest({"data": scenario.data, "questions": [asdict(q) for q in scenario.questions]})


def _source_sha256(scenario: core.Scenario) -> str:
    items = sorted(scenario.items, key=lambda item: item["id"])
    return _digest(_normalize_numbers({"items": items, "derived": scenario.derived}))


def _normalize_numbers(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    return value


def freeze_selection(evaluation: EvalResult, scenarios: Sequence[core.Scenario],
                     readers: Sequence[core.Reader], skip_render: bool) -> SelectionArtifact:
    _inputs(scenarios, readers, skip_render)
    if not isinstance(evaluation, EvalResult):
        raise SelectionError("selection requires a feature-search result")
    if (
        evaluation.scenario_count != len(scenarios)
        or evaluation.reader_count != len(readers)
        or evaluation.selection_inputs_sha256
        != _selection_inputs_sha256(scenarios, readers, skip_render)
    ):
        raise SelectionError("selection inputs do not match the feature-search result")
    raw = {
        "format": FORMAT,
        "glassbox_version": core.__version__,
        "features": sorted(evaluation.generator_features),
        "selection_reward": evaluation.generator_reward,
        "candidate_count": evaluation.candidate_count,
        "selection_protocol": evaluation.selection_protocol,
        "baseline_sampling": evaluation.baseline_sampling,
        "skip_render": skip_render,
        "readers": [_reader_metadata(reader) for reader in readers],
        "scenarios": [
            {"id": scenario.id, "sha256": core.scenario_sha256(scenario),
             "content_sha256": scenario_content_sha256(scenario),
             "source_sha256": _source_sha256(scenario),
             "snapshot": core.scenario_payload(scenario)}
            for scenario in scenarios
        ],
    }
    raw["sha256"] = _digest(raw)
    return parse_selection(raw)


def parse_selection(raw: Any) -> SelectionArtifact:
    raw = core.finite_json(raw, SelectionError)
    if not isinstance(raw, dict):
        raise SelectionError("selection has missing or unknown fields")
    if raw.get("format") != FORMAT:
        raise SelectionError("unsupported selection artifact format; expected %s" % FORMAT)
    _fields(raw, _FIELDS, "selection")
    if not isinstance(raw["glassbox_version"], str) or not raw["glassbox_version"].strip():
        raise SelectionError("glassbox_version must be nonempty text")
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
        scenario = core.parse_scenario(authored)
        errors = core.validate_scenario(scenario)
        if errors:
            raise SelectionError("invalid selection scenario: %s" % "; ".join(errors))
        if (item["id"] != scenario.id or scenario.id in ids
                or core.scenario_payload(scenario) != item["snapshot"]
                or item["sha256"] != core.scenario_sha256(scenario)
                or item["content_sha256"] != scenario_content_sha256(scenario)
                or item["source_sha256"] != _source_sha256(scenario)
                or item["content_sha256"] in contents):
            raise SelectionError("selection scenario identity, snapshot or digest is invalid")
        ids.add(scenario.id)
        contents.add(item["content_sha256"])
    _hash(raw["sha256"], "selection hash")
    unsigned = {key: value for key, value in raw.items() if key != "sha256"}
    if raw["sha256"] != _digest(unsigned):
        raise SelectionError("selection artifact digest does not match its content")
    return SelectionArtifact(_canonical(raw))


def selection_payload(artifact: SelectionArtifact) -> Dict[str, Any]:
    if not isinstance(artifact, SelectionArtifact) or not isinstance(artifact.payload_json, str):
        raise SelectionError("expected a SelectionArtifact")
    return core.strict_json(artifact.payload_json)


def validate_selection(artifact: SelectionArtifact) -> SelectionArtifact:
    validated = parse_selection(selection_payload(artifact))
    return SelectionArtifact(validated.payload_json, artifact.source_path)


def validate_evaluation(artifact: SelectionArtifact, scenarios: Sequence[core.Scenario]) -> None:
    artifact = validate_selection(artifact)
    if (not isinstance(scenarios, Sequence) or not scenarios
            or any(not isinstance(scenario, core.Scenario) for scenario in scenarios)):
        raise SelectionError("evaluation requires explicit scenarios")
    raw = selection_payload(artifact)
    selected_ids = {item["id"] for item in raw["scenarios"]}
    selected_content = {item["content_sha256"] for item in raw["scenarios"]}
    selected_sources = {item["source_sha256"] for item in raw["scenarios"]}
    seen, seen_ids, seen_sources = set(), set(), set()
    for scenario in scenarios:
        errors = core.validate_scenario(scenario)
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
        artifact = parse_selection(core.decode_yaml(source.read_text(encoding="utf-8"), "selection YAML"))
        return SelectionArtifact(artifact.payload_json, str(source))
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise SelectionError("cannot load selection: %s" % error) from None


def write_selection(path: str, artifact: SelectionArtifact) -> str:
    artifact = validate_selection(artifact)
    destination = Path(path)
    (destination,) = core.validate_output_files(destination.parent, (destination.name,))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(core.encode_yaml(selection_payload(artifact)))
    return str(destination)


# Anchor

class AnchorError(ValueError):
    pass


@dataclass(frozen=True)
class AnchorItem:
    scenario: str
    question: str
    human_accuracy: float

    def __post_init__(self) -> None:
        for name in ("scenario", "question"):
            object.__setattr__(self, name, core.config_text(getattr(self, name), name, AnchorError))
        value = self.human_accuracy
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
            or not math.isfinite(value)
        ):
            raise AnchorError("human_accuracy must be finite and in [0,1]")


@dataclass(frozen=True)
class AnchorSet:
    id: str
    source: str
    note: str
    condition: str
    items: List[AnchorItem]
    is_fixture: bool

    def __post_init__(self) -> None:
        for name in ("id", "source", "note"):
            core.config_text(getattr(self, name), name, AnchorError, empty=name != "id")
        core.config_text(self.condition, "condition", AnchorError)
        if self.condition not in core.ALL_VARIANTS:
            raise AnchorError("unknown presentation %r" % self.condition)
        core.boolean(self.is_fixture, "is_fixture", AnchorError)
        if not isinstance(self.items, (list, tuple)) or not self.items:
            raise AnchorError("items must be a nonempty list of AnchorItem objects")
        identities = []
        scenarios = core.load_all_scenarios()
        for item in self.items:
            if not isinstance(item, AnchorItem):
                raise AnchorError("items must contain AnchorItem objects")
            item.__post_init__()
            if item.scenario not in scenarios:
                raise AnchorError("unknown scenario %r" % item.scenario)
            if item.question not in {q.id for q in scenarios[item.scenario].questions}:
                raise AnchorError(
                    "unknown question %r in %r" % (item.question, item.scenario)
                )
            identities.append((item.scenario, item.question))
        if len(set(identities)) != len(identities):
            raise AnchorError("duplicate anchor items")


@dataclass(frozen=True)
class AnchorPoint:
    scenario: str
    question: str
    human_accuracy: float
    model_accuracy: float

    @property
    def abs_gap(self) -> float:
        return abs(self.model_accuracy - self.human_accuracy)


@dataclass(frozen=True)
class AnchorResult:
    anchor: AnchorSet
    readers: List[str]
    points: List[AnchorPoint]
    spearman: float
    pearson: float
    kendall: float

    @property
    def n_items(self) -> int:
        return len(self.points)

    def breaks(self, top: int = 3) -> List[AnchorPoint]:
        return sorted(self.points, key=lambda point: point.abs_gap, reverse=True)[:top]


def parse_anchor_set(raw: Dict) -> AnchorSet:
    core.object_fields(raw, AnchorSet.__dataclass_fields__, AnchorError)
    raw_items = raw.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise AnchorError("items must be a nonempty list")
    values = dict(raw, items=[core.parse_config(item, AnchorItem, AnchorError) for item in raw_items])
    return core.parse_config(values, AnchorSet, AnchorError,
                             source="", note="", condition="raw", is_fixture=True)


def load_anchor_set(source: str) -> AnchorSet:
    return core.load_config_object("anchors", source, parse_anchor_set, AnchorError)


def run_anchor(
    anchor: AnchorSet,
    readers: List[core.Reader],
    skip_render: bool = False,
    replicates: int = 1,
    out_dir: Optional[str] = None,
) -> AnchorResult:
    if not isinstance(anchor, AnchorSet):
        raise AnchorError("anchor must be an AnchorSet")
    anchor.__post_init__()
    core.boolean(skip_render, "skip_render", AnchorError)
    core.positive_integer(replicates, "replicates", AnchorError)
    if (
        not isinstance(readers, (list, tuple))
        or not readers
        or any(not isinstance(reader, core.Reader) for reader in readers)
    ):
        raise AnchorError("readers must be a nonempty list of Reader objects")
    names = [reader.name for reader in readers]
    if (
        any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise AnchorError("reader names must be nonempty and distinct")
    if out_dir is not None:
        protected = (
            (anchor._source_path,)
            if getattr(anchor, "_source_path", None) is not None
            else ()
        )
        core.validate_output_files(
            out_dir, ("anchor.md", "anchor.yaml"), protected_paths=protected
        )
    scenarios_by_id = core.load_all_scenarios()
    question_ids = {}
    for item in anchor.items:
        question_ids.setdefault(item.scenario, set()).add(item.question)
    needed = [
        replace(
            scenarios_by_id[scenario_id],
            questions=[
                question
                for question in scenarios_by_id[scenario_id].questions
                if question.id in selected_ids
            ],
        )
        for scenario_id, selected_ids in question_ids.items()
    ]
    results = read_all(
        needed,
        readers,
        variants=[anchor.condition],
        skip_render=skip_render,
        out_dir=out_dir,
        replicates=replicates,
    )

    by_item: Dict = {}
    for result in results:
        by_item.setdefault((result.scenario_id, result.question_id), []).append(result.correct)

    points: List[AnchorPoint] = []
    for item in anchor.items:
        answers = by_item.get((item.scenario, item.question), [])
        if not answers:
            raise AnchorError("no reader answers for %s/%s" % (item.scenario, item.question))
        points.append(AnchorPoint(item.scenario, item.question, item.human_accuracy,
                                  model_accuracy=sum(answers) / len(answers)))

    human = [point.human_accuracy for point in points]
    model = [point.model_accuracy for point in points]
    return AnchorResult(
        anchor=anchor,
        readers=names,
        points=points,
        spearman=spearman(model, human),
        pearson=pearson(model, human),
        kendall=kendall_tau(model, human),
    )


# Studies

class StudyError(ValueError):
    pass


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
        object.__setattr__(self, "id", core.config_text(self.id, "id", StudyError))
        core.config_text(self.description, "description", StudyError, empty=True)
        reader_specs = core.strings(self.readers, "readers", StudyError)
        try:
            core.expand_reader_specs(reader_specs)
            judge = validate_judge_spec(self.judge)
        except ValueError as exc:
            raise StudyError(str(exc)) from None
        object.__setattr__(self, "readers", reader_specs)
        object.__setattr__(self, "judge", judge)
        for field in ("scenarios", "interfaces"):
            values = core.strings(getattr(self, field), field, StudyError, optional=True)
            object.__setattr__(self, field, values)
        if self.interfaces is not None:
            for variant in self.interfaces:
                if variant not in core.INTERFACE_VARIANTS:
                    raise StudyError("unknown testable interface %r" % variant)
        core.positive_integer(self.replicates, "replicates", StudyError)
        core.boolean(self.skip_render, "skip_render", StudyError)

    def to_meta(self) -> Dict[str, Any]:
        return asdict(self)


def parse_study(raw: Dict[str, Any]) -> StudyConfig:
    return core.parse_config(raw, StudyConfig, StudyError)


def load_study(source: str) -> StudyConfig:
    return core.load_config_object("studies", source, parse_study, StudyError)


@dataclass(frozen=True)
class StudyResult:
    config: StudyConfig
    book: ScoreBook
    h1: H1Analysis
    paths: Dict[str, str]


def select_scenarios(scenario_ids: Optional[Sequence[str]]) -> List[core.Scenario]:
    scenario_ids = core.strings(scenario_ids, "scenarios", StudyError, optional=True)
    scenarios = core.load_all_scenarios()
    if scenario_ids is None:
        return list(scenarios.values())
    for scenario_id in scenario_ids:
        if scenario_id not in scenarios:
            raise StudyError(
                "unknown scenario %r (have: %s)" % (scenario_id, ", ".join(scenarios))
            )
    return [scenarios[scenario_id] for scenario_id in scenario_ids]


def run_study(config: StudyConfig, out_dir: str) -> StudyResult:
    if not isinstance(config, StudyConfig):
        raise StudyError("config must be a StudyConfig")
    config.__post_init__()
    protected = (
        (config._source_path,)
        if getattr(config, "_source_path", None) is not None
        else ()
    )
    core.validate_output_files(
        out_dir, ("report.md", "results.yaml"), protected_paths=protected
    )
    scenarios = select_scenarios(config.scenarios)
    interfaces = (
        list(core.INTERFACE_VARIANTS) if config.interfaces is None else config.interfaces
    )
    judge = build_judge(config.judge, variants=interfaces)
    readers = core.build_readers(config.readers)
    variants = list(core.REFERENCE_VARIANTS) + interfaces
    results = read_all(
        scenarios,
        readers,
        variants=variants,
        skip_render=config.skip_render,
        out_dir=out_dir,
        replicates=config.replicates,
    )
    book = ScoreBook(results)
    h1 = analyze_h1(book, scenarios, judge)
    paths = write_run(
        out_dir,
        scenarios,
        results,
        book,
        h1,
        config.skip_render,
        study=config.to_meta(),
        protected_paths=protected,
    )
    return StudyResult(config=config, book=book, h1=h1, paths=paths)


# Report

SIMULATED_CAVEAT = (
    "> **Simulated readers only.** Scores follow fixture rules and do not measure "
    "model or human comprehension."
)


def _report_values(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _report_values(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_report_values(v) for v in obj]
    return obj


def _dump(payload: Any, fh) -> None:
    fh.write(core.encode_yaml(_report_values(payload)))


def run_metadata(
    scenarios: Sequence[core.Scenario],
    reader_names: Sequence[str],
    skip_render: bool,
    study: Optional[Dict[str, Any]] = None,
    *,
    book: Optional[ScoreBook] = None,
    h1: Optional[H1Analysis] = None,
) -> Dict[str, Any]:
    meta = {
        "glassbox_version": core.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "renderer": core.renderer_version(skip=skip_render),
        "readers": list(reader_names),
        "scenarios": {s.id: core.scenario_sha256(s) for s in scenarios},
        "scenario_hash_format": core.SCENARIO_HASH_FORMAT,
        "scenario_snapshots": {s.id: core.scenario_payload(s) for s in scenarios},
        "deterministic": h1.deterministic if h1 is not None else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    if book is not None:
        actual_variants = list(dict.fromkeys(row.variant for row in book.results))
        meta["resolved_selection"] = {
            "scenarios": [s.id for s in scenarios],
            "interfaces": [v for v in actual_variants if v in book.interface_variants()],
            "variants": actual_variants,
            "readers": list(reader_names),
        }
        meta["reader_identity"] = {
            name: {
                key: getattr(book._reader_rows[name], "reader_" + key)
                for key in ("provider", "model", "model_family", "family_resolved",
                            "family_basis", "simulated", "deterministic")
            }
            for name in book.readers
        }
        meta["all_readers_simulated"] = book.all_simulated()
        meta["any_readers_simulated"] = book.any_simulated()
        meta["mixed_readers"] = book.any_simulated() and book.any_real()
    if h1 is not None:
        meta["judge"] = asdict(h1.judge)
    if study is not None:
        meta["study"] = study
    return meta


def _pct(value: float) -> str:
    return "%+.0f pts" % (value * 100) if math.isfinite(value) else "undefined"


def build_results(
    scenarios: Sequence[core.Scenario],
    results: Sequence[QuestionResult],
    book: ScoreBook,
    h1: H1Analysis,
    skip_render: bool,
    study: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    scen_h1 = {s.scenario_id: s for s in h1.scenarios}
    for sid in book.scenarios:
        per_reader = {}
        for r in book.readers:
            per_reader[r] = {
                "ceiling": book.ceiling(r, sid),
                "baseline": book.baseline(r, sid),
                "variants": {
                    v: {
                        "accuracy": book.accuracy(r, sid, v),
                        "lift": book.lift(r, sid, v),
                        "ceiling_gap": book.ceiling_gap(r, sid, v),
                    }
                    for v in book.variants
                },
            }
        aggregate = {
            v: {
                "mean_lift": book.mean_lift(sid, v),
                "mean_accuracy": book.mean_accuracy(sid, v),
            }
            for v in book.variants
        }
        sh = scen_h1[sid]
        metrics[sid] = {
            "readers": per_reader,
            "aggregate": aggregate,
            "h1": {
                "spearman": sh.spearman,
                "reader_agreement": sh.reader_agreement,
                "rows": [row.__dict__ for row in sh.rows],
                "reversals": [reversal.__dict__ for reversal in sh.reversals],
            },
        }
    return {
        "meta": run_metadata(
            scenarios, book.readers, skip_render, study, book=book, h1=h1
        ),
        "per_question": [asdict(r) for r in results],
        "metrics": metrics,
        "h1_summary": {
            "mean_spearman": h1.mean_spearman,
            "divergence_found": h1.divergence_found,
            "simulated": h1.simulated,
            "any_reader_simulated": h1.any_reader_simulated,
            "mixed_readers": h1.mixed_readers,
            "judge": asdict(h1.judge),
            "deterministic": h1.deterministic,
        },
    }


def _table(headers: Sequence[str], alignment: str, rows: Iterable[Sequence[Any]]) -> List[str]:
    return [
        "| " + " | ".join(headers) + " |", alignment,
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]


def render_report_md(
    book: ScoreBook, h1: H1Analysis
) -> str:
    target = h1.judge.score_label
    paragraphs = [
        "# GlassBox: MCQ lift vs %s" % target,
        "Readers: " + ", ".join("`%s`" % reader for reader in book.readers),
    ]
    if h1.simulated:
        paragraphs.append(SIMULATED_CAVEAT)
    elif h1.any_reader_simulated:
        paragraphs.append("> **Mixed readers.** Some scores come from synthetic fixtures.")
    judge_lines = [
        "Judge: `%s`; target **%s**; modality `%s`; protocol `%s`; comparisons %d."
        % (h1.judge.name, target, h1.judge.modality, h1.judge.protocol, h1.judge.comparison_count)
    ]
    if h1.judge.evidence:
        judge_lines.extend([
            "Judgments: %d valid (including %d ties), %d invalid, %d abstained."
            % (h1.judge.valid_count, h1.judge.tie_count,
               h1.judge.invalid_count, h1.judge.abstained_count),
            "Invalid and abstained replies receive no credit and are excluded from rate denominators.",
        ])
    if h1.judge.simulated:
        judge_lines.append(
            "> **Synthetic judge.** These scores use fixed fixture rules."
        )
    judge_lines.append("Declared deterministic: **%s**." % ("yes" if h1.deterministic else "no"))
    if h1.divergence_found:
        ranking_status = "a reversal is present"
    elif any(math.isfinite(row.pref_score) for scenario in h1.scenarios for row in scenario.rows):
        ranking_status = "no reversal among scored interfaces"
    else:
        ranking_status = "judge scores unavailable"
    paragraphs.extend([
        "\n".join(judge_lines),
        "Judge score and MCQ lift: **%s**; mean Spearman %s across %d scenarios."
        % (ranking_status, _corr(h1.mean_spearman), len(h1.scenarios)),
    ])
    if any(not math.isfinite(row.pref_score) for scenario in h1.scenarios for row in scenario.rows):
        paragraphs.append(
            "Judge scores with no valid comparisons are undefined. Ranks and correlations use finite scores only."
        )
    for scenario in h1.scenarios:
        paragraphs.extend([
            "## %s" % scenario.scenario_id,
            "Spearman(judge, lift) = **%s**; reader agreement = **%s**."
            % (_corr(scenario.spearman), _corr(scenario.reader_agreement)),
            "\n".join(_table(
                ["interface", "MCQ lift", "accuracy", target, "lift rank", "judge rank"],
                "|---|---:|---:|---:|:---:|:---:|",
                ([row.variant, _pct(row.comp_lift), _percent(row.mean_accuracy), _corr(row.pref_score),
                  row.comp_rank if row.comp_rank is not None else "undefined",
                  row.pref_rank if row.pref_rank is not None else "undefined"]
                 for row in sorted(scenario.rows, key=lambda row: (row.comp_rank is None, row.comp_rank or 0))),
            )),
        ])
        if scenario.reversals:
            paragraphs.append("\n".join(
                "- **Reversal:** `%s` scores above `%s` on %s (+%.2f), with lower MCQ lift (%s)."
                % (reversal.preferred, reversal.understood, target,
                   reversal.pref_gap, _pct(-reversal.comp_gap))
                for reversal in scenario.reversals
            ))
        else:
            scored = sum(math.isfinite(row.pref_score) and math.isfinite(row.comp_lift)
                         for row in scenario.rows)
            paragraphs.append("No reversed pair among scored interfaces." if scored >= 2
                              else "Too few scored interfaces to compare rankings.")
    if h1.judge.evidence:
        paragraphs.extend([
            "## Judge coverage",
            "Rates use valid comparisons only; coverage can differ after invalid or abstained replies.",
            "\n".join(_table(
                ["scenario", "interface", "credit", "valid", "planned"], "|---|---|---:|---:|---:|",
                ([evidence.scenario_id, score.variant, "%.2f" % score.credit,
                  score.denominator, score.planned_count]
                 for evidence in h1.judge.evidence for score in evidence.variant_scores),
            )),
            "A/B order, outcomes and raw judgments are in `results.yaml` under `meta.judge.evidence`.",
        ])
    return "\n\n".join(paragraphs) + "\n"


def _write_pair(paths: Sequence[str], markdown: str, payload: Any) -> None:
    with open(paths[0], "w", encoding="utf-8") as stream:
        stream.write(markdown)
    with open(paths[1], "w", encoding="utf-8") as stream:
        _dump(payload, stream)


def write_run(
    out_dir: str,
    scenarios: Sequence[core.Scenario],
    results: Sequence[QuestionResult],
    book: ScoreBook,
    h1: H1Analysis,
    skip_render: bool,
    study: Optional[Dict[str, Any]] = None,
    *,
    protected_paths: Sequence[str] = (),
) -> Dict[str, str]:
    paths = _destinations(out_dir, ("report.md", "results.yaml"), protected_paths)
    _write_pair(
        paths, render_report_md(book, h1),
        build_results(scenarios, results, book, h1, skip_render, study),
    )
    return dict(zip(("report", "results"), paths))


def render_transcript_md(
    result: RepairResult, scenario: core.Scenario, question: core.Question
) -> str:
    correct_text = next(choice.text for choice in question.choices if choice.id == question.answer)
    paragraphs = [
        "# Repair transcript: answer accuracy",
        "Scenario `%s` · reader `%s` · question `%s`"
        % (result.scenario_id, result.reader, result.question_id),
        "> %s" % question.stem,
        "Correct answer: **%s**" % correct_text,
    ]
    if result.reader.startswith("simulated:"):
        paragraphs.append(SIMULATED_CAVEAT)
    for turn in result.turns:
        chosen = next((choice.text for choice in question.choices if choice.id == turn.chosen), "unknown")
        header = "**Turn %d**: interface `%s` (%s)" % (turn.turn, turn.variant, ", ".join(turn.features))
        if turn.transform:
            header += "; repair: *%s*" % turn.transform
        paragraphs.extend([header, "- reader chose: %s → %s"
                           % (chosen, "correct" if turn.correct else "incorrect")])
    paragraphs.append(
        "**Result:** correct answer after **%d** repair turn(s)." % result.turns_to_understanding
        if result.converged else "**Result:** no correct answer within the turn budget."
    )
    return "\n\n".join(paragraphs)


def write_repair(
    out_dir: str,
    result: RepairResult,
    scenario: core.Scenario,
    question: core.Question,
    *,
    protected_paths: Sequence[str] = (),
) -> Dict[str, str]:
    paths = _destinations(out_dir, ("transcript.md", "repair.yaml"), protected_paths)
    payload = {
        "scenario": result.scenario_id,
        "reader": result.reader,
        "question": result.question_id,
        "converged": result.converged,
        "turns_to_understanding": result.turns_to_understanding,
        "turns": [asdict(turn) for turn in result.turns],
    }
    _write_pair(paths, render_transcript_md(result, scenario, question), payload)
    return dict(zip(("transcript", "repair"), paths))


ANCHOR_FIXTURE_CAVEAT = (
    "> **Fixture anchor set.** Human accuracy values are synthetic. Measured "
    "comparisons need matching questions and presentation conditions, with a "
    "documented source."
)


def _percent(value: float) -> str:
    return "%.0f%%" % (value * 100) if math.isfinite(value) else "undefined"


def _corr(value: float) -> str:
    return "undefined" if not math.isfinite(value) else "%.2f" % value


def render_anchor_md(result: AnchorResult) -> str:
    anchor = result.anchor
    lines = [
        "# GlassBox: model and human accuracy", "",
        "Anchor set: `%s`; %s" % (anchor.id, anchor.source),
        "Readers: " + ", ".join("`%s`" % reader for reader in result.readers),
        "Condition: readers see the `%s` presentation." % anchor.condition, "",
    ]
    if anchor.is_fixture:
        lines.extend([ANCHOR_FIXTURE_CAVEAT, ""])
    if result.readers and all(reader.startswith("simulated:") for reader in result.readers):
        lines.extend([SIMULATED_CAVEAT, ""])
    lines.extend([
        "**Correlation (model vs human accuracy, %d items):** Spearman %s · Pearson %s · Kendall %s."
        % (result.n_items, _corr(result.spearman), _corr(result.pearson), _corr(result.kendall)), "",
    ])
    lines.extend(_table(
        ["item", "human", "model", "gap (percentage points)"], "|---|---:|---:|---:|",
        (["%s/%s" % (point.scenario, point.question),
          _percent(point.human_accuracy), _percent(point.model_accuracy),
          "%.0f" % (point.abs_gap * 100)] for point in result.points),
    ))
    lines.extend(["", "**Largest accuracy gaps:**", ""])
    lines.extend(
        "- `%s/%s`: human %.0f%% vs model %.0f%% (gap %.0f pts)"
        % (point.scenario, point.question, point.human_accuracy * 100,
           point.model_accuracy * 100, point.abs_gap * 100)
        for point in result.breaks()
    )
    return "\n".join([*lines, ""])


def write_anchor(
    out_dir: str, result: AnchorResult, *, protected_paths: Sequence[str] = ()
) -> Dict[str, str]:
    paths = _destinations(out_dir, ("anchor.md", "anchor.yaml"), protected_paths)
    payload = {
        "anchor_id": result.anchor.id,
        "source": result.anchor.source,
        "is_fixture": result.anchor.is_fixture,
        "condition": result.anchor.condition,
        "readers": result.readers,
        "correlations": {
            "spearman": result.spearman,
            "pearson": result.pearson,
            "kendall": result.kendall,
        },
        "points": [asdict(point) for point in result.points],
    }
    _write_pair(paths, render_anchor_md(result), payload)
    return dict(zip(("report", "anchor"), paths))


OPTIMIZE_CAVEAT = (
    "> **In-sample feature search.** Candidate selection and reported scores use the "
    "same scenarios, questions and readers. A fresh plain-text baseline is sampled "
    "for every candidate and the cards comparison. There is no held-out evaluation "
    "or trained generator result. Restricted features do not prevent answer leakage or reward gaming."
)


def render_optimize_md(
    ev: EvalResult, readers: Sequence[str]
) -> str:
    paragraphs = [
        "# GlassBox: in-sample feature search",
        "Readers scoring the reward: " + ", ".join("`%s`" % reader for reader in readers),
        OPTIMIZE_CAVEAT,
    ]
    if ev.all_readers_simulated:
        paragraphs.append(SIMULATED_CAVEAT)
    elif ev.any_reader_simulated:
        paragraphs.append("> **Mixed readers.** Some rewards come from synthetic fixtures.")
    paragraphs.extend([
        "Protocol `%s`; %d candidates; %d scenarios; %d readers; baseline `%s`."
        % (ev.selection_protocol, ev.candidate_count, ev.scenario_count,
           ev.reader_count, ev.baseline_sampling),
        "**Selected features:** `%s`" % (", ".join(sorted(ev.generator_features)) or "(none)"),
        "\n".join(_table(
            ["interface", "comprehension lift"], "|---|---:|",
            [("selected interface", _pct(ev.generator_reward)),
             ("fixed polished cards", _pct(ev.cards_reward)), ("plain-text baseline", "+0 pts")],
        )),
        "Higher in-sample lift than fixed cards: **%s**; positive measured plain-text lift: **%s**."
        % ("yes" if ev.beats_preference_tuned else "no", "yes" if ev.beats_plaintext else "no"),
    ])
    return "\n\n".join(paragraphs) + "\n"


def write_optimize(
    out_dir: str,
    ev: EvalResult,
    readers: Sequence[str],
    *,
    protected_paths: Sequence[str] = (),
    selection: Optional[SelectionArtifact] = None,
) -> Dict[str, str]:
    names = ("optimize.md", "optimize.yaml")
    if selection is not None:
        names += ("selection.yaml",)
    destinations = _destinations(out_dir, names, protected_paths)
    markdown = render_optimize_md(ev, readers)
    if selection is not None:
        write_selection(destinations[2], selection)
        markdown += "\nFrozen selection: [selection.yaml](selection.yaml). Evaluate it on separate scenarios.\n"
    payload = {name: getattr(ev, name) for name in EvalResult.__dataclass_fields__
               if name != "selection_inputs_sha256"}
    payload.update(
        generator_features=sorted(ev.generator_features), readers=list(readers),
        selection_sha256=selection.sha256 if selection is not None else None,
        selection_file="selection.yaml" if selection is not None else None,
    )
    _write_pair(destinations, markdown, payload)
    return dict(zip(("report", "optimize", "selection"), destinations))


def render_evaluation_md(result: HeldOutResult) -> str:
    paragraphs = [
        "# GlassBox: frozen feature evaluation",
        "\n".join([
            "Selection scenarios: " + ", ".join(result.selection_scenario_ids),
            "Evaluation scenarios: " + ", ".join(s.scenario_id for s in result.scenarios),
            "Readers: " + ", ".join(result.reader_names),
            "Selection SHA-256: `%s`" % result.selection_sha256,
            "Protocol: `%s`; baseline: `%s`; skip render: `%s`."
            % (result.evaluation_protocol, result.baseline_sampling, result.skip_render),
        ]),
        "Features were frozen before evaluation. Scenario IDs and task content are disjoint from the recorded selection set.\n"
        "This checks recorded selection history; it cannot verify access outside this workflow.",
    ]
    if result.all_readers_simulated:
        paragraphs.append(SIMULATED_CAVEAT)
    elif result.any_reader_simulated:
        paragraphs.append("> **Mixed readers.** Some scores come from synthetic fixtures.")
    paragraphs.extend([
        "Selected features: " + (", ".join(sorted(result.selected_features)) or "(none)")
        + "\nSelection-set lift: %s (used for selection, not held-out)." % _pct(result.selection_reward),
        "\n".join(_table(
            ["evaluation scenario", "questions", "selected accuracy", "cards accuracy",
             "baseline accuracy", "selected lift", "cards lift"],
            "|---|---:|---:|---:|---:|---:|---:|",
            ([scenario.scenario_id, scenario.question_count, _percent(scenario.selected_accuracy),
              _percent(scenario.cards_accuracy), _percent(scenario.baseline_accuracy),
              _pct(scenario.selected_lift), _pct(scenario.cards_lift)] for scenario in result.scenarios),
        )),
        "Mean held-out lift: selected %s; cards %s.\n"
        % (_pct(result.aggregate.selected_lift), _pct(result.aggregate.cards_lift))
        + "Scenarios receive equal weight. The baseline is shared by the fixed selected and cards views within each evaluation scenario.\n"
        "Scores describe these scenarios and readers; they do not establish population generalization.",
    ])
    return "\n\n".join(paragraphs) + "\n"


def write_evaluation(
    out_dir: str,
    result: HeldOutResult,
    selection: SelectionArtifact,
    *,
    protected_paths: Sequence[str] = (),
) -> Dict[str, str]:
    selection = validate_selection(selection)
    if (
        result.selection_sha256 != selection.sha256
        or result.selected_features != selection.features
        or result.selection_scenario_ids != selection.scenario_ids
        or result.selection_reward != selection.selection_reward
    ):
        raise ValueError("evaluation result does not match the frozen selection")
    if selection.source_path is not None:
        protected_paths = tuple(protected_paths) + (selection.source_path,)
    report_path, data_path = _destinations(
        out_dir, ("evaluation.md", "evaluation.yaml"), protected_paths
    )
    payload = asdict(result)
    payload["selected_features"] = sorted(result.selected_features)
    payload["selection"] = selection_payload(selection)
    _write_pair((report_path, data_path), render_evaluation_md(result), payload)
    return {"report": report_path, "evaluation": data_path}


def _destinations(
    out_dir: str, names: Sequence[str], protected_paths: Sequence[str]
) -> List[str]:
    paths = core.validate_output_files(out_dir, names, protected_paths=protected_paths)
    os.makedirs(paths[0].parent, exist_ok=True)
    return [str(path) for path in paths]
