"""Compare judge scores with measured MCQ lift; report fixture status separately."""

from __future__ import annotations

import statistics
import math
from numbers import Real
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .judge import Judge, JudgeMetadata, judge_all
from .scoring import ScoreBook
from . import interfaces
from .schema import Scenario


# --- rank correlation (stdlib) ----------------------------------------------


def _fractional_ranks(values: Sequence[float]) -> List[float]:
    """1-based ranks with ties averaged (ascending)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
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
            vector.append(value)
        result.append(vector)
    return result[0], result[1]


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation, tie-corrected (Pearson on fractional ranks)."""
    xs, ys = _vectors(xs, ys)
    if len(xs) < 2:
        return float("nan")
    return _pearson(_fractional_ranks(xs), _fractional_ranks(ys))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation."""
    xs, ys = _vectors(xs, ys)
    if len(xs) < 2:
        return float("nan")
    return _pearson(xs, ys)


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Kendall's tau-b rank correlation (ties handled; matches scipy's default)."""
    xs, ys = _vectors(xs, ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    concordant = discordant = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
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


# --- H1 structures ----------------------------------------------------------


@dataclass(frozen=True)
class InterfaceRow:
    variant: str
    comp_lift: float  # mean comprehension lift across readers
    mean_accuracy: float
    pref_score: float
    comp_rank: int  # 1 = best understood
    pref_rank: int  # 1 = most preferred


@dataclass(frozen=True)
class Reversal:
    preferred: str  # variant preferred more ...
    understood: str  # ... yet understood less than this one
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


def _rank_desc(scores: List[float]) -> List[int]:
    """Competition ranks, 1 = highest; ties share a rank."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = [0] * len(scores)
    last_score = None
    last_rank = 0
    for pos, idx in enumerate(order, start=1):
        if last_score is None or scores[idx] != last_score:
            last_rank = pos
            last_score = scores[idx]
        ranks[idx] = last_rank
    return ranks


def _reader_agreement(book: ScoreBook, scenario: str, variants: List[str]) -> float:
    """Mean pairwise Spearman of readers' per-interface accuracy vectors."""
    readers = book.readers
    if len(readers) < 2 or len(variants) < 2:
        return float("nan")
    vecs = {r: [book.accuracy(r, scenario, v) for v in variants] for r in readers}
    corrs = []
    for i in range(len(readers)):
        for j in range(i + 1, len(readers)):
            rho = spearman(vecs[readers[i]], vecs[readers[j]])
            if rho == rho:  # skip nan (a constant vector)
                corrs.append(rho)
    return statistics.fmean(corrs) if corrs else float("nan")


def analyze_h1(
    book: ScoreBook, scenarios: Sequence[Scenario], judge: Judge
) -> H1Analysis:
    """Rank MCQ lift and the declared judge target separately."""
    variants = book.interface_variants()
    if not variants or not scenarios or not book.readers:
        raise ValueError("H1 needs scored interfaces, scenarios and readers")
    scenario_reports: List[ScenarioH1] = []

    for scenario in scenarios:
        sid = scenario.id
        presentations = [interfaces.variant(scenario, v) for v in variants]
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

        reversals: List[Reversal] = []
        for i in range(len(variants)):
            for j in range(len(variants)):
                if i == j:
                    continue
                if pref[i] > pref[j] and comp[i] < comp[j]:
                    reversals.append(
                        Reversal(
                            preferred=variants[i],
                            understood=variants[j],
                            pref_gap=pref[i] - pref[j],
                            comp_gap=comp[j] - comp[i],
                        )
                    )

        scenario_reports.append(
            ScenarioH1(
                scenario_id=sid,
                rows=rows,
                spearman=spearman(comp, pref),
                reader_agreement=_reader_agreement(book, sid, variants),
                reversals=reversals,
            )
        )

    valid_rhos = [s.spearman for s in scenario_reports if s.spearman == s.spearman]
    mean_rho = statistics.fmean(valid_rhos) if valid_rhos else float("nan")
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


# --- resolved model-family analysis ---


@dataclass(frozen=True)
class FamilyComprehension:
    family: str
    readers: List[str]
    comp_lift: Dict[str, float]  # interface variant -> mean lift within this family
    reversal_holds: bool  # some interface preferred over another is understood less
    reversals: List[Tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class CrossFamilyAnalysis:
    families: List[FamilyComprehension]
    interface_variants: List[str]
    preference: Dict[str, float]  # variant -> mean preference (family-independent)
    cross_family_agreement: float  # mean pairwise Spearman of family comp-lift vectors
    survives_across_families: bool  # >=2 families and the reversal holds in every one
    unresolved_readers: List[str] = field(default_factory=list)
    common_reversals: List[Tuple[str, str]] = field(default_factory=list)
    judge_target: str = "unknown"

    @property
    def n_families(self) -> int:
        return len(self.families)


def _reversal_present(
    variants: List[str], pref: Dict[str, float], comp: Dict[str, float]
) -> bool:
    return bool(_reversal_pairs(variants, pref, comp))


def _reversal_pairs(
    variants: List[str], pref: Dict[str, float], comp: Dict[str, float]
) -> List[Tuple[str, str]]:
    return [
        (i, j)
        for i in variants
        for j in variants
        if i != j and pref[i] > pref[j] and comp[i] < comp[j]
    ]


def analyze_cross_family(
    book: ScoreBook, scenarios: Sequence[Scenario], judge: Judge
) -> CrossFamilyAnalysis:
    """Pool lift within resolved model families; compare each with the same judge."""
    variants = book.interface_variants()

    # Mean judge score over scenarios.
    pref_totals = {v: [] for v in variants}
    for scenario in scenarios:
        presentations = [interfaces.variant(scenario, v) for v in variants]
        for p in judge_all(judge, scenario, presentations):
            if p.variant in pref_totals:
                pref_totals[p.variant].append(p.score)
    preference = {
        v: statistics.fmean(pref_totals[v]) if pref_totals[v] else 0.0 for v in variants
    }

    families: List[FamilyComprehension] = []
    for fam in book.families():
        fam_readers = book.readers_in_family(fam)
        comp = {}
        for v in variants:
            lifts = [book.lift(r, s.id, v) for r in fam_readers for s in scenarios]
            comp[v] = statistics.fmean(lifts) if lifts else 0.0
        families.append(
            FamilyComprehension(
                family=fam,
                readers=fam_readers,
                comp_lift=comp,
                reversal_holds=_reversal_present(variants, preference, comp),
                reversals=_reversal_pairs(variants, preference, comp),
            )
        )

    # Cross-family agreement: do families rank interfaces the same way by comprehension?
    corrs = []
    for a in range(len(families)):
        for b in range(a + 1, len(families)):
            va = [families[a].comp_lift[v] for v in variants]
            vb = [families[b].comp_lift[v] for v in variants]
            rho = spearman(va, vb)
            if rho == rho:  # skip nan (a constant vector)
                corrs.append(rho)
    agreement = statistics.fmean(corrs) if corrs else float("nan")

    survives = len(families) >= 2 and all(f.reversal_holds for f in families)
    common = (
        sorted(set.intersection(*(set(f.reversals) for f in families)))
        if len(families) >= 2
        else []
    )
    return CrossFamilyAnalysis(
        families=families,
        interface_variants=variants,
        preference=preference,
        cross_family_agreement=agreement,
        survives_across_families=survives,
        unresolved_readers=book.unresolved_readers(),
        common_reversals=common,
        judge_target=judge.target,
    )
