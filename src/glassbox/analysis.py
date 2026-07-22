"""H1 analysis — do preference and comprehension come apart?

For each scenario, rank the interface variants two ways: by **comprehension** (mean
lift across readers) and by **preference** (the judge). A rank correlation below 1 and,
more concretely, any *reversal* — an interface preferred over another yet understood
less — is the H1 finding. The canonical one is the polished cards being preferred over
the plain table while transferring less understanding.

Spearman is computed in the standard library; tests cross-check it against scipy when
available. When every reader is simulated, the result is labeled a pipeline
demonstration, not evidence.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Sequence

from .judge import Judge, judge_all
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
    n = len(a)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return float("nan")
    return cov / ((va ** 0.5) * (vb ** 0.5))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation, tie-corrected (Pearson on fractional ranks)."""
    if len(xs) != len(ys):
        raise ValueError("spearman needs equal-length sequences")
    if len(xs) < 2:
        return float("nan")
    return _pearson(_fractional_ranks(xs), _fractional_ranks(ys))


# --- H1 structures ----------------------------------------------------------

@dataclass(frozen=True)
class InterfaceRow:
    variant: str
    comp_lift: float       # mean comprehension lift across readers
    mean_accuracy: float
    pref_score: float
    comp_rank: int         # 1 = best understood
    pref_rank: int         # 1 = most preferred


@dataclass(frozen=True)
class Reversal:
    preferred: str         # variant preferred more ...
    understood: str        # ... yet understood less than this one
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

    @property
    def divergence_found(self) -> bool:
        return any(s.reversals for s in self.scenarios)


def _rank_desc(scores: List[float]) -> List[int]:
    """Dense-ish competition ranks, 1 = highest score (ties share a rank)."""
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


def analyze_h1(book: ScoreBook, scenarios: Sequence[Scenario], judge: Judge) -> H1Analysis:
    """Build the H1 divergence analysis from scored readings + judge preferences."""
    variants = book.interface_variants()
    scenario_reports: List[ScenarioH1] = []

    for scenario in scenarios:
        sid = scenario.id
        presentations = [interfaces.variant(scenario, v) for v in variants]
        prefs = {p.variant: p.score for p in judge_all(judge, scenario, presentations)}

        comp = [book.mean_lift(sid, v) for v in variants]
        pref = [prefs[v] for v in variants]
        comp_ranks = _rank_desc(comp)
        pref_ranks = _rank_desc(pref)

        rows = [InterfaceRow(variant=v, comp_lift=comp[i],
                             mean_accuracy=book.mean_accuracy(sid, v),
                             pref_score=pref[i], comp_rank=comp_ranks[i],
                             pref_rank=pref_ranks[i])
                for i, v in enumerate(variants)]

        reversals: List[Reversal] = []
        for i in range(len(variants)):
            for j in range(len(variants)):
                if i == j:
                    continue
                if pref[i] > pref[j] and comp[i] < comp[j]:
                    reversals.append(Reversal(preferred=variants[i], understood=variants[j],
                                              pref_gap=pref[i] - pref[j],
                                              comp_gap=comp[j] - comp[i]))

        scenario_reports.append(ScenarioH1(
            scenario_id=sid, rows=rows, spearman=spearman(comp, pref),
            reader_agreement=_reader_agreement(book, sid, variants), reversals=reversals))

    valid_rhos = [s.spearman for s in scenario_reports if s.spearman == s.spearman]
    mean_rho = statistics.fmean(valid_rhos) if valid_rhos else float("nan")
    return H1Analysis(scenarios=scenario_reports, mean_spearman=mean_rho,
                      simulated=book.all_simulated(), readers=book.readers)


# --- cross-family analysis (the M2 headline: does divergence survive families?) ---

@dataclass(frozen=True)
class FamilyComprehension:
    family: str
    readers: List[str]
    comp_lift: Dict[str, float]   # interface variant -> mean lift within this family
    reversal_holds: bool          # some interface preferred over another is understood less


@dataclass(frozen=True)
class CrossFamilyAnalysis:
    families: List[FamilyComprehension]
    interface_variants: List[str]
    preference: Dict[str, float]        # variant -> mean preference (family-independent)
    cross_family_agreement: float       # mean pairwise Spearman of family comp-lift vectors
    survives_across_families: bool      # >=2 families and the reversal holds in every one

    @property
    def n_families(self) -> int:
        return len(self.families)


def _reversal_present(variants: List[str], pref: Dict[str, float],
                      comp: Dict[str, float]) -> bool:
    for i in variants:
        for j in variants:
            if i != j and pref[i] > pref[j] and comp[i] < comp[j]:
                return True
    return False


def analyze_cross_family(book: ScoreBook, scenarios: Sequence[Scenario],
                         judge: Judge) -> CrossFamilyAnalysis:
    """Group readers by API family and test whether the H1 reversal survives across them.

    Preference is family-independent (the judge scores interfaces, not readers); the
    comprehension ranking is computed per family, pooled across scenarios."""
    variants = book.interface_variants()

    # Mean preference per interface, averaged over scenarios (family-independent).
    pref_totals = {v: [] for v in variants}
    for scenario in scenarios:
        presentations = [interfaces.variant(scenario, v) for v in variants]
        for p in judge_all(judge, scenario, presentations):
            if p.variant in pref_totals:
                pref_totals[p.variant].append(p.score)
    preference = {v: statistics.fmean(pref_totals[v]) if pref_totals[v] else 0.0
                  for v in variants}

    families: List[FamilyComprehension] = []
    for fam in book.families():
        fam_readers = book.readers_in_family(fam)
        comp = {}
        for v in variants:
            lifts = [book.lift(r, s.id, v) for r in fam_readers for s in scenarios]
            comp[v] = statistics.fmean(lifts) if lifts else 0.0
        families.append(FamilyComprehension(
            family=fam, readers=fam_readers, comp_lift=comp,
            reversal_holds=_reversal_present(variants, preference, comp)))

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

    survives = (len(families) >= 2 and all(f.reversal_holds for f in families))
    return CrossFamilyAnalysis(families=families, interface_variants=variants,
                               preference=preference, cross_family_agreement=agreement,
                               survives_across_families=survives)
