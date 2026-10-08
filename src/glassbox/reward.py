"""In-sample feature search using mean MCQ accuracy minus plain-text accuracy."""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Tuple

from . import interfaces, render
from .interfaces import VARIANT_FEATURES
from .readers.base import Reader
from .schema import Scenario
from .families import canonical_family, model_family as _model_family

GENERATOR_FEATURE_POOL = ["detail", "derived", "highlight", "sorted", "polished"]
BASELINE_VARIANT = "baseline"
CARDS_VARIANT = "cards"


def _accuracy(
    scenario: Scenario, presentation, readers: Sequence[Reader], skip_render: bool
) -> float:
    stim = render.render(presentation, skip=skip_render)
    accs = []
    for r in readers:
        correct = sum(
            1
            for q in scenario.questions
            if r.answer(scenario, q, stim).choice_id == q.answer
        )
        accs.append(correct / len(scenario.questions))
    return statistics.fmean(accs)


def interface_reward(
    scenario: Scenario,
    features: FrozenSet[str],
    readers: Sequence[Reader],
    skip_render: bool = True,
) -> float:
    """Use a fresh plain-text baseline for this candidate."""
    _inputs([scenario], readers, skip_render)
    candidate = interfaces.build(scenario, frozenset(features), variant="candidate")
    baseline = interfaces.variant(scenario, BASELINE_VARIANT)
    return _accuracy(scenario, candidate, readers, skip_render) - _accuracy(
        scenario, baseline, readers, skip_render
    )


def scenario_set_reward(
    scenarios: Sequence[Scenario],
    features: FrozenSet[str],
    readers: Sequence[Reader],
    skip_render: bool = True,
) -> float:
    _inputs(scenarios, readers, skip_render)
    return statistics.fmean(
        interface_reward(s, features, readers, skip_render) for s in scenarios
    )


def _subsets(pool: Sequence[str]):
    for r in range(len(pool) + 1):
        for combo in itertools.combinations(pool, r):
            yield frozenset(combo)


def search_best_interface(
    scenarios: Sequence[Scenario],
    readers: Sequence[Reader],
    pool: Optional[Sequence[str]] = None,
    skip_render: bool = True,
) -> List[Tuple[FrozenSet[str], float]]:
    """Rank every feature-set candidate by comprehension reward (best first)."""
    _inputs(scenarios, readers, skip_render)
    pool = _pool(pool)
    scored = [
        (features, scenario_set_reward(scenarios, features, readers, skip_render))
        for features in _subsets(pool)
    ]
    scored.sort(key=lambda t: -t[1])
    return scored


@dataclass(frozen=True)
class EvalResult:
    generator_features: FrozenSet[str]
    generator_reward: float
    cards_reward: (
        float
    )
    beats_preference_tuned: bool
    beats_plaintext: bool
    candidate_count: int = 32
    scenario_count: int = 0
    reader_count: int = 0
    selection_protocol: str = "in_sample_exhaustive_feature_search.v1"
    baseline_sampling: str = "fresh_per_candidate_and_cards"
    any_reader_simulated: bool = False
    all_readers_simulated: bool = False


def evaluate_generator(
    scenarios: Sequence[Scenario],
    readers: Sequence[Reader],
    pool: Optional[Sequence[str]] = None,
    skip_render: bool = True,
) -> EvalResult:
    """Select and report the winner on the same scenarios and readers."""
    pool = _pool(pool)
    ranked = search_best_interface(
        scenarios, readers, pool=pool, skip_render=skip_render
    )
    best_features, best_reward = ranked[0]
    cards_reward = scenario_set_reward(
        scenarios, VARIANT_FEATURES[CARDS_VARIANT], readers, skip_render
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
        any_reader_simulated=any(r.simulated for r in readers),
        all_readers_simulated=all(r.simulated for r in readers),
    )


def model_family(reader_spec: str) -> str:
    """Compatibility wrapper for the shared family resolver."""
    return _model_family(reader_spec)


def pool_excludes_family(reader_specs: Sequence[str], generator_family: str) -> bool:
    """Require known, distinct model families; this is a scoped exclusion check."""
    gf = canonical_family(generator_family)
    families = [model_family(spec) for spec in reader_specs]
    return (
        bool(families)
        and gf != "unknown"
        and all(family != "unknown" and family != gf for family in families)
    )


def _pool(pool: Optional[Sequence[str]]) -> List[str]:
    if pool is None:
        return list(GENERATOR_FEATURE_POOL)
    if isinstance(pool, (str, bytes)):
        raise ValueError("feature pool must be a sequence")
    result = list(pool)
    if any(
        not isinstance(f, str) or f not in GENERATOR_FEATURE_POOL for f in result
    ) or len(set(result)) != len(result):
        raise ValueError("feature pool must contain distinct supported features")
    return result


def _inputs(
    scenarios: Sequence[Scenario], readers: Sequence[Reader], skip_render: bool
) -> None:
    if not scenarios or any(
        not isinstance(s, Scenario) or not s.questions for s in scenarios
    ):
        raise ValueError("reward needs scenarios with questions")
    if not readers or any(not isinstance(r, Reader) for r in readers):
        raise ValueError("reward needs readers")
    if not isinstance(skip_render, bool):
        raise ValueError("skip_render must be a boolean")
