"""In-sample feature search using mean MCQ accuracy minus plain-text accuracy."""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Tuple

from . import interfaces, render
from .interfaces import Presentation, VARIANT_FEATURES
from .readers.base import Reader
from .schema import Scenario
from .families import canonical_family, model_family

GENERATOR_FEATURE_POOL = ["detail", "derived", "highlight", "sorted", "polished"]
BASELINE_VARIANT = "baseline"
CARDS_VARIANT = "cards"


def _accuracy(
    scenario: Scenario, presentation: Presentation, readers: Sequence[Reader], skip_render: bool
) -> float:
    stimulus = render.render(presentation, skip=skip_render)
    accuracies = []
    for reader in readers:
        correct = sum(
            reader.answer(scenario, question, stimulus).choice_id == question.answer
            for question in scenario.questions
        )
        accuracies.append(correct / len(scenario.questions))
    return statistics.fmean(accuracies)


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
        interface_reward(scenario, features, readers, skip_render) for scenario in scenarios
    )


def _subsets(pool: Sequence[str]):
    for size in range(len(pool) + 1):
        for combination in itertools.combinations(pool, size):
            yield frozenset(combination)


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
        any_reader_simulated=any(reader.simulated for reader in readers),
        all_readers_simulated=all(reader.simulated for reader in readers),
    )


def pool_excludes_family(reader_specs: Sequence[str], generator_family: str) -> bool:
    """Require known, distinct model families; this is a scoped exclusion check."""
    generator_family = canonical_family(generator_family)
    families = [model_family(spec) for spec in reader_specs]
    return (
        bool(families)
        and generator_family != "unknown"
        and all(family != "unknown" and family != generator_family for family in families)
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
    scenarios: Sequence[Scenario], readers: Sequence[Reader], skip_render: bool
) -> None:
    if not isinstance(scenarios, Sequence) or not scenarios or any(
        not isinstance(scenario, Scenario) or not scenario.questions for scenario in scenarios
    ):
        raise ValueError("reward needs scenarios with questions")
    if not isinstance(readers, Sequence) or not readers or any(
        not isinstance(reader, Reader) for reader in readers
    ):
        raise ValueError("reward needs readers")
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
