"""M4 — the comprehension score as a reward, and an offline optimizer over it.

The score is a computation, so it doubles as a training reward: an interface's reward is
its comprehension lift (reader accuracy on it minus on the plain-text baseline), averaged
over readers and scenarios. ``search_best_interface`` is a non-ML stand-in for the
generator — it enumerates the interface feature-lattice and returns the feature sets that
maximize the reward, proving the signal is optimizable. The real generator (SFT -> GRPO on
a VLM) replaces the search with a trained model; the reward function is unchanged.

Anti-gaming: ``pool_excludes_family`` enforces that the reader pool never contains the
generator's own model family, so a generator cannot win by pleasing a sibling model.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Tuple

from . import interfaces, render
from .interfaces import VARIANT_FEATURES
from .readers.base import Reader
from .schema import Scenario

# The features a generator may toggle when composing an interface.
GENERATOR_FEATURE_POOL = ["detail", "derived", "highlight", "sorted", "polished"]
BASELINE_VARIANT = "baseline"
CARDS_VARIANT = "cards"


def _accuracy(scenario: Scenario, presentation, readers: Sequence[Reader],
              skip_render: bool) -> float:
    stim = render.render(presentation, skip=skip_render)
    accs = []
    for r in readers:
        correct = sum(1 for q in scenario.questions
                      if r.answer(scenario, q, stim).choice_id == q.answer)
        accs.append(correct / len(scenario.questions))
    return statistics.fmean(accs) if accs else 0.0


def interface_reward(scenario: Scenario, features: FrozenSet[str],
                     readers: Sequence[Reader], skip_render: bool = True) -> float:
    """Comprehension lift of an interface (feature set) on one scenario — the reward."""
    candidate = interfaces.build(scenario, frozenset(features), variant="candidate")
    baseline = interfaces.variant(scenario, BASELINE_VARIANT)
    return (_accuracy(scenario, candidate, readers, skip_render)
            - _accuracy(scenario, baseline, readers, skip_render))


def scenario_set_reward(scenarios: Sequence[Scenario], features: FrozenSet[str],
                        readers: Sequence[Reader], skip_render: bool = True) -> float:
    return statistics.fmean(
        interface_reward(s, features, readers, skip_render) for s in scenarios)


def _subsets(pool: Sequence[str]):
    for r in range(len(pool) + 1):
        for combo in itertools.combinations(pool, r):
            yield frozenset(combo)


def search_best_interface(scenarios: Sequence[Scenario], readers: Sequence[Reader],
                          pool: Optional[Sequence[str]] = None,
                          skip_render: bool = True) -> List[Tuple[FrozenSet[str], float]]:
    """Rank every feature-set candidate by comprehension reward (best first)."""
    pool = list(pool) if pool else GENERATOR_FEATURE_POOL
    scored = [(features, scenario_set_reward(scenarios, features, readers, skip_render))
              for features in _subsets(pool)]
    scored.sort(key=lambda t: -t[1])
    return scored


@dataclass(frozen=True)
class EvalResult:
    generator_features: FrozenSet[str]
    generator_reward: float          # comprehension lift of the optimized interface
    cards_reward: float              # comprehension lift of polished cards (preference-tuned analog)
    beats_preference_tuned: bool     # generator > cards on comprehension
    beats_plaintext: bool            # generator > 0 (baseline is the zero point)


def evaluate_generator(scenarios: Sequence[Scenario], readers: Sequence[Reader],
                       pool: Optional[Sequence[str]] = None,
                       skip_render: bool = True) -> EvalResult:
    """Evaluate the offline search generator against the preference-tuned and plain-text
    baselines on comprehension — the M4 eval, with the search standing in for a model."""
    ranked = search_best_interface(scenarios, readers, pool=pool, skip_render=skip_render)
    best_features, best_reward = ranked[0]
    cards_reward = scenario_set_reward(scenarios, VARIANT_FEATURES[CARDS_VARIANT],
                                       readers, skip_render)
    return EvalResult(
        generator_features=best_features, generator_reward=best_reward,
        cards_reward=cards_reward, beats_preference_tuned=best_reward > cards_reward,
        beats_plaintext=best_reward > 0.0)


# Map a reader spec to its underlying MODEL family (not its API family). Provider-locked
# APIs are unambiguous; aggregators (openrouter) are resolved from the model slug. This is
# a heuristic map — a production deployment would resolve against a real model registry.
_SLUG_FAMILIES = {
    "qwen": ("qwen",),
    "anthropic": ("claude",),
    "openai": ("gpt", "o1", "o3", "o4"),
    "llama": ("llama",),
    "gemini": ("gemini",),
    "mistral": ("mistral", "mixtral"),
    "deepseek": ("deepseek",),
}


def model_family(reader_spec: str) -> str:
    """The generating model's family for a reader spec, or 'unknown' if unrecognized."""
    provider, _, model = reader_spec.partition(":")
    provider, model = provider.lower(), model.lower()
    if provider in ("simulated", "anthropic", "openai"):
        return provider  # provider-locked: the API fixes the model family
    for family, tokens in _SLUG_FAMILIES.items():  # aggregator/bare slug: infer
        if any(t in model for t in tokens):
            return family
    return "unknown"


def pool_excludes_family(reader_specs: Sequence[str], generator_family: str) -> bool:
    """True if no reader resolves to the generator's model family (anti-gaming).

    A substring check fails open when a sibling model is spelled differently; this
    compares resolved families instead. Readers whose family cannot be resolved
    ('unknown') are not treated as excluded here — ``validate_training_config`` rejects a
    pool containing any unknown so the guard cannot silently pass an unrecognized slug."""
    gf = generator_family.lower()
    return all(model_family(spec) != gf for spec in reader_specs)
